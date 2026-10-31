"""Pin and verify B's public PRM without modifying A's assets or runtime."""
import hashlib
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from common import atomic_json, file_hash
from prepare_assets import fetch, get_json

MODEL = "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2"
REVISION = "bc44d4a24bf1f7453fc3892dec25c4669e54ede5"
WEIGHT_SHA = "9b26289b485cfd170bfd63d752e86da344b21cdc2a42545d5b172d9636d50d24"


def fetch_weights(url, target, total, expected):
    """Bounded range transfers; verify ranges, assembled size and the LFS hash."""
    if target.exists():
        assert file_hash(target) == expected
        return
    chunk_size = 4 * 1024 * 1024
    parts = target.parent / (target.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    count = (total + chunk_size - 1) // chunk_size
    started = time.time()

    def download(i):
        start, end = i * chunk_size, min(total, (i + 1) * chunk_size) - 1
        part = parts / f"{i:05d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        partial = part.with_suffix(".partial")
        headers = part.with_suffix(".headers")
        for attempt in range(4):
            result = subprocess.run(["curl", "--noproxy", "*", "--fail", "--location", "--silent", "--show-error",
                "--range", f"{start}-{end}", "--connect-timeout", "15", "--max-time", "120",
                "--speed-limit", "1024", "--speed-time", "30", "--dump-header", str(headers),
                "--output", str(partial), f"{url}?download=true&part={i}&attempt={attempt}"],
                text=True, capture_output=True, timeout=135)
            ranges = re.findall(r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", headers.read_text()) if headers.exists() else []
            correct_range = ranges and tuple(map(int, ranges[-1])) == (start, end, total)
            if result.returncode == 0 and correct_range and partial.stat().st_size == end - start + 1:
                partial.replace(part)
                return part
            print(f"Retry part {i} attempt {attempt + 1}: exit={result.returncode}, valid_range={bool(correct_range)}", flush=True)
            time.sleep(1)
        raise RuntimeError(f"Bounded download retries exhausted for weight part {i}")

    completed, downloaded = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, i) for i in range(count)]
        for future in concurrent.futures.as_completed(futures):
            part = future.result()
            completed += 1
            downloaded += part.stat().st_size
            elapsed = time.time() - started
            atomic_json(ROOT / "receipts/B_prm_download_progress.json", {
                "status": "downloading", "file": target.name, "completed_parts": completed, "total_parts": count,
                "verified_range_bytes": downloaded, "total_bytes": total, "elapsed_seconds": elapsed,
                "average_bytes_per_second": downloaded / max(elapsed, 1), "heartbeat_unix": time.time()})
            if completed % 16 == 0:
                print(f"Verified byte ranges: {completed}/{count}, {downloaded}/{total} bytes", flush=True)
    staging = target.with_name(target.name + ".assembled.partial")
    with staging.open("wb") as output:
        for i in range(count):
            with (parts / f"{i:05d}.part").open("rb") as stream:
                for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    output.write(data)
    assert staging.stat().st_size == total
    assert file_hash(staging) == expected, "Assembled PRM weights fail upstream LFS SHA256"
    staging.replace(target)
    atomic_json(ROOT / "receipts/B_prm_download_progress.json", {
        "status": "completed_and_sha256_verified", "file": target.name, "total_bytes": total, "sha256": expected,
        "finished_unix": time.time(), "elapsed_seconds": time.time() - started})


def main():
    endpoint = "https://hf-mirror.com"
    info = get_json(f"{endpoint}/api/models/{MODEL}/revision/{REVISION}?blobs=true")
    assert info["sha"] == REVISION
    dest = ROOT / "models" / "H4-PRM-1.5B-0.2" / REVISION
    receipt = ROOT / "receipts/B_prm_assets.json"
    state = {"model_id": MODEL, "model_revision": REVISION, "model_path": str(dest),
             "endpoint": endpoint, "files": {}, "started_unix": time.time(), "status": "downloading"}
    atomic_json(receipt, state)
    selected = {"config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
                "added_tokens.json", "special_tokens_map.json", "model.safetensors", "README.md"}
    for entry in sorted(info["siblings"], key=lambda e: e["rfilename"] == "model.safetensors"):
        name = entry["rfilename"]
        if name not in selected:
            continue
        expected = entry.get("lfs", {}).get("sha256")
        if name == "model.safetensors":
            assert expected == WEIGHT_SHA
        target = dest / name
        print("Downloading", name, flush=True)
        url = f"{endpoint}/{MODEL}/resolve/{REVISION}/{name}"
        if expected and entry["size"] > 4 * 1024 * 1024:
            fetch_weights(url, target, entry["size"], expected)
        else:
            fetch(url, target, expected)
        if not expected:
            raw = target.read_bytes()
            assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == entry["blobId"]
        state["files"][name] = {"sha256": file_hash(target), "bytes": target.stat().st_size,
                                "lfs_sha256": expected, "git_blob_sha1": entry.get("blobId")}
        atomic_json(receipt, state)
    assert selected == state["files"].keys()
    state.update(status="completed", finished_unix=time.time())
    atomic_json(receipt, state)
    print(json.dumps({"status": state["status"], "model_revision": REVISION}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        receipt = ROOT / "receipts/B_prm_assets.json"
        state = json.loads(receipt.read_text()) if receipt.exists() else {}
        state.update(status="failed", error=str(exc), finished_unix=time.time())
        atomic_json(receipt, state)
        raise
