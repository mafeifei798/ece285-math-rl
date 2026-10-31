"""Complete B's pinned public asset via existing CDN redirects, avoiding API rate limits.

Operational download recovery only. Frozen B training sources are not changed.
"""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from common import atomic_json, file_hash

SIZE = 3087473468
SHA = "9b26289b485cfd170bfd63d752e86da344b21cdc2a42545d5b172d9636d50d24"
CHUNK = 4 * 1024 * 1024


def main():
    receipt = ROOT / "receipts/B_prm_assets.json"
    state = json.loads(receipt.read_text())
    assert state["model_revision"] == "bc44d4a24bf1f7453fc3892dec25c4669e54ede5"
    dest = Path(state["model_path"])
    lock = (dest / "recovery.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parts = dest / "model.safetensors.parts"
    # Keep signed public CDN URLs internal; only provenance host is recorded.
    url = None
    for path in sorted(parts.glob("*.headers"), key=lambda p: p.stat().st_mtime, reverse=True):
        locations = re.findall(r"(?im)^location:\s*(\S+)", path.read_text())
        if locations:
            parsed = urllib.parse.urlsplit(locations[-1])
            if parsed.hostname == "cas-bridge.xethub.hf.co":
                expires = int(urllib.parse.parse_qs(parsed.query)["Expires"][0])
                if expires > time.time() + 600:
                    url = locations[-1]
                    break
    assert url, "No unexpired previously verified public CDN redirect"
    for name, item in state["files"].items():
        assert file_hash(dest / name) == item["sha256"]
    state.update(status="downloading", recovery="Reuse existing verified byte ranges and public CDN redirect")
    atomic_json(receipt, state)
    total_parts = (SIZE + CHUNK - 1) // CHUNK
    started = time.time()

    def fetch(i):
        start, end = i * CHUNK, min(SIZE, (i + 1) * CHUNK) - 1
        target = parts / f"{i:05d}.part"
        if target.exists() and target.stat().st_size == end - start + 1:
            return i, True, 0
        tmp, headers = target.with_suffix(".recovery.partial"), target.with_suffix(".recovery.headers")
        for attempt in range(5):
            try:
                result = subprocess.run(["curl", "--noproxy", "*", "--fail", "--location", "--silent", "--show-error",
                    "--range", f"{start}-{end}", "--connect-timeout", "15", "--max-time", "90",
                    "--speed-limit", "1024", "--speed-time", "25", "--dump-header", str(headers),
                    "--output", str(tmp), url], text=True, capture_output=True, timeout=105)
                ranges = re.findall(r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", headers.read_text())
                if result.returncode == 0 and ranges and tuple(map(int, ranges[-1])) == (start, end, SIZE):
                    assert tmp.stat().st_size == end - start + 1
                    tmp.replace(target)
                    return i, True, target.stat().st_size
            except subprocess.TimeoutExpired:
                pass
            time.sleep(2 * (attempt + 1))
        return i, False, 0

    completed, new_bytes, failed = 0, 0, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch, i) for i in range(total_parts)]):
            i, ok, transferred = future.result()
            completed += ok
            new_bytes += transferred
            if not ok:
                failed.append(i)
            atomic_json(ROOT / "receipts/B_prm_download_progress.json", {
                "status": "recovering_missing_ranges", "file": "model.safetensors", "completed_parts": completed,
                "total_parts": total_parts, "new_bytes_this_recovery": new_bytes,
                "new_bytes_per_second": new_bytes / max(time.time() - started, 1),
                "failed_parts": failed, "heartbeat_unix": time.time()})
            if completed % 64 == 0:
                print(f"Available parts: {completed}/{total_parts}; failed={failed}", flush=True)
    assert not failed, f"Missing byte ranges retained for recovery: {failed}"
    target = dest / "model.safetensors"
    staging = target.with_suffix(".recovery.assembled")
    with staging.open("wb") as output:
        for i in range(total_parts):
            with (parts / f"{i:05d}.part").open("rb") as stream:
                for data in iter(lambda: stream.read(CHUNK), b""):
                    output.write(data)
    assert staging.stat().st_size == SIZE and file_hash(staging) == SHA
    staging.replace(target)
    state["files"]["model.safetensors"] = {"sha256": SHA, "bytes": SIZE, "lfs_sha256": SHA,
        "git_blob_sha1": "0bc72b4dcf47871c323237e12b84ffb5839cd5f9"}
    assert len(state["files"]) == 9
    state.update(status="completed", finished_unix=time.time(), recovery_elapsed_seconds=time.time() - started)
    atomic_json(receipt, state)
    atomic_json(ROOT / "receipts/B_prm_download_progress.json", {"status": "completed_and_sha256_verified",
        "file": "model.safetensors", "total_bytes": SIZE, "sha256": SHA, "finished_unix": time.time()})
    print(json.dumps({"status": "completed", "sha256": SHA}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        receipt = ROOT / "receipts/B_prm_assets.json"
        state = json.loads(receipt.read_text())
        state.update(status="failed", error=str(exc), finished_unix=time.time())
        atomic_json(receipt, state)
        raise
