"""Download public pinned artifacts; verify LFS SHA256 before use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

def get_json(url):
    return json.loads(subprocess.check_output(["curl", "--fail", "--location", "--silent", "--show-error",
                      "--retry", "3", "--max-time", "90", url], text=True))

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, target, expected=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected and sha256(target) != expected:
            raise RuntimeError(f"Existing file fails checksum: {target}")
        return
    partial = target.with_name(target.name + ".partial")
    subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "3",
                    "--connect-timeout", "30", "--max-time", "3600", "--continue-at", "-",
                    "--output", str(partial), url], check=True)
    if expected and sha256(partial) != expected:
        raise RuntimeError(f"Downloaded file fails checksum: {target}")
    partial.replace(target)

def fetch_prm_inventory(commit, target):
    target = Path(target)
    pointer = target.with_suffix(".lfs")
    if target.exists() and target.stat().st_size < 1024 and target.read_text().startswith("version https://git-lfs.github.com/spec/v1"):
        target.replace(pointer)
    fetch(f"https://raw.githubusercontent.com/openai/prm800k/{commit}/prm800k/math_splits/train.jsonl", pointer)
    fields = dict(line.split(" ", 1) for line in pointer.read_text().splitlines())
    assert fields["version"] == "https://git-lfs.github.com/spec/v1"
    expected = fields["oid"].removeprefix("sha256:")
    fetch(f"https://media.githubusercontent.com/media/openai/prm800k/{commit}/prm800k/math_splits/train.jsonl", target, expected)
    assert target.stat().st_size == int(fields["size"])
    return expected

def main():
    (ROOT / "receipts").mkdir(exist_ok=True)
    model = "Qwen/Qwen2.5-Math-1.5B"
    info = get_json(f"{ENDPOINT}/api/models/{model}?blobs=true")
    revision = info["sha"]
    dest = ROOT / "models" / "Qwen2.5-Math-1.5B" / revision
    manifest = {"model_id": model, "model_revision": revision, "model_path": str(dest),
                "download_endpoint": ENDPOINT, "started_unix": time.time(), "files": {}, "datasets": {}}
    meta = ROOT / "receipts/assets.json"
    if meta.exists():
        old = json.loads(meta.read_text())
        if old["model_revision"] != revision:
            raise RuntimeError("Remote model revision changed; inspect existing manifest before updating")
    required = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
    for entry in info["siblings"]:
        name = entry["rfilename"]
        if name in required or name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            print("DOWNLOAD", name, flush=True)
            expected = entry.get("lfs", {}).get("sha256")
            fetch(f"{ENDPOINT}/{model}/resolve/{revision}/{name}", dest / name, expected)
            if not expected and entry.get("blobId"):
                content = (dest / name).read_bytes()
                blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                if blob != entry["blobId"]:
                    raise RuntimeError(f"Git blob checksum mismatch: {name}")
            manifest["files"][name] = {"sha256": sha256(dest / name), "lfs_sha256": expected,
                                       "git_blob_sha1": entry.get("blobId"),
                                       "bytes": (dest / name).stat().st_size}
    for key, repo in {"math": "EleutherAI/hendrycks_math", "math500": "HuggingFaceH4/MATH-500", "gsm8k": "openai/gsm8k"}.items():
        entry = get_json(f"{ENDPOINT}/api/datasets/{repo}")
        manifest["datasets"][key] = {"id": repo, "revision": entry["sha"]}
    # Public PRM training question inventory, used only for exposure auditing.
    commit = get_json("https://api.github.com/repos/openai/prm800k/commits/main")["sha"]
    target = ROOT / "data/raw/prm800k_train_questions.jsonl"
    expected = fetch_prm_inventory(commit, target)
    manifest["prm800k_audit"] = {"commit": commit, "path": str(target), "sha256": sha256(target), "lfs_sha256": expected}
    manifest.update(status="completed", finished_unix=time.time())
    meta.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "completed", "model_revision": revision, "datasets": manifest["datasets"]}), flush=True)

if __name__ == "__main__":
    main()
