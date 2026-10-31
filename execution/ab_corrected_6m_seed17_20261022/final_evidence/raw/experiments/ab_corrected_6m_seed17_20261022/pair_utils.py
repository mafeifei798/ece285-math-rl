"""Pinned paired-run bookkeeping; no model imports or automatic execution."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "training"))
from common import atomic_json as atomic, file_hash as sha, read_jsonl


def read(path):
    return json.loads(Path(path).read_text())


def verify_sources():
    release = read(HERE / "release.json")
    for name, digest in release["source_hashes"].items():
        assert sha(ROOT / name) == digest, "Changed pinned source: " + name
    for name, digest in release["runtime_inputs"].items():
        assert sha(ROOT / name) == digest, "Changed runtime input: " + name
    return release


def preservation_snapshot():
    paths = []
    for folder in ["training", "training_b", "operations", "data/prepared"]:
        paths += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for name in ["A_seed17", "A_seed17_24M", "B_seed17_6M", "B_seed17_24M", "B_Qwen7B_seed17_6M"]:
        paths += [p for p in (ROOT / "runs" / name).rglob("*") if p.is_file()]
    inventory, hashes = {}, {}
    for p in sorted(set(paths)):
        if p.suffix == ".lock":
            continue
        st = p.stat()
        key = str(p.relative_to(ROOT))
        inventory[key] = [st.st_size, st.st_mtime_ns]
        if p.suffix in {".json", ".jsonl", ".py", ".csv", ".txt", ".log", ".safetensors"}:
            hashes[key] = sha(p)
    return dict(inventory=inventory, sha256=hashes)


def assert_preserved(before):
    assert preservation_snapshot() == before, "Historical source/result changed"
    return dict(passed=True, files=len(before["inventory"]), hashed=len(before["sha256"]))
