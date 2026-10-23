"""Helpers for an isolated pilot. No live training module is changed."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "artifacts"
CONFIG = json.loads((HERE / "config.json").read_text())
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "training_b"))
from common import atomic_json, file_hash, question_key, read_jsonl, write_jsonl


def rank(namespace, identifier):
    return hashlib.sha256(f"285:{namespace}:{identifier}".encode()).hexdigest()


def emit(stage, **values):
    row = {"stage": stage, "pid": os.getpid(), "time_unix": time.time(), **values}
    atomic_json(OUT / "progress.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def guard():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Pilot must be CPU-only"
    p = ROOT / "runs/B_seed17_24M/execution_source_hashes.json"
    expected = json.loads(p.read_text())
    actual = {name: file_hash(ROOT / name) for name in expected}
    assert actual == expected, "Live B source hash changed; investigate without editing B"
    assert set(str(p.relative_to(ROOT)) for p in (ROOT / "training_b").glob("*.py")) == {
        name for name in expected if name.startswith("training_b/") and name.endswith(".py")}
    return {"B_source_hashes_unchanged": True, "source_manifest_sha256": file_hash(p),
            "B_progress": json.loads((ROOT / "runs/B_seed17_24M/progress.json").read_text())}


def cpu_setup():
    import torch
    torch.set_num_threads(CONFIG["threads"])
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()


def json_read(path):
    return json.loads(Path(path).read_text())
