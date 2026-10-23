"""Experiment-local provenance helpers. No original files are written."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def source_hashes():
    paths = list(HERE.glob("*.py")) + [HERE / "protocol.json"]
    paths += list((ROOT / "training_b").glob("*.py")) + [ROOT / "training_b/config.json"]
    paths += [ROOT / name for name in [
        "training/baseline_config.json", "training/train_baseline.py", "training/reward.py",
        "training/common.py", "training/evaluate_policy.py", "operations/continue_baseline.py",
        "operations/run_continuation.py", "experiments/qwen7b_offline_20261021/core.py",
        "experiments/reward_repair_v2_20261021/v2core.py",
        "experiments/reward_repair_v2_20261021/b_audit_patch.py",
        "experiments/reward_repair_v2_20261021/legacy_snapshot/training_b/train_b.py"]]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def preservation_snapshot():
    """Hash original sources/results; inventory larger optimizer files separately."""
    paths = []
    for folder in ["training", "training_b", "operations"]:
        paths += [p for p in (ROOT / folder).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts]
    paths += list((ROOT / "data/prepared").glob("*.json*"))
    for run in (ROOT / "runs").iterdir():
        if run.is_dir() and run.name.startswith(("A_seed", "B_seed", "pilot_A_seed", "pilot_B_seed")):
            paths += [p for p in run.rglob("*") if p.is_file()]
    hashed, inventory = {}, {}
    for p in sorted(set(paths)):
        key, st = str(p.relative_to(ROOT)), p.stat()
        inventory[key] = [st.st_size, st.st_mtime_ns]
        if p.suffix in {".json", ".jsonl", ".py", ".csv", ".txt", ".log", ".safetensors"}:
            hashed[key] = sha(p)
    return {"sha256": hashed, "inventory": inventory}


def assert_preserved(before):
    after = preservation_snapshot()
    assert before == after, "An original A/B source or result changed; inspect provenance"
    return {"passed": True, "hashed_original_files": len(before["sha256"]),
            "inventoried_original_files": len(before["inventory"])}
