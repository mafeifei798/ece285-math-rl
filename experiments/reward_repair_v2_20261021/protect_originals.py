"""Read-only hashes/inventory of original A/B code, inputs and result files."""
import json
from pathlib import Path
import sys
import time

from v2core import atomic_json, sha256


def snapshot(root):
    files = set()
    for directory in ("training", "training_b", "operations"):
        files.update(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.update(root.glob("protocol*.json"))
    for rel in ("data/prepared/train.jsonl", "data/prepared/manifest.json", "receipts/B_prm_assets.json"):
        p = root / rel
        if p.is_file():
            files.add(p)
    inventory, hashed = {}, {}
    for run in (root / "runs").iterdir():
        if not run.name.startswith(("A_", "B_")):
            continue
        for p in run.rglob("*"):
            if not p.is_file():
                continue
            s = p.stat()
            inventory[str(p.relative_to(root))] = {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns}
            if p.suffix in {".json", ".jsonl", ".txt", ".log", ".csv"} or p.name == "adapter_model.safetensors":
                files.add(p)
    for p in sorted(files):
        hashed[str(p.relative_to(root))] = sha256(p)
    return {"root": str(root), "hashed": hashed, "result_file_inventory": inventory,
            "scope": "all original source/config files; A/B JSON/JSONL/text/log/CSV and adapters hashed; all A/B result-file sizes/mtimes inventoried"}


def main():
    root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    current = snapshot(root)
    if len(sys.argv) > 3:
        before = json.loads(Path(sys.argv[3]).read_text())
        current["hashed_files_unchanged"] = current["hashed"] == before["hashed"]
        current["result_inventory_unchanged"] = current["result_file_inventory"] == before["result_file_inventory"]
        current["passed"] = current["hashed_files_unchanged"] and current["result_inventory_unchanged"]
        atomic_json(output, current)
        assert current["passed"], "Original sources/results changed during offline replay"
    else:
        if output.exists():
            raise RuntimeError("Keep existing before receipt")
        atomic_json(output, current)
    print(json.dumps({k: v for k, v in current.items() if k not in {"hashed", "result_file_inventory"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
