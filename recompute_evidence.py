"""Recompute descriptive counts from the two saved official ProcessBench files.

No network, model inference, training, or dataset modification.
Run from any working directory with Python 3.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent / "evidence"
results = []
for split in ("gsm8k", "math"):
    path = ROOT / f"processbench_{split}.json"
    raw = path.read_bytes()
    rows = json.loads(raw)
    correct = [r for r in rows if r["final_answer_correct"]]
    inconsistent = [r for r in correct if r["label"] != -1]
    result = {
        "split": split,
        "n": len(rows),
        "correct_final_answers": len(correct),
        "correct_answer_with_annotated_process_error": len(inconsistent),
        "rate_given_correct_final_answer": len(inconsistent) / len(correct),
        "source": f"https://huggingface.co/datasets/Qwen/ProcessBench/resolve/3bdcd53/{split}.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "ids": [r["id"] for r in inconsistent],
    }
    results.append(result)
    print(json.dumps({k: v for k, v in result.items() if k != "ids"}))

recorded = json.loads((ROOT / "processbench_audit.json").read_text())
for actual, expected in zip(results, recorded):
    for key, value in actual.items():
        assert value == expected[key], (actual["split"], key, value, expected[key])
print("Saved audit verified; descriptive benchmark statistics only.")
