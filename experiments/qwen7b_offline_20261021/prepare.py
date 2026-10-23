"""Prespecify all saved responses, fixed reviews and both input views before scoring."""
import json
from pathlib import Path
import sys
from core import atomic_json, read_jsonl, sha256, SOURCE_SHA


def main():
    root, out = map(lambda x: Path(x).resolve(), sys.argv[1:3])
    if (out / "input_manifest.json").exists():
        raise RuntimeError("Prepared audit exists; do not replace it")
    out.mkdir(parents=True, exist_ok=True)
    inputs = ["runs/B_seed17_6M/rollouts.jsonl", "data/prepared/train.jsonl",
              "audits/reward_repair_v2_20261021/outcomes.jsonl",
              "audits/reward_repair_v2_20261021/quarantine.json",
              "audits/qwen7b_offline_20261021/reviewed_records.json"]
    checksums = {rel: sha256(root / rel) for rel in inputs}
    assert checksums[inputs[0]] == SOURCE_SHA
    rows = read_jsonl(root / inputs[0])
    assert len(rows) == 9120 and len(rows) % 8 == 0
    questions = {r["problem_id"]: r for r in read_jsonl(root / inputs[1])}
    for offset in range(0, len(rows), 8):
        group = rows[offset:offset + 8]
        assert len({r["problem_id"] for r in group}) == 1
        assert len({r["optimizer_step_before_update"] for r in group}) == 1
    reviews = json.loads((root / inputs[-1]).read_text())
    for review in reviews:
        source = rows[review["source_row"]]
        assert source["completion"] == review["saved"]["completion"]
        assert questions[source["problem_id"]]["problem"] == review["question"]
        assert len(review["steps"]) == len(review["review"]["step_labels"]) == source["step_count"]
    valid = [i for i, r in enumerate(rows) if r["valid_for_process_reward"]]
    assert len(valid) == 7472
    manifest = {"inputs": checksums, "rows": len(rows), "groups": len(rows) // 8,
                "eligible_rows": valid, "review_rows": [r["source_row"] for r in reviews],
                "extra_known_cases": [78, 2095, 4975], "group_size": 8, "epsilon": 1e-4,
                "reward": "same original outcome + 0.2 * min(native step scores); invalid total zero",
                "primary": "raw double-LF steps; both models freshly scored; original outcomes held fixed",
                "secondary": "CRLF-normalized PRM-only views; shared previously fixed typed-unit outcomes",
                "quarantine": "exclude entire previously identified contradictory-question group in primary summaries; keep all raw records",
                "microbatch": 8, "length_sort": True, "thresholds": [0.5, 0.05, 0.8],
                "quantization": "BF16 model; FP32 softmax; FP32 actual installed TRL reward kernel",
                "model_selection": "no fitting, no checkpoint or threshold tuning",
                "review_labels": "reuse prior Codex single-reviewer labels, not an independent human gold set",
                "new_training": False, "new_rollouts": 0, "model_weight_updates": 0}
    atomic_json(out / "input_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"eligible_rows", "review_rows", "inputs"}}), flush=True)


if __name__ == "__main__":
    main()
