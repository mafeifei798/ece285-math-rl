"""Verify source pins and replay evidence, never construct a model/trainer."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    release = json.loads((HERE / "release.json").read_text())
    for name, expected in release["source_hashes"].items():
        path = ROOT / name
        if not args.runtime and name == "data/prepared/train.jsonl":
            path = ROOT / "execution/b7b_training_audit_20261021/inputs" / name
        if sha(path) != expected:
            raise AssertionError("Source pin mismatch: " + name)
    result = dict(source_pins_verified=len(release["source_hashes"]), training_started=False)
    if args.runtime:
        for name, expected in release["trusted_runtime_inputs"].items():
            assert sha(ROOT / name) == expected, name
        from trainer import trainer_class
        cls = trainer_class()
        result.update(installed_class_binding_passed=cls.__name__ == "CorrectedBTrainer",
                      runtime_inputs_verified=len(release["trusted_runtime_inputs"]), model_constructed=False)
    else:
        out = ROOT / "execution/b_baseline_corrected_20261021"
        protected = json.loads((out / "protected_before.json").read_text())
        assert all(sha(ROOT / name) == digest for name, digest in protected.items())
        result["protected_original_files_unchanged"] = len(protected)
        trace = list(csv.DictReader((out / "loss_input_trace.csv").open()))
        assert len(trace) == 9056
        assert [int(r["source_row"]) for r in trace] == list(range(9056))
        fixed = [json.loads(s) for s in (ROOT / "execution/reward_repair_v3_20261021/variants/combined.jsonl").open()]
        y = np.array([r["outcome_reward"] for r in fixed]).reshape(-1, 8)
        p = np.array([r["min_score"] or 0 for r in fixed]).reshape(-1, 8)
        rr = np.array([r["reward"] for r in fixed], dtype=np.float32).astype(np.float64).reshape(-1,8)
        expected = (rr-rr.mean(1,keepdims=True))/(rr.std(1,ddof=1,keepdims=True)+1e-4)
        observed = np.array([float(r["advantage_at_loss"]) for r in trace]).reshape(-1,8)
        error = float(np.abs(observed - expected).max())
        assert error < 2e-5
        evidence = json.loads((out / "complete.json").read_text())
        assert evidence["passed"] and evidence["new_training"] is False and evidence["optimizer_steps"] == 0
        assert len(evidence["negative_controls_detected"]) == 4
        result.update(replayed_rows_verified=9056, independent_numpy_max_error=error,
                      negative_controls_verified=4, passed=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
