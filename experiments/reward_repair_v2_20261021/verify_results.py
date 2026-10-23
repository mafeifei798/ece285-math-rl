"""Independent Python/float64 QA of saved replay outputs, no model or GPU."""
import json
import itertools
import math
from pathlib import Path
import struct
import sys

from v2core import atomic_json, read_jsonl, sha256


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def main():
    out = Path(sys.argv[1])
    summary = json.loads((out / "replay_summary.json").read_text())
    scoring = json.loads((out / "paired_scoring_receipt.json").read_text())
    maximum, groups = 0., 0
    worst = None
    arrays = {}
    for path in sorted((out / "variants").glob("*.jsonl")):
        rows = read_jsonl(path)
        arrays[path.stem] = rows
        assert len(rows) == summary["rows"]
        for i, r in enumerate(rows):
            assert r["source_row"] == i
            expected_reward = r["outcome_reward"] + .2 * r["min_score"] if r["process_scored"] else 0.
            assert abs(r["reward"] - expected_reward) <= 1e-12
            assert r["reward_fp32"] == f32(r["reward"])
            assert r["process_scored"] or r["min_score"] is None
        for offset in range(0, len(rows), 8):
            group = rows[offset:offset + 8]
            assert len({r["problem_id"] for r in group}) == 1
            x = [f32(r["reward"]) for r in group]
            mean = math.fsum(x) / 8
            sd = math.sqrt(math.fsum((v - mean) ** 2 for v in x) / 7)
            reference = [(v - mean) / (sd + 1e-4) for v in x]
            error = max(abs(a - r["advantage"]) for a, r in zip(reference, group))
            scale = max(map(abs, x), default=0.)
            # Independent real-arithmetic reference, with a conservative
            # variance-conditioned allowance for FP32 reductions. This does not
            # replace the strict same-device installed-TRL checks.
            allowance = max(2e-4, 16 * scale * 2**-23 / (sd + 1e-4) *
                            (1 + max(abs(v - mean) for v in x) / (sd + 1e-4)))
            assert error <= allowance, (path.name, offset, error, allowance)
            if error > maximum:
                maximum = error
                worst = {"variant": path.stem, "group": offset // 8, "error": error, "allowance": allowance}
            for r in group:
                rank = 1 + sum(t["reward_fp32"] > r["reward_fp32"] for t in group) + .5 * (sum(t["reward_fp32"] == r["reward_fp32"] for t in group) - 1)
                assert rank == r["reward_rank"]
            groups += 1
    assert scoring["model_frozen_verified"] and scoring["model_weight_updates"] == 0
    assert scoring["repeat_max_difference"] <= 1e-7 and scoring["unchanged_input_max_difference"] <= 1e-7
    protection = json.loads((out / "originals_after.json").read_text())
    assert protection["passed"]
    def sgn(x):
        return 0 if abs(x) <= 1e-6 else 1 if x > 0 else -1
    retained = set(range(summary["groups"])) - set(summary["quarantined_groups"])
    verified_counts = {}
    for key, variant in [("units_only", "units_only"), ("split_only", "split_only"), ("combined_matched", "combined")]:
        old, new = arrays["legacy_replayed"], arrays[variant]
        ids = [i for i in range(len(old)) if i // 8 in retained]
        counts = {"outcome_label_changed_rows": sum(old[i]["outcome_reward"] != new[i]["outcome_reward"] for i in ids),
                  "reward_changed_rows_gt_1e_minus_6": sum(abs(old[i]["reward"] - new[i]["reward"]) > 1e-6 for i in ids),
                  "advantage_sign_flip_rows": sum(sgn(old[i]["advantage"]) != sgn(new[i]["advantage"]) for i in ids),
                  "advantage_changed_rows_gt_0p01": sum(abs(old[i]["advantage"] - new[i]["advantage"]) > .01 for i in ids)}
        changed_groups = 0
        for group in retained:
            ix = list(range(group * 8, group * 8 + 8))
            changed_groups += any(sgn(old[i]["reward_fp32"] - old[j]["reward_fp32"]) !=
                                  sgn(new[i]["reward_fp32"] - new[j]["reward_fp32"])
                                  for i, j in itertools.combinations(ix, 2))
        counts["reward_order_changed_groups"] = changed_groups
        for field, value in counts.items():
            assert summary["common_retained_groups"][key][field] == value, (key, field)
        verified_counts[key] = counts
    positions_checked = 0
    paired = read_jsonl(out / "paired_scores.jsonl")
    if len(sys.argv) > 2:
        source = Path(sys.argv[2])
        manifest = json.loads((out / "input_manifest.json").read_text())
        assert sha256(source) == manifest["source"]["rollouts"]
        raw = read_jsonl(source)
        for entry in paired:
            text = raw[entry["source_row"]]["completion"]
            for name in ("legacy", "normalized"):
                view = entry[name]
                normalized = name == "normalized"
                assert len(view["steps"]) == view["step_count"] == len(view["positions"]) == len(view["step_scores"])
                assert view["min_score"] == min(view["step_scores"])
                assert all(math.isfinite(p) and 0 <= p <= 1 for p in view["step_scores"])
                assert len(set(view["positions"])) == len(view["positions"])
                assert view["positions"][-1] == view["prm_input_tokens"] - 1
                for step, span in zip(view["steps"], view["spans"]):
                    original_slice = text[span["raw_start"]:span["raw_end"]]
                    assert (original_slice.replace("\r\n", "\n") if normalized else original_slice) == step
                    positions_checked += 1
    qa = {"passed": True, "independent_arithmetic": "Python math.fsum/sample variance on FP32-quantized rewards",
          "checked_variant_groups": groups, "max_FP64_vs_GPU_FP32_advantage_difference": maximum,
          "worst_group": worst, "all_rewards_and_midranks_recomputed": True,
          "original_sources_and_results_protected": True, "new_training": False,
          "new_rollouts": 0, "model_updates": 0,
          "independent_comparison_counts": verified_counts,
          "paired_rows": len(paired), "score_positions_and_raw_spans_checked": positions_checked}
    atomic_json(out / "independent_QA.json", qa)
    print(json.dumps(qa))


if __name__ == "__main__":
    main()
