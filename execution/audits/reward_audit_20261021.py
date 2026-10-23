"""Read saved A/B records only. Run CPU-only; never import a training entrypoint."""
import collections
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": len(a), "mean": float(a.mean()),
            "quantiles_0_10_25_50_75_90_100": np.quantile(a, [0, .1, .25, .5, .75, .9, 1]).tolist()}


def corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def normalized(values):
    r = np.asarray(values, dtype=np.float32)
    return (r - r.mean()) / (r.std(ddof=1) + 1e-4)


def profile(rows, arm):
    valid = [r for r in rows if r["status"] in {"exact", "symbolic"}]
    outcome_key = "outcome_reward" if arm == "B" else "reward"
    ans = {"n": len(rows), "tokens": sum(r["completion_tokens"] for r in rows),
           "accuracy": float(np.mean([r[outcome_key] for r in rows])),
           "status": dict(collections.Counter(r["status"] for r in rows)),
           "length": stats([r["completion_tokens"] for r in rows]),
           "no_double_newline_fraction": float(np.mean(["\n\n" not in r["completion"].strip() for r in rows])),
           "raw_step_count": stats([len([s for s in r["completion"].split("\n\n") if s.strip()]) for r in rows])}
    if arm == "B":
        ans["valid_min_score"] = stats([r["min_score"] for r in valid])
        for y in [0, 1]:
            sub = [r for r in valid if r["outcome_reward"] == y]
            ans[f"valid_outcome_{y}"] = {"min_score": stats([r["min_score"] for r in sub]),
                 "steps": stats([r["step_count"] for r in sub]),
                 "corr_score_steps": corr([r["min_score"] for r in sub], [r["step_count"] for r in sub]),
                 "corr_score_length": corr([r["min_score"] for r in sub], [r["completion_tokens"] for r in sub]),
                 "min_below_001": sum(r["min_score"] < .01 for r in sub),
                 "min_below_01": sum(r["min_score"] < .1 for r in sub),
                 "min_above_08": sum(r["min_score"] > .8 for r in sub)}
    return ans


A = read_jsonl(ROOT / "runs/A_seed17/rollouts.jsonl")
B = read_jsonl(ROOT / "runs/B_seed17_6M/rollouts.jsonl")
GD = read_jsonl(ROOT / "runs/B_seed17_6M/group_diagnostics.jsonl")
TRAIN = {r["problem_id"]: r for r in read_jsonl(ROOT / "data/prepared/train.jsonl")}
assert len(B) % 32 == 0 and len(GD) * 8 == len(B)
result = {"scope": "Immutable initial 6M runs; descriptive replay only; no actor or PRM parameters changed",
          "rows_A": len(A), "rows_B": len(B), "groups_B": len(GD),
          "first_32_completions_identical": all(a["completion"] == b["completion"] for a, b in zip(A[:32], B[:32])),
          "common_prefix_prompt_order_identical": all(a["problem_id"] == b["problem_id"] for a, b in zip(A, B)),
          "first32_outcome_checks_identical": all(a["reward"] == b["outcome_reward"] and a["status"] == b["status"] and a["answer"] == b["answer"] for a, b in zip(A[:32], B[:32])),
          "profiles": {"A_6M": profile(A, "A"), "B_6M": profile(B, "B")}}
errors = []
for i, r in enumerate(B):
    valid = r["status"] in {"exact", "symbolic"}
    steps = [s.strip() for s in r["completion"].split("\n\n") if s.strip()]
    expected_min = min(r["step_scores"]) if valid else 0
    expected_reward = r["outcome_reward"] + .2 * expected_min if valid else 0
    checks = [r["valid_for_process_reward"] == valid, math.isclose(expected_reward, r["reward"], abs_tol=1e-12),
              expected_min == r["min_score"], r["step_count"] == len(r["step_scores"]),
              all(math.isfinite(x) and 0 <= x <= 1 for x in r["step_scores"]),
              not valid or len(steps) == r["step_count"], valid or not r["step_scores"],
              not r["outcome_reward"] or (valid and r["terminated"])]
    if not all(checks): errors.append({"row": i, "checks": checks})
result["reward_replay_errors"] = errors

strata = collections.defaultdict(list)
group_records = []
pairs = []
discordant_examples = []
for gi, g in enumerate(GD):
    rows = B[8 * gi:8 * (gi + 1)]
    y = np.array([r["outcome_reward"] for r in rows])
    r = np.array([r["reward"] for r in rows])
    b = normalized(r)
    a = normalized(y)
    m = np.array([r["min_score"] for r in rows])
    valid = np.array([r["valid_for_process_reward"] for r in rows])
    assert len({row["problem_id"] for row in rows}) == 1
    assert rows[0]["problem_id"] == g["problem_id"]
    assert abs(float(np.abs(b).mean()) - g["mean_abs_advantage"]) < 5e-5
    assert abs(float(np.std(b, ddof=1)) - g["advantage_std"]) < 5e-5
    entry = {"gi": gi, "type": g["outcome_type"], "all_valid": bool(valid.all()),
             "n_correct": int(y.sum()), "n_invalid": int((~valid).sum()),
             "mean_abs_A": float(np.abs(a).mean()), "mean_abs_B": float(np.abs(b).mean()),
             "energy_A": float(np.sum(a*a)), "energy_B": float(np.sum(b*b)),
             "energy_delta": float(np.sum((b-a)**2)),
             "correct_negative": int(np.sum((y == 1) & (b < -1e-6))),
             "wrong_positive": int(np.sum((y == 0) & (b > 1e-6))),
             "all_wrong_positive_mean_process": float(np.mean(m[b > 1e-6])) if y.sum() == 0 and (b > 1e-6).any() else None,
             "mixed_sign_flips": int(np.sum(a*b < -1e-6)) if g["outcome_type"] == "mixed" else 0,
             "mean_process": float(m.mean()), "reward_std": float(r.std(ddof=1))}
    group_records.append(entry)
    strata[g["outcome_type"]].append(entry)
    strata[g["outcome_type"] + ("_all_valid" if valid.all() else "_has_invalid")].append(entry)
    yes = [i for i, row in enumerate(rows) if row["outcome_reward"] and row["valid_for_process_reward"]]
    no = [i for i, row in enumerate(rows) if not row["outcome_reward"] and row["valid_for_process_reward"]]
    comparisons = [(float(m[i] > m[j]) + .5 * float(m[i] == m[j])) for i in yes for j in no]
    if comparisons:
        pairs.append({"gi": gi, "auc_within_prompt": float(np.mean(comparisons)), "pair_count": len(comparisons),
                      "same_prompt_min_gap": float(m[yes].mean() - m[no].mean()),
                      "answer_correct_worse_than_wrong_pairs": int(np.sum(np.asarray(comparisons) == 0))})
        for i in yes:
            for j in no:
                if m[i] < .1 and m[j] > .8:
                    discordant_examples.append({"gi": gi, "correct_row": 8*gi+i, "wrong_row": 8*gi+j, "gap": float(m[j]-m[i])})

total_energy = sum(g["energy_B"] for g in group_records)
result["strata"] = {}
for key, entries in strata.items():
    result["strata"][key] = {"groups": len(entries), "fraction_of_all_groups": len(entries)/len(GD),
        "advantage_energy_share": sum(x["energy_B"] for x in entries)/total_energy,
        "mean_abs_A": float(np.mean([x["mean_abs_A"] for x in entries])),
        "mean_abs_B": float(np.mean([x["mean_abs_B"] for x in entries])),
        "mean_std_reward": float(np.mean([x["reward_std"] for x in entries])),
        "correct_negative_count": sum(x["correct_negative"] for x in entries),
        "wrong_positive_count": sum(x["wrong_positive"] for x in entries),
        "mixed_sign_flip_count": sum(x["mixed_sign_flips"] for x in entries)}
result["energy_note"] = "Squared scalar advantage only; this is not gradient energy, causal contribution, or policy improvement."
result["same_prompt_score_vs_outcome"] = {
    "note": "Final answer correctness is a proxy, not process labels. Scores can reasonably reward useful steps in wrong answers.",
    "mixed_groups_with_both_valid_outcomes": len(pairs),
    "mean_group_AUC": float(np.mean([p["auc_within_prompt"] for p in pairs])),
    "pair_weighted_AUC": sum(p["auc_within_prompt"]*p["pair_count"] for p in pairs)/sum(p["pair_count"] for p in pairs),
    "pairs": sum(p["pair_count"] for p in pairs),
    "correct_ranked_below_wrong_pairs": sum(p["answer_correct_worse_than_wrong_pairs"] for p in pairs),
    "mean_correct_minus_wrong_score": float(np.mean([p["same_prompt_min_gap"] for p in pairs]))}
rng = np.random.default_rng(17)
scores = np.array([p["auc_within_prompt"] for p in pairs])
boot = [float(np.mean(rng.choice(scores, len(scores), replace=True))) for _ in range(2000)]
result["same_prompt_score_vs_outcome"]["group_bootstrap_AUC_95CI"] = np.quantile(boot, [.025,.975]).tolist()
result["profiles_by_window"] = {}
for arm, rows in [("A", A), ("B", B)]:
    for key, selected in [("first1600", rows[:1600]), ("last1600", rows[-1600:])]:
        result["profiles_by_window"][arm+"_"+key] = profile(selected, arm)

# Representative audit records: this is selected illustration, never an error-rate estimate.
examples = []
seen = set()
for p in sorted(discordant_examples, key=lambda x: -x["gap"]):
    if p["gi"] in seen: continue
    seen.add(p["gi"])
    c, w = B[p["correct_row"]], B[p["wrong_row"]]
    if max(c["completion_tokens"], w["completion_tokens"]) > 1600: continue
    examples.append({**p, "question": TRAIN[c["problem_id"]]["problem"], "correct": c, "wrong": w})
    if len(examples) >= 6: break
(OUT / "selected_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False))
(OUT / "groups_replayed.json").write_text(json.dumps(group_records))
result["discordant_example_pair_count"] = len(discordant_examples)

# Bounded independent verifier replay, stratified over correctness/status.
sys.path.insert(0, str(ROOT / "training"))
from reward import outcome
selected_indices = set(range(32))
rr = random.Random(17)
for status in sorted({r["status"] for r in B}):
    ix = [i for i, row in enumerate(B) if row["status"] == status]
    selected_indices.update(rr.sample(ix, min(16, len(ix))))
verifier_errors = []
for i in sorted(selected_indices):
    row = B[i]
    checked = outcome(row["completion"], row["gold"], row["terminated"])
    if checked != {"reward": row["outcome_reward"], "status": row["status"], "answer": row["answer"]}:
        verifier_errors.append({"i": i, "checked": checked, "saved": {k: row[k] for k in ["outcome_reward", "status", "answer"]}})
result["independent_verifier_replay"] = {"n": len(selected_indices), "errors": verifier_errors}

(OUT / "reward_replay.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
print(json.dumps(result, ensure_ascii=False, allow_nan=False))
