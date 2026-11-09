"""CPU-only checks for report numbers and algebra; never loads a model.

Run inside the original project checkout. Historical loss-input / tensor audits
are cited separately; this script does not claim to repeat those model checks.
"""
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import random
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVIDENCE = ROOT / "execution/ab_corrected_6m_seed17_20261022/final_evidence"
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def note(path):
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return path


def read(path):
    return json.loads(note(path).read_text())


def rows(path):
    with note(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalized(values, eps=1e-4):
    mu = statistics.mean(values)
    sd = statistics.stdev(values)
    return [(v - mu) / (sd + eps) for v in values]


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b, tol)


def main():
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "GPU_used": False, "model_calls": 0, "optimizer_updates": 0,
        "submitted_anything": False,
        "scope": "Recount frozen saved labels and replay scalar algebra only.",
    }
    recorded = read(EVIDENCE / "results_summary.json")
    signal = read(EVIDENCE / "signal_audit/analysis.json")
    training = {}
    orders = {}
    for arm, n, tokens, steps in [("A", 9088, 6022435, 284), ("B", 8960, 6008457, 280)]:
        base = EVIDENCE / f"raw/runs/{arm}_corrected_seed17_6M_20261022"
        data = rows(base / "rollouts.jsonl")
        assert len(data) == n
        assert sum(r["completion_tokens"] for r in data) == tokens
        assert max(r["optimizer_step_before_update"] for r in data) + 1 == steps
        conf = read(base / "run_config.json")
        args = conf["shared_training_arguments"]
        for key, value in {"learning_rate": 5e-6, "num_generations": 8,
                           "num_iterations": 1, "loss_type": "grpo", "scale_rewards": "group",
                           "beta": .04, "epsilon": .2, "max_completion_length": 2048,
                           "gradient_accumulation_steps": 32,
                           "vllm_importance_sampling_correction": False}.items():
            assert args[key] == value, (arm, key)
        groups = Counter()
        wp = Counter()
        error = 0.
        cancellation_error = 0.
        total_abs = homogeneous_abs = 0.
        orders[arm] = []
        for start in range(0, n, 8):
            g = data[start:start + 8]
            assert len({r["problem_id"] for r in g}) == 1
            orders[arm].append(g[0]["problem_id"])
            y = [r["outcome_reward"] for r in g]
            m = [r["min_score"] for r in g]
            gt = "all_wrong" if sum(y) == 0 else "all_correct" if sum(y) == 8 else "mixed"
            groups[gt] += 1
            rewards = [yi + (.2 * mi if arm == "B" else 0) for yi, mi in zip(y, m)]
            for r, reward in zip(g, rewards):
                close(r["reward"], reward, 2e-7)
            expected = normalized(rewards)
            for r, ai in zip(g, expected):
                error = max(error, abs(r["advantage"] - ai))
                amount = abs(r["advantage"])
                total_abs += amount
                if gt != "mixed":
                    homogeneous_abs += amount
                if r["outcome_reward"] == 0 and r["advantage"] > 0:
                    wp[gt] += 1
            if arm == "B" and gt != "mixed":
                alternate = normalized(m, 1e-4 / .2)
                cancellation_error = max(cancellation_error, max(abs(a-b) for a,b in zip(expected, alternate)))
        assert error < 2e-5
        close(homogeneous_abs / total_abs, signal[arm]["homogeneous_group_share_of_total_absolute_actual_advantage"], 1e-8)
        audit = read(EVIDENCE / f"training_audit/{arm}.json")
        assert audit["passed"]
        training[arm] = {
            "responses": n, "completion_tokens": tokens, "updates": steps,
            "groups": dict(groups), "wrong_positive_advantages": dict(wp),
            "max_FP64_advantage_difference": error,
            "homogeneous_absolute_advantage_share": homogeneous_abs / total_abs,
            "normalization_cancellation_max_difference": cancellation_error,
            "historical_native_tensor_audit_passed": audit["passed"],
        }
    assert orders["A"][:1120] == orders["B"]
    assert training["B"]["wrong_positive_advantages"] == {"all_wrong": 571, "mixed": 6}
    assert training["B"]["groups"] == {"mixed": 878, "all_wrong": 212, "all_correct": 30}
    assert not training["A"]["wrong_positive_advantages"]
    result["main_training"] = training

    # Recount actual saved predictions. This does not change or replace the
    # historical verifier labels; independent regrading has a separate receipt.
    evaluation = {}
    for ds, n in [("math500", 500), ("gsm8k", 1319)]:
        arms = {}
        for arm in ["A", "B"]:
            p = EVIDENCE / f"raw/runs/{arm}_corrected_seed17_6M_20261022/evaluation/{ds}.jsonl"
            data = rows(p)
            assert len(data) == n
            arms[arm] = data
            assert sum(r["reward"] for r in data) == recorded["datasets"][ds][f"{arm}_correct"]
        assert [r["problem_id"] for r in arms["A"]] == [r["problem_id"] for r in arms["B"]]
        wins_a = sum(a["reward"] > b["reward"] for a,b in zip(arms["A"], arms["B"]))
        wins_b = sum(b["reward"] > a["reward"] for a,b in zip(arms["A"], arms["B"]))
        assert wins_a == recorded["datasets"][ds]["A_only_correct"]
        assert wins_b == recorded["datasets"][ds]["B_only_correct"]
        close(100 * (wins_b - wins_a) / n, recorded["datasets"][ds]["B_minus_A_pp"])
        evaluation[ds] = recorded["datasets"][ds]
    result["main_evaluation_recount"] = evaluation
    result["historical_verifier_regrade"] = {"passed": read(EVIDENCE / "eval_audit/verified.json")["passed"]}
    assert result["historical_verifier_regrade"]["passed"]

    # Independent finite enumeration of the exact graph in the saved case.
    graph = read(EVIDENCE / "case_review/graph_group_independent_check.json")
    count = sum(all(colors[i] != colors[j] for i,j in graph["edges"])
                for colors in product(range(3), repeat=len(graph["vertices"])))
    assert count == graph["colorings_3_colors"] == 54
    assert 4*6 + 8*3 - 28/2 == 34
    case = read(EVIDENCE / "case_review/prm_high_wrong_review.json")["summary"]
    assert len(case["mathematical_error_high_PRM"]) == 4
    assert len(case["correct_math_but_boxing_or_unit_protocol"]) == 4
    assert len(case["actual_positive_advantage_among_selected"]) == 1
    result["targeted_case_checks"] = {"graph_colorings": count, "arithmetic_value": 34,
        "high_PRM_top10_counts_verified": True, "representative_population_labels": False}

    # Verify the seven algebraic constraints used in the brief bounded section.
    rng = random.Random(285)
    worst_sum = worst_bound = 0.
    for bits in product([0., 1.], repeat=8):
        m = [rng.random() for _ in bits]
        nc = sum(bits)
        mu = sum(mi*yi for mi,yi in zip(m,bits))/nc if nc else 0.
        c = [.2*yi*(mi-mu) for mi,yi in zip(m,bits)]
        out = normalized(bits)
        combined = [a+b for a,b in zip(out,c)]
        assert max(abs(x) for x in c) <= .2 + 1e-12
        assert abs(sum(c)) < 1e-12
        assert all(a <= 0 for yi,a in zip(bits,combined) if yi == 0)
        if nc == 0:
            assert combined == [0.]*8
        worst_sum = max(worst_sum, abs(sum(c)))
        worst_bound = max(worst_bound, max(abs(x) for x in c))
    for u in [.001, .01, .1, .5, 1., 2., 10., 1000.]:
        assert u - math.log(u) - 1 >= -1e-12
    # A ratio evaluated at old==current is 1, but its derivative is 1 when the
    # denominator is held fixed. This is a scalar check, not an autograd run.
    h = 1e-6
    derivative = (math.exp(h) - math.exp(-h))/(2*h)
    close(derivative, 1., 1e-8)
    result["formula_checks"] = {"binary_group_patterns_tested": 256,
        "bounded_process_max_abs": worst_bound, "max_group_process_sum_abs": worst_sum,
        "wrong_advantage_never_positive": True, "all_wrong_reward_advantage_zero": True,
        "sampled_KL_estimator_nonnegative": True, "detached_ratio_finite_difference": derivative}

    preliminary = {}
    for seed, directory in [(17, "abc_seed17_resumed_v1_20261104"), (42, "ac_seed42_v1_20261106")]:
        d = read(ROOT / f"execution/{directory}/completed_evidence/comparison.json")
        assert d["passed"] and d["global_step"] == 288
        preliminary[str(seed)] = {a: {ds: d["summary"][a][ds]["accuracy"] for ds in ["math500", "gsm8k"]}
                                 for a in ["A", "C_bounded"]}
        for a in ["A", "C_bounded"]:
            assert d["accounting"][a]["loss_wrong_positive"] == 0
        preliminary[str(seed)]["paired"] = {ds: d["paired"][ds]["C_bounded minus A"]
                                             for ds in ["math500", "gsm8k"]}
    expected = {"17": {"A": [.53, 882/1319], "C_bounded": [.578, 915/1319]},
                "42": {"A": [.528, 890/1319], "C_bounded": [.548, 896/1319]}}
    for seed in expected:
        for arm in expected[seed]:
            for ds, value in zip(["math500", "gsm8k"], expected[seed][arm]):
                close(preliminary[seed][arm][ds], value)
    result["separate_preliminary_experiment"] = preliminary
    result["requirements"] = read(ROOT / "execution/midterm_requirements_20261109/requirements.json")
    result["style_sha256"] = sha(HERE / "neurips_2025.sty")
    assert result["style_sha256"] == sha(ROOT / "proposal_neurips/neurips_2025.sty")
    result["source_sha256"] = sha(HERE / "main.tex")
    result["inputs_sha256"] = INPUTS
    result["passed"] = True
    (HERE / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": True, "input_files": len(INPUTS), "training_rows_replayed": 18048,
                      "prediction_rows_recounted": 3638, "GPU_used": False}))


if __name__ == "__main__":
    main()
