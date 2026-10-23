#!/usr/bin/env python3
"""Independent CPU-only description of the completed corrected A/B pair.

Imports only Python's standard library. No project/training modules are imported.
This is an audit of saved training signals, not a new reward or training objective.
The default CLI requires both final completed runs and creates --output exclusively.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct

GROUP = 8
COEFFICIENT = 0.2
EPS = 1e-4
VALID = {"exact", "symbolic", "unit_exact", "unit_symbolic"}
KINDS = ("mixed", "all_correct", "all_wrong")


class SignalError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SignalError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    def reject(value):
        raise SignalError(f"Nonfinite JSON value {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def read_lines(path):
    with Path(path).open() as stream:
        return [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
            SignalError(f"Nonfinite JSON value {value}"))) for line in stream if line.strip()]


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"Invalid numeric {label}: {value!r}")
    return float(value)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def describe(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "min": None, "median": None, "p95": None, "max": None}
    def quantile(q):
        position = q * (len(values) - 1)
        left = math.floor(position)
        right = math.ceil(position)
        return values[left] + (values[right] - values[left]) * (position - left)
    return {"n": len(values), "mean": statistics.fmean(values), "min": values[0],
            "median": quantile(.5), "p95": quantile(.95), "max": values[-1]}


def canonical(raw):
    """Independent CRLF and double-LF reconstruction with raw character offsets."""
    chars, offsets = [], []
    cursor = 0
    while cursor < len(raw):
        offsets.append(cursor)
        if raw.startswith("\r\n", cursor):
            chars.append("\n")
            cursor += 2
        else:
            chars.append(raw[cursor])
            cursor += 1
    offsets.append(len(raw))
    text = "".join(chars)
    steps, spans = [], []
    start = 0
    for end in [m.start() for m in re.finditer("\n\n", text)] + [len(text)]:
        chunk = text[start:end]
        if chunk.strip():
            left = start + len(chunk) - len(chunk.lstrip())
            right = end - len(chunk) + len(chunk.rstrip())
            steps.append(chunk.strip())
            spans.append({"canonical_start": left, "canonical_end": right,
                          "raw_start": offsets[left], "raw_end": offsets[right]})
        start = end + 2
    return {"text": text, "steps": steps, "spans": spans,
            "crlf_replacements": raw.count("\r\n"), "raw_sha256": text_sha(raw)}


def group_decomposition(rows):
    """FP64 arithmetic over the actual FP32-quantized reward inputs.

    A = centered(y) / (sd(R)+eps) + centered(R-y) / (sd(R)+eps).
    R-y is the applied process component including float32 rounding. This is an
    algebraic decomposition of existing advantages, never a candidate objective.
    """
    require(len(rows) == GROUP, "Incomplete group")
    y = [finite(r["outcome_reward"], "outcome_reward") for r in rows]
    require(all(v in (0., 1.) for v in y), "Nonbinary outcome")
    reward = [f32(finite(r["reward"], "reward")) for r in rows]
    actual = [finite(r["advantage"], "advantage") for r in rows]
    sd = statistics.stdev(reward)  # Bessel/sample denominator (n-1), as pinned TRL.
    denom = sd + EPS
    y_mean = statistics.fmean(y)
    z = [r - v for r, v in zip(reward, y)]
    z_mean = statistics.fmean(z)
    y_term = [(v - y_mean) / denom for v in y]
    p_term = [(v - z_mean) / denom for v in z]
    expected = [a + b for a, b in zip(y_term, p_term)]
    outcome_denom = statistics.stdev(y) + EPS
    outcome_only = [(v - y_mean) / outcome_denom for v in y]
    return {"kind": "all_correct" if sum(y) == GROUP else "all_wrong" if sum(y) == 0 else "mixed",
            "reward_sample_std": sd, "normalization_gain": 1 / denom,
            "effective_process_coefficient": COEFFICIENT / denom,
            "outcome_term": y_term, "process_term": p_term,
            "expected": expected, "actual": actual, "outcome_only": outcome_only,
            "residual_vs_outcome_only": [a - b for a, b in zip(actual, outcome_only)],
            "reference_max_absolute_difference": max(abs(a-b) for a,b in zip(actual, expected))}


def prefix_comparison(a, b):
    """Different group counts are expected at equal generated-token budgets."""
    common = min(len(a), len(b))
    mismatches = [i for i in range(common) if a[i] != b[i]]
    prefix = mismatches[0] if mismatches else common
    return {"a_groups": len(a), "b_groups": len(b), "overlap_groups": common,
            "identical_initial_prefix_groups": prefix, "mismatched_overlap_groups": len(mismatches),
            "first_mismatch_group": mismatches[0] if mismatches else None,
            "overlap_prompt_order_equal": not mismatches,
            "unpaired_a_tail_groups": len(a) - common, "unpaired_b_tail_groups": len(b) - common,
            "a_order_sha256": text_sha(json.dumps(a, ensure_ascii=False, separators=(",", ":"))),
            "b_order_sha256": text_sha(json.dumps(b, ensure_ascii=False, separators=(",", ":")))}


def row_checks(row, arm, index):
    label = f"{arm} row {index}"
    raw = row["completion"]
    require(isinstance(raw, str), label + " missing text")
    require(isinstance(row["terminated"], bool), label + " termination not boolean")
    require(isinstance(row["completion_tokens"], int) and 0 < row["completion_tokens"] <= 2048,
            label + " invalid completion token count")
    require(isinstance(row["optimizer_step_before_update"], int)
            and row["optimizer_step_before_update"] >= 0, label + " invalid update index")
    require(row["advantage_rule"] == "mixed_reward_original_group_std", label + " wrong rule")
    require(not row["status"].startswith("verifier_error"), label + " verifier exception")
    y = finite(row["outcome_reward"], label + " outcome")
    require(y in (0., 1.), label + " nonbinary outcome")
    valid = row["status"] in VALID
    require(not y or valid, label + " correct invalid output")
    require(row["terminated"] == (row["status"] != "truncated"), label + " truncation mismatch")
    view = canonical(raw)
    require(row["raw_no_double_newline"] == ("\n\n" not in raw.strip()), label + " raw split flag")
    scored = arm == "B" and valid
    require(row["valid_for_process_reward"] == scored, label + " PRM validity mismatch")
    scores = row["step_scores"]
    require(isinstance(scores, list), label + " scores not list")
    for score in scores:
        require(0 <= finite(score, label + " step score") <= 1, label + " probability outside [0,1]")
    if scored:
        require(scores and len(scores) == row["step_count"] == len(view["steps"]), label + " step alignment")
        require(row["step_raw_spans"] == view["spans"], label + " raw/canonical spans")
        require(row["raw_sha256"] == view["raw_sha256"], label + " raw text hash")
        require(row["crlf_replacements"] == view["crlf_replacements"], label + " CRLF count")
        require(row["canonical_no_double_newline"] == ("\n\n" not in view["text"].strip()), label + " canonical flag")
        require(row["no_internal_separator"] == (len(scores) == 1), label + " native step flag")
        require(row["repair_version"] == "reward-repair-v3-20261021", label + " repair version")
        require(0 < row["prm_input_tokens"] <= 4096, label + " native PRM context")
        require("<extra_0>" not in raw, label + " literal native marker")
        require(abs(row["min_score"] - min(scores)) <= 1e-12, label + " minimum mismatch")
        expected = y + COEFFICIENT * min(scores)
    else:
        require(not scores and row["step_count"] == 0 and row["min_score"] == 0
                and row["prm_input_tokens"] == 0, label + " unscored placeholder not zero/empty")
        expected = y if arm == "A" else 0.
    require(abs(finite(row["reward"], label + " reward") - expected) <= 1e-12,
            label + " saved reward differs from original formula")
    return view, scored


def analyze_rows(rows, arm):
    require(rows and len(rows) % GROUP == 0, arm + " not complete groups")
    require(len(rows) % 32 == 0, arm + " not complete optimizer updates")
    data = defaultdict(list)
    counts = Counter()
    status = Counter()
    group_data = {kind: defaultdict(list) for kind in KINDS}
    group_counts = {kind: Counter() for kind in KINDS}
    curve = defaultdict(Counter)
    order = []
    tokens = 0
    maximum_error = 0.
    mismatching_reference = 0
    for i, row in enumerate(rows):
        view, scored = row_checks(row, arm, i)
        counts["rows"] += 1
        counts["valid_outputs"] += row["status"] in VALID
        counts["correct"] += row["outcome_reward"]
        counts["truncated"] += not row["terminated"]
        counts["terminated_unparsed"] += row["terminated"] and row["status"] not in VALID
        counts["scored"] += scored
        counts["crlf_raw_rows"] += view["crlf_replacements"] > 0
        counts["crlf_changes_step_count_rows"] += len(view["steps"]) != len([s for s in row["completion"].split("\n\n") if s.strip()])
        counts["raw_no_double_newline"] += row["raw_no_double_newline"]
        counts["canonical_single_step_rows"] += len(view["steps"]) == 1
        counts["canonical_empty_rows"] += not view["steps"]
        status[row["status"]] += 1
        data["completion_tokens"].append(row["completion_tokens"])
        data["canonical_text_step_count_all_rows"].append(len(view["steps"]))
        data["absolute_actual_advantage"].append(abs(row["advantage"]))
        if scored:
            data["PRM_min_score_scored_only"].append(row["min_score"])
            data["PRM_step_score_scored_only"].extend(row["step_scores"])
            data["PRM_min_score_correct" if row["outcome_reward"] else "PRM_min_score_wrong"].append(row["min_score"])
            data["native_scored_step_count"].append(row["step_count"])
            data["prm_input_tokens_scored_only"].append(row["prm_input_tokens"])
            counts["native_alignment_rows_checked"] += 1
        tokens += row["completion_tokens"]
        bucket = (tokens - 1) // 1_000_000
        curve[bucket].update(rows=1, correct=row["outcome_reward"], truncated=not row["terminated"],
                             completion_tokens=row["completion_tokens"])
        if scored:
            curve[bucket].update(scored=1, prm_min_sum=row["min_score"], native_steps=row["step_count"])
    for start in range(0, len(rows), GROUP):
        group = rows[start:start + GROUP]
        require(len({r["problem_id"] for r in group}) == 1, arm + " group problem IDs differ")
        require(len({r["optimizer_step_before_update"] for r in group}) == 1, arm + " group update IDs differ")
        require(all(r["optimizer_step_before_update"] == start // 32 for r in group), arm + " unexpected update sequence")
        order.append(group[0]["problem_id"])
        d = group_decomposition(group)
        kind = d["kind"]
        c, accum = group_counts[kind], group_data[kind]
        c.update(groups=1, rows=GROUP)
        e = d["reference_max_absolute_difference"]
        maximum_error = max(maximum_error, e)
        # Exact GPU FP32 equality is covered by the separately saved live guard.
        # Here roundoff error grows with reward mean / a near-zero denominator.
        tolerance = 16 * (2 ** -23) * max(1., max(abs(f32(r["reward"])) for r in group)) / (d["reward_sample_std"] + EPS)
        c["FP64_reference_groups_outside_roundoff_bound"] += e > tolerance
        mismatching_reference += e > tolerance
        accum["FP64_reference_absolute_error"].append(e)
        accum["FP64_reference_roundoff_allowance"].append(tolerance)
        accum["reward_sample_std"].append(d["reward_sample_std"])
        accum["normalization_gain"].append(d["normalization_gain"])
        accum["effective_process_coefficient"].append(d["effective_process_coefficient"] if arm == "B" else 0.)
        accum["absolute_actual_advantage"].extend(abs(v) for v in d["actual"])
        accum["absolute_outcome_term_shared_denominator"].extend(abs(v) for v in d["outcome_term"])
        accum["absolute_process_term_shared_denominator"].extend(abs(v) for v in d["process_term"])
        accum["absolute_residual_vs_outcome_only"].extend(abs(v) for v in d["residual_vs_outcome_only"])
        c["nonzero_advantage_rows"] += sum(abs(v) > 1e-7 for v in d["actual"])
        c["wrong_positive_advantage_rows"] += sum(r["outcome_reward"] == 0 and a > 1e-7 for r,a in zip(group,d["actual"]))
        c["correct_negative_advantage_rows"] += sum(r["outcome_reward"] == 1 and a < -1e-7 for r,a in zip(group,d["actual"]))
        c["process_term_abs_exceeds_outcome_term_rows"] += sum(abs(p) > abs(y) + 1e-7 for p,y in zip(d["process_term"],d["outcome_term"]))
        c["nonzero_outcome_sign_reversals"] += sum(abs(ay)>1e-7 and a*ay < -1e-7 for ay,a in zip(d["outcome_only"],d["actual"]))
        c["normalization_gain_over_one_groups"] += d["normalization_gain"] > 1.
        scored = [r for r in group if r["valid_for_process_reward"]]
        c["all_rows_scored_groups"] += len(scored) == GROUP
        c["no_rows_scored_groups"] += len(scored) == 0
        c["mixed_scoring_eligibility_groups"] += 0 < len(scored) < GROUP
        c["scored_wrong_positive_advantage_rows"] += sum(
            r["valid_for_process_reward"] and not r["outcome_reward"] and a > 1e-7
            for r,a in zip(group,d["actual"]))
        c["unscored_negative_advantage_rows"] += sum(
            not r["valid_for_process_reward"] and a < -1e-7 for r,a in zip(group,d["actual"]))
        if scored:
            minima = [r["min_score"] for r in scored]
            accum["scored_only_within_group_min_score_range"].append(max(minima)-min(minima))
        positive = [r["min_score"] for r in scored if r["outcome_reward"]]
        negative = [r["min_score"] for r in scored if not r["outcome_reward"]]
        c["correct_wrong_scored_pairs"] += len(positive) * len(negative)
        c["PRM_wrong_outranks_correct_pairs"] += sum(n > p for p in positive for n in negative)
        c["PRM_correct_wrong_tied_pairs"] += sum(n == p for p in positive for n in negative)
    require(not mismatching_reference, arm + " saved advantages outside FP64/FP32 audit roundoff bounds")
    group_report = {}
    for kind in KINDS:
        c = group_counts[kind]
        values = group_data[kind]
        process_mass = math.fsum(values["absolute_process_term_shared_denominator"])
        outcome_mass = math.fsum(values["absolute_outcome_term_shared_denominator"])
        group_report[kind] = {"counts": dict(c), "metrics": {k: describe(v) for k,v in values.items()},
            "group_fraction": fraction(c["groups"], len(rows)//GROUP),
            "process_share_of_sum_absolute_components": fraction(process_mass, process_mass+outcome_mass),
            "PRM_wrong_outranks_correct_pair_fraction": fraction(c["PRM_wrong_outranks_correct_pairs"], c["correct_wrong_scored_pairs"])}
    bins = []
    for bucket, c in sorted(curve.items()):
        bins.append({"completion_token_bin": [bucket*1_000_000, (bucket+1)*1_000_000],
                     "counts": dict(c), "outcome_rate": fraction(c["correct"], c["rows"]),
                     "truncation_rate": fraction(c["truncated"], c["rows"]),
                     "mean_completion_tokens": fraction(c["completion_tokens"], c["rows"]),
                     "PRM_mean_min_scored_only": fraction(c["prm_min_sum"], c["scored"])})
    all_abs = math.fsum(data["absolute_actual_advantage"])
    homogeneous_abs = math.fsum(group_data[kind]["absolute_actual_advantage"][i]
        for kind in ("all_correct", "all_wrong")
        for i in range(len(group_data[kind]["absolute_actual_advantage"])))
    return {"arm": arm, "completion_tokens": tokens, "counts": dict(counts), "status_counts": dict(status),
            "outcome_rate": fraction(counts["correct"], len(rows)),
            "invalid_output_fraction": fraction(len(rows)-counts["valid_outputs"], len(rows)),
            "truncation_fraction": fraction(counts["truncated"], len(rows)),
            "metrics": {k:describe(v) for k,v in data.items()}, "groups": group_report,
            "homogeneous_group_share_of_total_absolute_actual_advantage": fraction(homogeneous_abs, all_abs),
            "FP64_reference_max_absolute_difference": maximum_error,
            "course_of_training": bins}, order


def load_arm(root, relative, arm):
    run = (root / relative).resolve()
    require(run.is_relative_to(root), "Run path escapes root")
    files = {name: run / name for name in ["status.json", "run_config.json", "rollouts.jsonl", "initial_sampling.json", "live_audits.json"]}
    for path in files.values():
        require(path.is_file(), "Missing required completed input: " + str(path))
    before = {name: sha(path) for name,path in files.items()}
    state, config, guard = read(files["status.json"]), read(files["run_config.json"]), read(files["live_audits.json"])
    require(state["status"] == "completed" and not state.get("pilot"), arm + " still pending or pilot; completed input required")
    require(state["arm"] == config["arm"] == arm, arm + " arm mismatch")
    require(config["seed"] == 17 and config["target_completion_tokens"] == 6_000_000, arm + " wrong protocol")
    require(config["source_checkpoint"] is None and config["base_actor_initialization"], arm + " not base initialization")
    require(guard["passed"] and guard["original_TRL_advantages_retained"] and guard["loss_input_values_preserved"], arm + " failed live guard")
    rows = read_lines(files["rollouts.jsonl"])
    report, order = analyze_rows(rows, arm)
    require(report["completion_tokens"] == state["accounting"]["generated_completion_tokens"], arm + " token accounting mismatch")
    require(len(rows) == state["accounting"]["generated_completions"] == guard["loss_rows"], arm + " row accounting mismatch")
    require(6_000_000 <= report["completion_tokens"] < 6_000_000 + 32*2048, arm + " budget mismatch")
    require(len(rows)//32 == state["global_step"] == guard["global_step"], arm + " update count mismatch")
    if arm == "B":
        require(guard["PRM_frozen"] and state["prm_frozen_verified"], "B frozen PRM check missing")
    initial = read(files["initial_sampling.json"])
    require(initial == [{k:r[k] for k in ["problem_id", "completion", "completion_tokens"]} for r in rows[:32]], arm + " initial samples differ from ledger")
    after = {name:sha(path) for name,path in files.items()}
    require(before == after, arm + " completed inputs changed during analysis")
    report["input_hashes"] = {str(path.relative_to(root)):before[name] for name,path in files.items()}
    report["global_steps"] = state["global_step"]
    return report, order, config, initial


def analyze_pair(root, a_run, b_run):
    a, a_order, ac, ai = load_arm(root, a_run, "A")
    b, b_order, bc, bi = load_arm(root, b_run, "B")
    shared_fields = ["seed", "base_actor_initialization", "source_checkpoint", "actor_adapter_initial_sha256",
        "packages", "model_revision", "config_sha256", "data_manifest_sha256", "shared_training_arguments",
        "verifier_version", "unit_metadata_sha256", "advantage", "raw_vllm_engine_seed", "source_release_sha256"]
    shared = {key: ac[key] == bc[key] for key in shared_fields}
    require(all(shared.values()), "A/B shared configuration differs: " + str([k for k,v in shared.items() if not v]))
    require(ai == bi, "A/B first generation batch not identical")
    order = prefix_comparison(a_order, b_order)
    require(order["overlap_prompt_order_equal"], "A/B common prompt order diverged")
    differences = {key: b[key] - a[key] for key in ["outcome_rate", "invalid_output_fraction", "truncation_fraction"]}
    differences["mean_completion_tokens"] = b["metrics"]["completion_tokens"]["mean"] - a["metrics"]["completion_tokens"]["mean"]
    differences["mean_canonical_text_step_count"] = b["metrics"]["canonical_text_step_count_all_rows"]["mean"] - a["metrics"]["canonical_text_step_count_all_rows"]["mean"]
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "completed_training_signals_cpu_only", "passed": True,
        "analyst_script_sha256": sha(__file__), "A": a, "B": b,
        "paired_controls": {"shared_configuration_checks": shared, "first_32_samples_identical": ai == bi,
                            "prompt_order": order},
        "B_minus_A_descriptive_training_differences": differences,
        "definitions": {
            "decomposition": "actual advantage approximately equals centered(outcome)/(sample_std(float32 reward)+1e-4) plus centered(float32 reward-outcome)/(same denominator); approximation is FP32 GPU versus CPU FP64 rounding",
            "process_residual": "actual mixed-reward advantage minus outcome-only standardized advantage; includes both the additive process component and the changed denominator",
            "amplification": "normalization_gain = 1/(sample_std(float32 reward)+1e-4); effective_process_coefficient = 0.2*normalization_gain; neither quantity is a new applied coefficient",
            "PRM_pair_proxy": "within-question scored correct/wrong pairs only; wrong-outranks-correct is an outcome-disagreement proxy, not a labeled mathematical error",
            "binning": "rows are assigned by cumulative completion tokens at row end; a row crossing a 1M boundary is wholly assigned to its end bin"},
        "limitations": [
            "Saved training outcomes are observed on evolving different policies; A/B rollout averages are descriptive and cannot replace held-out greedy evaluation or establish a causal accuracy effect.",
            "A positive advantage for an all-wrong answer or a negative advantage for an all-correct answer follows relative group ranking and is not automatically a mathematical PRM error.",
            "Within all-wrong groups, some signal can distinguish valid scored outputs from invalid/unscored zero placeholders. Mixed scoring eligibility and scored-only score ranges are reported separately from variation among scored mathematical derivations.",
            "No step-level human ground truth is present. Wrong final answers can contain useful steps, and correct final answers can contain errors. These statistics cannot count genuine false punishments or missed mathematical errors.",
            "Invalid/unscored min_score=0 values are original reward placeholders. They are used in reward decomposition but excluded from all PRM score distributions and correct/wrong ranking proxies.",
            "Raw-to-canonical spans and score count are independently checked. Native token marker positions/logits were not saved per response and cannot be independently regenerated without tokenizer/model inference; the existing pinned runtime checks are the evidence for that layer.",
            "The CPU reference allows documented FP32 rounding. Exact GPU tensor retention and loss coverage are checked using the saved live audit, not recomputed gradients.",
            "No new model loading, scoring, training, reward transformation, loss change, or candidate objective is performed.",
            "One training seed does not quantify training stochasticity. Repeated within-question rows are not independent samples; no unpaired significance claim is made."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a-run", default="runs/A_corrected_seed17_6M_20261022")
    parser.add_argument("--b-run", default="runs/B_corrected_seed17_6M_20261022")
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; refuses overwrite")
    args = parser.parse_args()
    require(not args.output.exists(), "Output exists; choose a new evidence path")
    result = analyze_pair(args.root.resolve(), args.a_run, args.b_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "output": str(args.output.resolve()),
                      "sha256": sha(args.output), "A_tokens": result["A"]["completion_tokens"],
                      "B_tokens": result["B"]["completion_tokens"]}))


if __name__ == "__main__":
    main()
