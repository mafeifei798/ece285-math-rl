"""Descriptive matched-run audit; stored data only, never trains a model."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
import numpy as np
from prepare import ROOT, OUT, INPUTS, read, rows, save, sha

ARMS = {"A": "A_seed17", "B_old": "B_seed17_6M", "B7": "B_Qwen7B_seed17_6M"}
EPS = 1e-4


def average(values):
    return float(np.mean(values)) if len(values) else None


def rate(n, total):
    return n / total if total else None


def failure(row, y):
    status = row["status"]
    if y:
        return "correct"
    if status == "truncated":
        return "truncated"
    if status == "missing_or_invalid_box":
        return "missing_or_invalid_box"
    if status == "unparseable":
        return "unparseable"
    if status.startswith("verifier_error"):
        return "verifier_error"
    assert status in {"exact", "symbolic"}, status
    return "parsed_but_wrong"


def augment(records, arm):
    total, batch_before, batches = 0, {}, {}
    for i, row in enumerate(records):
        step = row["optimizer_step_before_update"]
        batches.setdefault(step, []).append(row)
        row["source_row"] = i
    for step, batch in sorted(batches.items()):
        assert len(batch) == 32
        end = total + sum(r["completion_tokens"] for r in batch)
        phase = min(5, int((total + end) / 2 / 1e6))
        for r in batch:
            r["phase"] = phase
            r["token_midpoint"] = (total + end) / 2
            r["y"] = r.get("outcome_reward", r["reward"])
            r["failure_type"] = failure(r, r["y"])
            r["raw_steps"] = [s.strip() for s in r["completion"].split("\n\n") if s.strip()]
            r["raw_step_count"] = len(r["raw_steps"])
            normalized = r["completion"].replace("\r\n", "\n")
            r["crlf_normalized_step_count"] = len([s for s in normalized.split("\n\n") if s.strip()])
            r["missing_internal_separator"] = "\n\n" not in r["completion"].strip()
            r["has_crlf"] = "\r\n" in r["completion"]
            r["crlf_merges_steps"] = r["crlf_normalized_step_count"] != r["raw_step_count"]
            r["scored"] = arm != "A" and r["valid_for_process_reward"]
        total = end
    return batches, total


def summarize(records):
    n = len(records)
    scored = [r for r in records if r["scored"]]
    complete = [r for r in records if r["terminated"]]
    counts = Counter(r["failure_type"] for r in records)
    correct_scored = [r for r in scored if r["y"]]
    wrong_scored = [r for r in scored if not r["y"]]
    return {"n": n, "correct": int(sum(r["y"] for r in records)),
        "outcome_accuracy": average([r["y"] for r in records]), "failure_counts": dict(counts),
        "failure_rates": {k: v / n for k,v in counts.items()},
        "completion_tokens_mean": average([r["completion_tokens"] for r in records]),
        "completion_tokens_median": float(np.median([r["completion_tokens"] for r in records])) if n else None,
        "raw_steps_mean": average([r["raw_step_count"] for r in records]),
        "complete_steps_mean": average([r["raw_step_count"] for r in complete]),
        "scored_steps_mean": average([r["raw_step_count"] for r in scored]),
        "raw_single_step_rate": average([r["raw_step_count"] == 1 for r in records]),
        "missing_separator_n": sum(r["missing_internal_separator"] for r in records),
        "missing_separator_rate": average([r["missing_internal_separator"] for r in records]),
        "scored_single_step_rate": average([r["raw_step_count"] == 1 for r in scored]),
        "crlf_n": sum(r["has_crlf"] for r in records),
        "crlf_merges_steps_n": sum(r["crlf_merges_steps"] for r in records),
        "crlf_merges_steps_rate": average([r["crlf_merges_steps"] for r in records]),
        "scored_n": len(scored), "min_score_mean_scored": average([r["min_score"] for r in scored]),
        "min_score_median_scored": float(np.median([r["min_score"] for r in scored])) if scored else None,
        "all_native_step_score_mean": average([p for r in scored for p in r["step_scores"]]),
        "min_score_mean_correct": average([r["min_score"] for r in correct_scored]),
        "min_score_mean_wrong": average([r["min_score"] for r in wrong_scored]),
        "mixed_reward_mean": average([r["reward"] for r in records]),
        "correct_answer_min_below_0p5_n": sum(r["min_score"] < .5 for r in correct_scored),
        "wrong_answer_min_ge_0p5_n": sum(r["min_score"] >= .5 for r in wrong_scored),
        "scored_correct_n": len(correct_scored), "scored_wrong_n": len(wrong_scored)}


def groups(records, arm, actual=None, denominators=None):
    result = []
    for offset in range(0, len(records), 8):
        group = records[offset:offset + 8]
        assert len(group) == 8 and len({r["problem_id"] for r in group}) == 1
        assert len({r["optimizer_step_before_update"] for r in group}) == 1
        y = np.array([r["y"] for r in group], dtype=np.float64)
        r = np.array([r["reward"] for r in group], dtype=np.float32).astype(np.float64)
        den = float(np.std(r, ddof=1) + EPS)
        if denominators is not None:
            den = denominators[offset // 8]
        reference = (r - np.mean(r)) / den
        adv = actual[offset:offset + 8] if actual is not None else reference
        ay = (y - y.mean()) / den
        ap = ((r - y) - (r - y).mean()) / den
        assert np.max(np.abs(ay + ap - reference)) < 1e-10
        expected = (y - y.mean()) / (y.std(ddof=1) + EPS)
        success = int(y.sum())
        kind = "all_correct" if success == 8 else "all_wrong" if success == 0 else "mixed"
        for i, row in enumerate(group):
            row.update(actual_or_replayed_advantage=float(adv[i]), outcome_component=float(ay[i]),
                process_component=float(ap[i]), outcome_only_counterfactual_advantage=float(expected[i]),
                outcome_group=kind, group_index=offset // 8)
        result.append({"group_index": offset // 8, "problem_id": group[0]["problem_id"],
            "optimizer_step": group[0]["optimizer_step_before_update"], "phase": group[0]["phase"],
            "kind": kind, "invalid_n": sum(not r["scored"] for r in group) if arm != "A" else sum(r["status"] not in {"exact","symbolic"} for r in group),
            "mean_abs_advantage": float(np.abs(adv).mean()),
            "outcome_component_L1": float(np.abs(ay).sum()), "process_component_L1": float(np.abs(ap).sum()),
            "total_advantage_L1": float(np.abs(adv).sum()), "zero_advantages": bool(np.max(np.abs(adv)) < 1e-6),
            "positive_advantage_wrong": int(((adv > 1e-6) & (y == 0)).sum()),
            "negative_advantage_correct": int(((adv < -1e-6) & (y == 1)).sum()),
            "actual_vs_float64_max_error": float(np.max(np.abs(adv-reference)))})
    return result


def group_summary(group_rows):
    def summarize_group(g):
        out = sum(r["outcome_component_L1"] for r in g)
        proc = sum(r["process_component_L1"] for r in g)
        return {"groups": len(g), "fraction": len(g) / len(group_rows),
            "all_valid_groups": sum(r["invalid_n"] == 0 for r in g),
            "has_invalid_groups": sum(r["invalid_n"] > 0 for r in g),
            "mean_abs_advantage": average([r["mean_abs_advantage"] for r in g]),
            "outcome_component_L1": out, "process_component_L1": proc,
            "process_component_abs_share": proc / (proc + out) if proc + out else 0,
            "zero_advantage_groups": sum(r["zero_advantages"] for r in g),
            "positive_advantage_wrong": sum(r["positive_advantage_wrong"] for r in g),
            "negative_advantage_correct": sum(r["negative_advantage_correct"] for r in g)}
    return {"all": summarize_group(group_rows), **{kind: summarize_group([r for r in group_rows if r["kind"] == kind])
            for kind in ["mixed", "all_correct", "all_wrong"]}}


def evaluations():
    result, predictions = {}, {}
    paths = {"A": "runs/A_seed17/evaluation", "B_old": "runs/B_seed17_6M/milestones/6000000/evaluation",
             "B7": "runs/B_Qwen7B_seed17_6M/milestones/6000000/evaluation"}
    for arm, relative in paths.items():
        result[arm] = {}
        for dataset in ["math500", "gsm8k"]:
            records = rows(INPUTS / relative / (dataset + ".jsonl"))
            for row in records:
                row["failure_type"] = failure(row, row["reward"])
                row["completion_tokens"] = row["generated_tokens"]
                row["raw_step_count"] = len([s for s in row["completion"].split("\n\n") if s.strip()])
            predictions[(arm,dataset)] = {r["problem_id"]:r for r in records}
            counts = Counter(r["failure_type"] for r in records)
            result[arm][dataset] = {"n":len(records),"correct":counts["correct"],"accuracy":counts["correct"]/len(records),
                "failure_counts":dict(counts), "mean_tokens":average([r["completion_tokens"] for r in records]),
                "mean_steps":average([r["raw_step_count"] for r in records]),
                "missing_separator_rate":average(["\n\n" not in r["completion"].strip() for r in records])}
    attribution = {}
    for other in ["A","B_old"]:
        attribution["B7_minus_"+other] = {}
        for dataset in ["math500","gsm8k"]:
            new, old = predictions[("B7",dataset)],predictions[(other,dataset)]
            assert new.keys() == old.keys()
            lost, gained = [], []
            for ident in new:
                assert new[ident]["gold"] == old[ident]["gold"] and new[ident]["question_sha256"] == old[ident]["question_sha256"]
                if old[ident]["reward"] == 1 and new[ident]["reward"] == 0:lost.append(new[ident])
                if old[ident]["reward"] == 0 and new[ident]["reward"] == 1:gained.append(old[ident])
            keys = set(result["B7"][dataset]["failure_counts"]) | set(result[other][dataset]["failure_counts"])
            net = {k:result["B7"][dataset]["failure_counts"].get(k,0)-result[other][dataset]["failure_counts"].get(k,0)
                   for k in sorted(keys) if k != "correct"}
            assert sum(net.values()) == len(lost)-len(gained)
            attribution["B7_minus_"+other][dataset] = {"other_correct_B7_wrong":len(lost),"B7_correct_other_wrong":len(gained),
                "lost_by_B7_failure":dict(Counter(r["failure_type"] for r in lost)),
                "gained_by_other_failure":dict(Counter(r["failure_type"] for r in gained)),"net_increase_failures":net,
                "net_correctness_flip_by_failure":{k:sum(r["failure_type"]==k for r in lost)-sum(r["failure_type"]==k for r in gained)
                    for k in ["parsed_but_wrong","truncated","missing_or_invalid_box","unparseable","verifier_error"]},
                "lost_problem_ids":[r["problem_id"] for r in lost],"gained_problem_ids":[r["problem_id"] for r in gained]}
    return result, attribution


def complete_boxes(text):
    contents=[]
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{",text):
        start=match.end();depth=1
        for i in range(start,len(text)):
            if text[i]=="{":depth+=1
            elif text[i]=="}":depth-=1
            if depth==0:
                content=text[start:i].strip()
                if content:contents.append(content)
                break
    return contents


def box_statistics(records, arm, is_eval=False):
    exact=lambda s:re.sub(r"\s+","",unicodedata.normalize("NFKC",s))
    multiple,correct_before_failure,examples=0,0,[]
    for index,r in enumerate(records):
        boxes=complete_boxes(r["completion"])
        multiple+=len(boxes)>1
        y=r.get("y",r.get("outcome_reward",r["reward"]))
        if not y and any(exact(b)==exact(r["gold"]) for b in boxes):
            correct_before_failure+=1
            examples.append({"source_row":index,"problem_id":r["problem_id"],"status":r["status"],
                "gold":r["gold"],"logged_answer":r["answer"],"nonempty_boxes":boxes})
    return {"n":len(records),"multiple_nonempty_box_rows":multiple,"multiple_box_rate":multiple/len(records),
        "exact_gold_box_present_but_logged_incorrect":correct_before_failure,"examples":examples,
        "caution":"String-exact lower bound only; not a substitute verifier. Multiple boxes alone do not imply a failure. A matching early box does not establish valid reasoning or complete termination."}


def main():
    all_rows, all_groups = {}, {}
    checks = {}
    for arm, name in ARMS.items():
        records = rows(INPUTS / "runs" / name / "rollouts.jsonl")
        batches, total = augment(records, arm)
        status = read(INPUTS / "runs" / name / "status.json")
        assert total == status["accounting"]["generated_completion_tokens"]
        assert len(records) == status["accounting"]["generated_completions"]
        assert sum(r["y"] for r in records) == status["accounting"]["correct_completions"]
        score_errors = []
        for r in records:
            if arm == "A":
                assert r["reward"] == r["y"]
            elif r["scored"]:
                assert len(r["step_scores"]) == r["step_count"] == r["raw_step_count"]
                assert all(math.isfinite(p) and 0 <= p <= 1 for p in r["step_scores"])
                assert min(r["step_scores"]) == r["min_score"]
                assert abs(r["reward"] - r["y"] - .2*r["min_score"]) < 1e-12
            else:
                assert r["step_scores"] == [] and r["step_count"] == 0 and r["min_score"] == r["reward"] == r["y"] == 0
        actual, denominators = None, None
        if arm == "B7":
            receipts = [read(p) for p in (INPUTS / "runs" / name / "advantage_audits").glob("*.json")]
            audit_by_step = {}
            for receipt in receipts:
                assert receipt["passed"] and receipt["changes_actual_advantages"] is False
                # Filename prefix is the completed optimizer count before generation.
            for path in (INPUTS / "runs" / name / "advantage_audits").glob("*.json"):
                step = int(path.name.split("_")[0]);audit_by_step[step]=read(path)
            assert len(audit_by_step)==status["global_step"]
            actual, denominators = [], []
            for step, batch in sorted(batches.items()):
                a = audit_by_step[step]
                assert a["problem_ids"] == [r["problem_id"] for r in batch]
                assert a["rewards"] == np.asarray([r["reward"] for r in batch],dtype=np.float32).tolist()
                actual += a["actual_advantages"]
                denominators += [s+EPS for s in a["reference_sample_std"]]
            actual = np.asarray(actual)
            checks[arm+"_actual_advantage_audits"] = len(audit_by_step)
        group_rows = groups(records, arm, actual, denominators)
        all_rows[arm],all_groups[arm]=records,group_rows
        checks[arm] = {"rows":len(records),"groups":len(group_rows),"exact_reward_and_accounting":True,
                       "max_stored_advantage_vs_float64_error":max(g["actual_vs_float64_max_error"] for g in group_rows)}
    shared = min(len(g) for g in all_groups.values())
    for i in range(shared):
        assert len({all_groups[a][i]["problem_id"] for a in ARMS})==1
    result = {"scope":"Descriptive audit of stored seed17 6M runs; no causal attribution from online trends alone",
        "implementation_checks":checks,"train":{a:summarize(r) for a,r in all_rows.items()},
        "phases":{a:[{"bin_million_completion_tokens":[p,p+1],**summarize([r for r in rec if r["phase"]==p])} for p in range(6)] for a,rec in all_rows.items()},
        "groups":{a:group_summary(g) for a,g in all_groups.items()},
        "matched_training_question_groups":{"n":shared,"same_question_order":True,
            "arms":{a:summarize(r[:shared*8]) for a,r in all_rows.items()}},
        "matched_question_windows":{a:[{"group_range":[lo,min(hi,shared)],**summarize(r[lo*8:min(hi,shared)*8])}
            for lo,hi in [(0,200),(shared-200,shared)]] for a,r in all_rows.items()},
        "advantage_share_definition":"Pooled sum(abs(process_component)) / (sum(abs(process_component))+sum(abs(outcome_component))). Components use the SAME total-reward standard deviation. This is advantage amplitude, NOT gradient contribution or causal credit. New B7 uses actual saved GPU advantages/denominators; A and old B use independent FP64 replay of FP32 rewards.",
        "phase_definition":"6 descriptive bins by optimizer batch midpoint in cumulative generated completion tokens; keep each 32-response batch intact. Online cohorts differ, so apparent trends do not prove hacking.",
        "process_score_denominator":"Only complete, reliably parsed outputs actually scored by PRM; invalid placeholders are excluded.",
        "negative_advantage_caution":"Correct final answers may have erroneous reasoning; wrong final answers may have useful partial reasoning. Advantage sign alone is not an error label."}
    result["evaluation"],result["evaluation_failure_attribution"]=evaluations()
    result["box_patterns"]={a:box_statistics(rec,a) for a,rec in all_rows.items()}
    eval_paths={"A":"runs/A_seed17/evaluation","B_old":"runs/B_seed17_6M/milestones/6000000/evaluation","B7":"runs/B_Qwen7B_seed17_6M/milestones/6000000/evaluation"}
    result["evaluation_box_patterns"]={a:{d:box_statistics(rows(INPUTS/rel/(d+".jsonl")),a,True)
        for d in ["math500","gsm8k"]} for a,rel in eval_paths.items()}
    result["scored_step_length_buckets"]={a:[{"raw_steps":[lo,hi],**summarize([r for r in rec if r["scored"] and lo<=r["raw_step_count"]<=hi])}
        for lo,hi in [(1,1),(2,4),(5,9),(10,19),(20,9999)]] for a,rec in all_rows.items() if a!="A"}
    save(OUT / "statistics.json",result)
    save(OUT / "group_statistics.json",all_groups)
    with (OUT / "training_rows.csv").open("w") as stream:
        columns=["arm","source_row","problem_id","optimizer_step_before_update","phase","y","failure_type","completion_tokens","raw_step_count","crlf_normalized_step_count","missing_internal_separator","has_crlf","scored","min_score","outcome_group","actual_or_replayed_advantage","outcome_component","process_component"]
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore");writer.writeheader()
        for a,rec in all_rows.items():
            for r in rec:writer.writerow({"arm":a,**r})
    print(json.dumps({"train":result["train"],"groups":result["groups"],"evaluation":result["evaluation"],
                     "failure_attribution":result["evaluation_failure_attribution"],"checks":checks},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
