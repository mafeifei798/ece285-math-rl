"""Reward/rank/actual-TRL advantage replay on saved B answers; no training."""
import csv
import itertools
import json
import math
from pathlib import Path
import sys
import time

from trl_kernel import actual_trl_advantages, save_provenance
from v2core import (advantages_like_trl, atomic_json, audit_advantages, midranks, mixed_reward,
                    read_jsonl, sha256, valid_for_prm, verify_frozen_inputs, write_jsonl)

VARIANTS = {
    "logged": ("logged", "logged"),
    "logged_labels_replayed_scores": ("logged", "legacy"),
    "legacy_replayed": ("legacy_recomputed", "legacy"),
    "units_only": ("v2_outcome", "legacy"),
    "split_only": ("legacy_recomputed", "normalized"),
    "combined": ("v2_outcome", "normalized"),
}


def sign(x):
    return 0 if abs(x) <= 1e-6 else 1 if x > 0 else -1


def pair_order(a, b):
    return sign(a - b)


def group_type(ys):
    count = sum(ys)
    return "all_correct" if count == 8 else "all_wrong" if count == 0 else "mixed"


def percentile(values, p):
    if not values:
        return None
    a = sorted(values)
    k = (len(a) - 1) * p
    i = int(k)
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (k - i)


def summarize_difference(old, new, retained_groups, indices=None):
    if indices is None:
        indices = list(range(len(old)))
    indices = [i for i in indices if i // 8 in retained_groups]
    groups = sorted({i // 8 for i in indices})
    reward_changes = [new[i]["reward"] - old[i]["reward"] for i in indices]
    adv_changes = [new[i]["advantage"] - old[i]["advantage"] for i in indices]
    reward_changed = lambda i: abs(new[i]["reward"] - old[i]["reward"]) > 1e-6
    av_changed = lambda i: abs(new[i]["advantage"] - old[i]["advantage"]) > 1e-6
    pair_changes, pair_denominator, top_changes, ordering_groups = 0, 0, 0, 0
    process_pairs, process_pair_denominator, process_ordering_groups = 0, 0, 0
    for g in groups:
        ix = list(range(g * 8, g * 8 + 8))
        this_changed = False
        for i, j in itertools.combinations(ix, 2):
            pair_denominator += 1
            changed = pair_order(old[i]["reward_fp32"], old[j]["reward_fp32"]) != pair_order(new[i]["reward_fp32"], new[j]["reward_fp32"])
            pair_changes += changed
            this_changed |= changed
        ordering_groups += this_changed
        old_max, new_max = max(old[i]["reward_fp32"] for i in ix), max(new[i]["reward_fp32"] for i in ix)
        old_top = {i for i in ix if abs(old[i]["reward_fp32"] - old_max) <= 1e-6}
        new_top = {i for i in ix if abs(new[i]["reward_fp32"] - new_max) <= 1e-6}
        top_changes += old_top != new_top
        valid_ix = [i for i in ix if old[i]["process_scored"] and new[i]["process_scored"]]
        process_changed = False
        for i, j in itertools.combinations(valid_ix, 2):
            process_pair_denominator += 1
            changed = pair_order(old[i]["min_score"], old[j]["min_score"]) != pair_order(new[i]["min_score"], new[j]["min_score"])
            process_pairs += changed
            process_changed |= changed
        process_ordering_groups += process_changed
    process_ix = [i for i in indices if old[i]["process_scored"] and new[i]["process_scored"]]
    return {"rows": len(indices), "groups": len(groups),
            "outcome_label_changed_rows": sum(old[i]["outcome_reward"] != new[i]["outcome_reward"] for i in indices),
            "outcome_0_to_1": sum(old[i]["outcome_reward"] == 0 and new[i]["outcome_reward"] == 1 for i in indices),
            "outcome_1_to_0": sum(old[i]["outcome_reward"] == 1 and new[i]["outcome_reward"] == 0 for i in indices),
            "reward_changed_rows_gt_1e_minus_6": sum(reward_changed(i) for i in indices),
            "reward_changed_groups": sum(any(reward_changed(i) for i in range(g * 8, g * 8 + 8)) for g in groups),
            "mean_reward_delta": sum(reward_changes) / len(indices) if indices else None,
            "mean_abs_reward_delta": sum(abs(x) for x in reward_changes) / len(indices) if indices else None,
            "max_abs_reward_delta": max(map(abs, reward_changes), default=0),
            "process_scored_both_rows": len(process_ix),
            "process_min_changed_rows_gt_1e_minus_6": sum(abs(new[i]["min_score"] - old[i]["min_score"]) > 1e-6 for i in process_ix),
            "step_count_changed_rows": sum(new[i]["step_count"] != old[i]["step_count"] for i in process_ix),
            "mean_process_min_delta_on_scored_both": sum(new[i]["min_score"] - old[i]["min_score"] for i in process_ix) / len(process_ix) if process_ix else None,
            "reward_pairwise_order_changes": pair_changes, "reward_pairwise_pairs": pair_denominator,
            "reward_order_changed_groups": ordering_groups, "reward_top_set_changed_groups": top_changes,
            "process_pairwise_order_changes": process_pairs, "process_pairwise_pairs": process_pair_denominator,
            "process_order_changed_groups": process_ordering_groups,
            "advantage_changed_rows_gt_1e_minus_6": sum(av_changed(i) for i in indices),
            "advantage_changed_rows_gt_0p01": sum(abs(new[i]["advantage"] - old[i]["advantage"]) > .01 for i in indices),
            "advantage_sign_flip_rows": sum(sign(old[i]["advantage"]) != sign(new[i]["advantage"]) for i in indices),
            "advantage_nonzero_sign_flip_rows": sum(sign(old[i]["advantage"]) * sign(new[i]["advantage"]) == -1 for i in indices),
            "mean_abs_advantage_delta": sum(abs(x) for x in adv_changes) / len(indices) if indices else None,
            "p95_abs_advantage_delta": percentile(list(map(abs, adv_changes)), .95),
            "max_abs_advantage_delta": max(map(abs, adv_changes), default=0)}


def main():
    source, out, review_file, trl_source = map(Path, sys.argv[1:5])
    if (out / "replay_summary.json").exists():
        raise RuntimeError("Replay complete; inspect before repeating")
    verify_frozen_inputs(out, source)
    import torch
    torch.set_num_threads(4)
    started = time.monotonic()
    rows = read_jsonl(source)
    outcomes = read_jsonl(out / "outcomes.jsonl")
    paired = {r["source_row"]: r for r in read_jsonl(out / "paired_scores.jsonl")}
    reviews = json.loads(review_file.read_text())
    quarantine = json.loads((out / "quarantine.json").read_text())
    assert len(rows) == len(outcomes)
    for i, r in enumerate(rows):
        assert abs(r["reward"] - mixed_reward({"status": r["status"], "reward": r["outcome_reward"]}, r["step_scores"])) < 1e-12
    excluded = {i // 8 for i, r in enumerate(rows) if r["problem_id"] in quarantine}
    retained = set(range(len(rows) // 8)) - excluded
    arrays = {}
    guard = {"batches": 0, "same_device_failures": 0, "old_cross_device_fail_batches": 0,
             "max_old_cross_device_error": 0., "max_float64_difference": 0., "actual_advantages_mutated": False}
    save_provenance(trl_source, out / "replayed_trl_tensor_kernel.json")
    for name, (outcome_kind, score_kind) in VARIANTS.items():
        values = []
        for i, r in enumerate(rows):
            result = {"reward": r["outcome_reward"], "status": r["status"], "answer": r["answer"]} if outcome_kind == "logged" else outcomes[i][outcome_kind]
            valid = valid_for_prm(result)
            if not valid:
                p, count, origin = [], 0, "not_scored"
            elif score_kind != "logged" and i in paired:
                scored = paired[i][score_kind]
                p, count, origin = scored["step_scores"], scored["step_count"], "paired_frozen_BF16_" + score_kind
            else:
                assert r["valid_for_process_reward"], f"Missing fresh scores for newly eligible {i}"
                p, count, origin = r["step_scores"], r["step_count"], "unchanged_frozen_score_cache"
            value = mixed_reward(result, p)
            if name == "logged":
                assert value == r["reward"]
            values.append({"source_row": i, "problem_id": r["problem_id"], "status": result["status"],
                           "outcome_reward": result["reward"], "process_scored": valid,
                           "min_score": min(p) if valid else None, "step_count": count,
                           "reward": value, "score_provenance": origin, "quarantined": i // 8 in excluded})
        for offset in range(0, len(values), 32):
            batch = values[offset:offset + 32]
            tensor, actual = actual_trl_advantages([r["reward"] for r in batch], trl_source, "cuda:0")
            before = actual.clone()
            audit_advantages(tensor, actual, receipt_path=out / "advantage_audits" / name / f"batch_{offset // 32:04d}.json",
                             problem_ids=[r["problem_id"] for r in batch])
            assert torch.equal(before, actual)
            if name == "logged":
                cpu = torch.as_tensor([r["reward"] for r in batch], dtype=torch.float32).view(-1, 8)
                old_reference = ((cpu - cpu.mean(1, keepdim=True)) / (cpu.std(1, keepdim=True) + 1e-4)).reshape(-1)
                error = (actual.cpu() - old_reference).abs()
                old_close = torch.isclose(actual.cpu(), old_reference, atol=2e-5, rtol=2e-5)
                guard["old_cross_device_fail_batches"] += int(not old_close.all())
                guard["max_old_cross_device_error"] = max(guard["max_old_cross_device_error"], float(error.max()))
                cpu64 = advantages_like_trl(tensor.detach().cpu().double())
                guard["max_float64_difference"] = max(guard["max_float64_difference"], float((actual.detach().cpu().double() - cpu64).abs().max()))
            guard["batches"] += 1
            for r, av, reward32 in zip(batch, actual.cpu().tolist(), tensor.cpu().tolist()):
                r["advantage"] = av
                r["reward_fp32"] = reward32
        for offset in range(0, len(values), 8):
            batch = values[offset:offset + 8]
            ranks = midranks([r["reward_fp32"] for r in batch])
            valid = [j for j, r in enumerate(batch) if r["process_scored"]]
            process_ranks = dict(zip(valid, midranks([batch[j]["min_score"] for j in valid])))
            kind = group_type([r["outcome_reward"] for r in batch])
            for j, r in enumerate(batch):
                r.update(reward_rank=ranks[j], process_rank=process_ranks.get(j), group_type=kind,
                         all_group_members_valid=all(z["process_scored"] for z in batch))
        arrays[name] = values
        write_jsonl(out / "variants" / (name + ".jsonl"), values)
        print(json.dumps({"stage": "reward_and_advantage_replay", "variant": name, "rows": len(values)}), flush=True)
    unit_isolation = all(a["min_score"] == b["min_score"] and a["step_count"] == b["step_count"]
                         for a, b in zip(arrays["legacy_replayed"], arrays["units_only"]))
    split_isolation = all(a["outcome_reward"] == b["outcome_reward"] and a["process_scored"] == b["process_scored"]
                          for a, b in zip(arrays["legacy_replayed"], arrays["split_only"]))
    assert unit_isolation and split_isolation
    assert all(not r["process_scored"] or r["min_score"] is not None for a in arrays.values() for r in a)
    assert all(r["process_scored"] or (r["min_score"] is None and r["reward"] == 0) for a in arrays.values() for r in a)
    comparisons = {
        "backend_replay_control": ("logged", "logged_labels_replayed_scores"),
        "legacy_verifier_reproduction_control": ("logged_labels_replayed_scores", "legacy_replayed"),
        "units_only": ("legacy_replayed", "units_only"),
        "split_only": ("legacy_replayed", "split_only"),
        "combined_matched": ("legacy_replayed", "combined"),
        "combined_vs_historical_including_backend": ("logged", "combined"),
    }
    summary = {"rows": len(rows), "groups": len(rows) // 8, "new_training": False,
               "new_actor_samples": 0, "prm_parameter_updates": 0, "retained_groups": len(retained),
               "quarantined_groups": sorted(excluded), "quarantined_rows": 8 * len(excluded),
               "frozen_same_responses": True, "comparison_notes": {
                   "backend_replay_control": "Re-inference/batch-padding drift only, not a repair gain.",
                   "units_only": "Answer-unit rule on identical legacy scores and original groups.",
                   "split_only": "CRLF input change on paired identical backend/padding, legacy labels fixed.",
                   "combined_matched": "Unit+CRLF effect against the matched legacy replay; not trained policy accuracy.",
                   "quarantine": "Remove entire invalid-question group from common-subset summaries; original rows preserved. Not an improvement estimate from deleting hard questions."},
               "all_original_groups": {}, "common_retained_groups": {}, "strata": {}}
    for comparison, (old_name, new_name) in comparisons.items():
        old, new = arrays[old_name], arrays[new_name]
        summary["all_original_groups"][comparison] = summarize_difference(old, new, set(range(len(rows) // 8)))
        summary["common_retained_groups"][comparison] = summarize_difference(old, new, retained)
        summary["strata"][comparison] = {}
        baseline = arrays["legacy_replayed"]
        for kind in ("all_correct", "all_wrong", "mixed"):
            for validity in (True, False):
                ix = [i for i, r in enumerate(baseline) if r["group_type"] == kind and r["all_group_members_valid"] == validity]
                summary["strata"][comparison][kind + ("_all_valid" if validity else "_has_invalid")] = summarize_difference(old, new, retained, ix)
    atomic_json(out / "numerical_audit_summary.json", guard)
    atomic_json(out / "replay_QA.json", {"unit_patch_holds_PRM_scores_fixed": unit_isolation,
                "split_patch_holds_outcome_labels_fixed": split_isolation,
                "no_invalid_placeholder_misread_as_PRM_score": True,
                "all_original_partners_retained_for_advantages": True,
                "original_logged_reward_formula_reproduced": True,
                "all_actual_advantages_checked_against_installed_TRL": True,
                "actual_advantages_not_mutated_by_audit": True,
                "all_variants_have_original_row_count": all(len(a) == len(rows) for a in arrays.values()),
                "new_training": False})
    examples = []
    for review in reviews:
        i = review["source_row"]
        examples.append({"review_id": review["review_id"], "source_row": i, "question": review["question"],
                         "sampling_category": review["sampling_category"], "quarantined": i // 8 in excluded,
                         "variants": {name: records[i] for name, records in arrays.items()}})
    for i in (78, 2095, 4975):
        examples.append({"review_id": "prior_row_" + str(i), "source_row": i,
                         "variants": {name: records[i] for name, records in arrays.items()}})
    atomic_json(out / "case_comparisons.json", examples)
    with (out / "case_comparisons.csv").open("w", newline="") as stream:
        fields = ["review_id", "source_row", "variant", "outcome_reward", "min_score", "step_count", "reward", "reward_fp32", "reward_rank", "process_rank", "advantage", "group_type", "quarantined"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for e in examples:
            for variant, r in e["variants"].items():
                writer.writerow({**{k: r[k] for k in fields if k in r}, "review_id": e["review_id"], "variant": variant})
    with (out / "group_changes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["comparison", "group", "problem_id", "baseline_type", "new_type", "quarantined", "reward_changed", "rank_changed", "sign_changed", "max_advantage_delta"])
        writer.writeheader()
        for comparison, (a, b) in comparisons.items():
            for group in range(len(rows) // 8):
                x, y = arrays[a][group * 8:group * 8 + 8], arrays[b][group * 8:group * 8 + 8]
                delta = max(abs(j["advantage"] - i["advantage"]) for i, j in zip(x, y))
                rank_changed = any(i["reward_rank"] != j["reward_rank"] for i, j in zip(x, y))
                if delta <= 1e-6 and not rank_changed and group not in excluded:
                    continue
                writer.writerow({"comparison": comparison, "group": group, "problem_id": x[0]["problem_id"],
                                 "baseline_type": x[0]["group_type"], "new_type": y[0]["group_type"], "quarantined": group in excluded,
                                 "reward_changed": any(abs(i["reward"] - j["reward"]) > 1e-6 for i, j in zip(x, y)),
                                 "rank_changed": rank_changed, "sign_changed": sum(sign(i["advantage"]) != sign(j["advantage"]) for i, j in zip(x, y)),
                                 "max_advantage_delta": delta})
    summary.update(complete=True, elapsed_seconds=time.monotonic() - started,
                   inputs_sha256={name: sha256(out / name) for name in ("outcomes.jsonl", "paired_scores.jsonl", "unit_metadata.json", "quarantine.json")})
    atomic_json(out / "replay_summary.json", summary)
    print(json.dumps({"complete": True, "rows": len(rows), "guard": guard, "main_comparisons": summary["common_retained_groups"]}), flush=True)


if __name__ == "__main__":
    main()
