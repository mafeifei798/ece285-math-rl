#!/usr/bin/env python3
"""Read-only, standard-library audit of the fixed corrected A/B 6M experiment.

No project modules are imported, no model is loaded, and no training can start.
Optional --output uses exclusive creation; otherwise the only output is stdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time

EXPERIMENT = "ab_corrected_6m_seed17_20261022"
VALID = {"exact", "symbolic", "unit_exact", "unit_symbolic"}
LIMITATIONS = [
    "This audit is CPU postprocessing only; it does not replay gradients or score responses with a PRM.",
    "Token totals are independently summed from saved counts, not retokenized; runtime completion-mask checks provide the separate generation check.",
    "Stored outcome labels are checked for consistency, not mathematically rejudged; a separate frozen-verifier evaluation replay is needed.",
    "Independent advantage arithmetic uses FP64 over FP32 rewards with an explicit rounding allowance; exact tensor preservation relies on the live loss guards.",
    "Source and checkpoint hashes establish linkage to saved receipts, not an external cryptographic trust anchor.",
    "PRM scoring elapsed time is receipt-checked, not independently remeasured; it includes encoding, padding, transfer, inference and score extraction.",
    "A single arm audit does not establish A/B equality of shared configuration or question order; compare both audit outputs and run configurations.",
]


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    def reject_constant(value):
        raise AuditError(f"Nonfinite JSON constant {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Path escapes project root: {relative}")
    return path


def integer(value, label, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"Invalid integer {label}: {value!r}")
    return value


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            f"Nonfinite/non-numeric {label}: {value!r}")
    return float(value)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def safetensors_inventory(path):
    """Inspect tensor metadata and payload SHA without torch/safetensors imports."""
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, f"Short safetensors header: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        require(0 < header_size <= 16 * 1024 * 1024, f"Invalid safetensors header size: {path}")
        header = json.loads(stream.read(header_size))
        data_start = 8 + header_size
        size = path.stat().st_size - data_start
        items, ranges = {}, []
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            integer(begin, "tensor offset"); integer(end, "tensor end")
            require(begin <= end <= size, f"Tensor outside payload: {name}")
            require(isinstance(info["shape"], list), f"Invalid tensor shape: {name}")
            for dimension in info["shape"]:
                integer(dimension, "tensor dimension")
            stream.seek(data_start + begin)
            digest = hashlib.sha256()
            remaining = end - begin
            while remaining:
                chunk = stream.read(min(4 * 1024 * 1024, remaining))
                require(bool(chunk), f"Truncated tensor: {name}")
                digest.update(chunk); remaining -= len(chunk)
            items[name] = {"dtype": info["dtype"], "shape": info["shape"],
                           "bytes": end - begin, "sha256": digest.hexdigest()}
            ranges.append((begin, end))
        require(items, "No adapter tensors")
        nonempty = [(a, b) for a, b in sorted(ranges) if a != b]
        require(not nonempty or (nonempty[0][0] == 0 and nonempty[-1][1] == size),
                "Adapter payload has unclaimed boundary bytes")
        require(all(a[1] == b[0] for a, b in zip(nonempty, nonempty[1:])),
                "Adapter tensors overlap or have gaps")
    return items


def audit_rows(rows, arm):
    """Independent schema/formula/accounting checks, reusable by fixture tests."""
    require(rows and len(rows) % 32 == 0, "Rollout count must be nonzero and divisible by 32")
    totals = dict(generated_completion_tokens=0, generated_completions=len(rows),
                  generated_groups=len(rows) // 8, mixed_reward_groups=0,
                  correct_completions=0, truncated_completions=0,
                  prm_input_tokens=0, prm_padded_tokens=0, prm_scored_completions=0)
    statuses, outcome_groups = Counter(), Counter()
    group_order = []
    batch_totals = []
    advantage_max_error = 0.0
    advantage_max_allowance = 0.0
    missing_advantages = 0
    for batch_index, start in enumerate(range(0, len(rows), 32)):
        batch = rows[start:start + 32]
        batch_cost = {k: 0 for k in totals}
        batch_cost.update(generated_completions=32, generated_groups=4)
        selected = []
        for index, row in enumerate(batch, start):
            label = f"row {index}"
            require(row["optimizer_step_before_update"] == batch_index,
                    f"{label}: optimizer step/order differs from 32-row updates")
            count = integer(row["completion_tokens"], label + " completion_tokens", 1)
            require(count <= 2048, f"{label}: token count exceeds cap")
            require(isinstance(row["completion"], str), f"{label}: missing raw completion")
            require(isinstance(row["terminated"], bool), f"{label}: terminated must be bool")
            status = row["status"]
            require(isinstance(status, str) and not status.startswith("verifier_error"),
                    f"{label}: verifier error status")
            require((status == "truncated") == (not row["terminated"]),
                    f"{label}: termination/status disagree")
            outcome = finite(row["outcome_reward"], label + " outcome")
            reward = finite(row["reward"], label + " reward")
            require(outcome in (0, 1), f"{label}: outcome must be binary")
            valid = status in VALID
            require(valid or outcome == 0, f"{label}: invalid output has positive outcome")
            scores = row["step_scores"]
            require(isinstance(scores, list), f"{label}: step_scores must be a list")
            for score in scores:
                require(0 <= finite(score, label + " step score") <= 1,
                        f"{label}: process score outside [0,1]")
            step_count = integer(row["step_count"], label + " step_count")
            require(step_count == len(scores), f"{label}: step_count/score length mismatch")
            input_tokens = integer(row["prm_input_tokens"], label + " prm_input_tokens")
            minimum = finite(row["min_score"], label + " min_score")
            require(abs(minimum - (min(scores) if scores else 0)) <= 1e-12,
                    f"{label}: inconsistent minimum process score")
            if arm == "A":
                require(row["valid_for_process_reward"] is False and not scores and input_tokens == 0,
                        f"{label}: A contains process scoring")
                expected = outcome
            else:
                require(row["valid_for_process_reward"] is valid,
                        f"{label}: PRM eligibility differs from valid statuses")
                if valid:
                    require(bool(scores) and input_tokens > 0, f"{label}: eligible PRM output not scored")
                    text = row["completion"].replace("\r\n", "\n")
                    steps = [s.strip() for s in text.split("\n\n") if s.strip()]
                    require(len(steps) == step_count, f"{label}: CRLF canonical segmentation mismatch")
                    require(row["raw_sha256"] == hashlib.sha256(row["completion"].encode()).hexdigest(),
                            f"{label}: raw completion hash mismatch")
                    require(row["crlf_replacements"] == row["completion"].count("\r\n"),
                            f"{label}: CRLF replacement count mismatch")
                    require(row["repair_version"] == "reward-repair-v3-20261021",
                            f"{label}: different repair version")
                    selected.append(input_tokens)
                else:
                    require(not scores and input_tokens == 0, f"{label}: invalid output was scored")
                expected = outcome + .2 * minimum if valid else 0
            require(abs(reward - expected) <= 1e-12, f"{label}: reward formula mismatch")
            batch_cost["generated_completion_tokens"] += count
            batch_cost["correct_completions"] += int(outcome)
            batch_cost["truncated_completions"] += int(not row["terminated"])
            batch_cost["prm_input_tokens"] += input_tokens
            statuses[status] += 1
        batch_cost["prm_scored_completions"] = len(selected)
        batch_cost["prm_padded_tokens"] = sum(max(selected[i:i+4]) * len(selected[i:i+4])
                                                  for i in range(0, len(selected), 4))
        for offset in range(0, 32, 8):
            group = batch[offset:offset + 8]
            require(len({r["problem_id"] for r in group}) == 1, "Group crosses problem IDs")
            require(len({r["gold"] for r in group}) == 1, "Group crosses gold answers")
            group_order.append(group[0]["problem_id"])
            rewards = [f32(r["reward"]) for r in group]
            mean = statistics.fmean(rewards)
            std = statistics.stdev(rewards)
            expected_advantages = [(r - mean) / (std + 1e-4) for r in rewards]
            # Bound FP32 reduction/subtraction rounding, most relevant in near-tied groups.
            allowance = 1e-5 + 2 * 2**-23 * max(abs(r) for r in rewards) / (std + 1e-4)
            advantage_max_allowance = max(advantage_max_allowance, allowance)
            for row, expected in zip(group, expected_advantages):
                if "advantage" not in row:
                    missing_advantages += 1
                    continue
                actual = finite(row["advantage"], "stored advantage")
                error = abs(actual - expected)
                advantage_max_error = max(advantage_max_error, error)
                require(error <= allowance, "Stored advantage inconsistent with original GRPO group standardization")
                require(row.get("advantage_rule") == "mixed_reward_original_group_std",
                        "Unexpected stored advantage rule")
            batch_cost["mixed_reward_groups"] += int(max(rewards) > min(rewards))
            correct = sum(r["outcome_reward"] for r in group)
            outcome_groups["all_correct" if correct == 8 else "all_wrong" if correct == 0 else "mixed"] += 1
        for key in totals:
            if key not in {"generated_completions", "generated_groups"}:
                totals[key] += batch_cost[key]
        batch_totals.append(batch_cost)
    require(missing_advantages in (0, len(rows)), "Only some rows contain advantages")
    totals.update(reward_status_counts=dict(statuses), outcome_group_counts=dict(outcome_groups))
    return dict(totals=totals, last_batch=batch_totals[-1], global_step=len(batch_totals),
                previous_update_tokens=sum(b["generated_completion_tokens"] for b in batch_totals[:-1]),
                group_order_sha256=hashlib.sha256(json.dumps(group_order, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                advantage_check=dict(present=not missing_advantages, rows_checked=len(rows)-missing_advantages,
                                     max_abs_error=advantage_max_error, max_rounding_allowance=advantage_max_allowance))


def audit(root, arm):
    root = Path(root).resolve()
    experiment = inside(root, "experiments/" + EXPERIMENT)
    plan, release = read(experiment / "plan.json"), read(experiment / "release.json")
    require(plan["seed"] == 17 and plan["completion_token_budget_per_arm"] == 6000000,
            "Not the authorized seed17/6M plan")
    require(plan["group_size"] == 8 and plan["process_coefficient"] == .2, "Plan changed reward/group")
    for collection in ("source_hashes", "runtime_inputs"):
        for relative, digest in release[collection].items():
            require(sha(inside(root, relative)) == digest, f"Pinned {collection} hash changed: {relative}")
    run = inside(root, plan["runs"][arm])
    status, spec, progress, live = [read(run / name) for name in
                                  ("status.json", "run_config.json", "progress.json", "live_audits.json")]
    require(status["status"] == "completed" and status["arm"] == arm and status["pilot"] is False,
            "Formal run is not completed")
    require(status["budget"] == 6000000 and status["stop_reason"] == "token_budget_reached",
            "Different budget or stop reason")
    require(spec["arm"] == arm and spec["seed"] == 17 and spec["pilot"] is False,
            "Run configuration arm/seed/pilot mismatch")
    require(spec["source_release_sha256"] == sha(experiment / "release.json"), "Release receipt mismatch")
    require(spec["unit_metadata_sha256"] == release["unit_metadata_sha256"], "Unit metadata mismatch")
    require(spec["verifier_version"] == "reward-repair-v3-20261021", "Unexpected verifier version")
    require(spec["source_checkpoint"] is None and spec["base_actor_initialization"] is True,
            "Formal run did not start at common base")
    require(spec["target_completion_tokens"] == 6000000 and spec["raw_vllm_engine_seed"] == 0,
            "Run token target or generation seed changed")
    require(spec["advantage"] == "original_TRL_mixed_reward_group_standardization",
            "Unexpected advantage rule in configuration")
    require(spec["reward"] == ("outcome only" if arm == "A" else "outcome+0.2*min(native step probabilities)"),
            "Unexpected configured reward")
    expected_arguments = dict(seed=17, data_seed=17, num_generations=8,
                              gradient_accumulation_steps=32, per_device_train_batch_size=1,
                              max_completion_length=2048, max_prompt_length=2048,
                              loss_type="grpo", scale_rewards="group", num_iterations=1,
                              beta=.04, epsilon=.2, learning_rate=5e-6)
    for key, value in expected_arguments.items():
        require(spec["shared_training_arguments"][key] == value, "Training argument differs: " + key)
    require(spec["config_sha256"] == sha(inside(root, "training/baseline_config.json")),
            "Training configuration hash mismatch")
    require(spec["data_manifest_sha256"] == sha(inside(root, "data/prepared/manifest.json")),
            "Training data manifest hash mismatch")
    require(spec["model_revision"] == read(inside(root, "receipts/assets.json"))["model_revision"],
            "Actor model revision mismatch")
    require(status["actor_adapter_changed"] is True and finite(status["lora_B_abs_sum"], "LoRA norm") > 0,
            "No actor parameter update recorded")
    require(status["prm_frozen_verified"] is (arm == "B"), "Frozen PRM receipt mismatch")
    data = []
    with (run / "rollouts.jsonl").open() as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank rollout row at line {number}")
            data.append(json.loads(line))
    computed = audit_rows(data, arm)
    require(computed["advantage_check"]["present"], "Expected formal stored advantages missing")
    step, tokens = computed["global_step"], computed["totals"]["generated_completion_tokens"]
    require(status["global_step"] == step == progress["global_step"], "Final step mismatch")
    require(6000000 <= tokens < 6000000 + 32 * 2048, "Final token budget/overshoot mismatch")
    require(computed["previous_update_tokens"] < 6000000, "Did not stop at first complete budget-reaching update")
    checkpoint = run / f"checkpoint-{step}"
    recorded = Path(status["checkpoint"])
    require(recorded == checkpoint or recorded.parts[-3:] == checkpoint.parts[-3:],
            "Recorded checkpoint points to a different run/step")
    hashes = status["checkpoint_sha256"]
    required = {"adapter_model.safetensors", "adapter_config.json", "trainer_state.json", "completion_accounting.json",
                "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"}
    require(required <= hashes.keys(), "Final checkpoint hash receipt incomplete")
    require(set(hashes) == {p.name for p in checkpoint.iterdir() if p.is_file()}, "Checkpoint file inventory differs")
    for name, digest in hashes.items():
        require(Path(name).name == name, "Unsafe checkpoint hash filename")
        require(sha(checkpoint / name) == digest, f"Final checkpoint hash mismatch: {name}")
    account = read(checkpoint / "completion_accounting.json")
    require(read(checkpoint / "trainer_state.json")["global_step"] == step, "Checkpoint trainer step mismatch")
    for source_name, source in [("status", status["accounting"]), ("progress", progress), ("checkpoint", account)]:
        for key, value in computed["totals"].items():
            require(source[key] == value, f"{source_name} accounting mismatch: {key}")
        for key, value in computed["last_batch"].items():
            require(source["last_batch"][key] == value, f"{source_name} last_batch mismatch: {key}")
    require(account == status["accounting"], "Checkpoint and final status accounting differ")
    prm_seconds = finite(account["prm_forward_seconds"], "PRM scoring seconds")
    require(prm_seconds >= 0 and (arm != "A" or prm_seconds == 0), "Invalid PRM scoring timing")
    require(progress["prm_forward_seconds"] == prm_seconds, "PRM scoring time receipt mismatch")
    require(live["passed"] is True and live["global_step"] == step and live["generation_batches"] == step,
            "Final live audit missing/incomplete")
    require(live["loss_rows"] == len(data) == live["loss_calls"], "Live loss coverage differs")
    require(live["original_TRL_advantages_retained"] is True and live["loss_input_values_preserved"] is True,
            "Live original-advantage guards failed")
    require(live["PRM_frozen"] is (True if arm == "B" else None), "Live frozen PRM receipt mismatch")
    adapter_a = checkpoint / "adapter_model.safetensors"
    adapter_b = run / "final_adapter/adapter_model.safetensors"
    tensor_a, tensor_b = safetensors_inventory(adapter_a), safetensors_inventory(adapter_b)
    require(tensor_a == tensor_b, "Checkpoint and final_adapter tensor content differs")
    require(read(checkpoint / "adapter_config.json") == read(run / "final_adapter/adapter_config.json"),
            "Final adapter configuration differs")
    diagnostics = []
    for candidate in sorted(run.glob("group_diagnostics*")):
        if candidate.is_file():
            diagnostics.append(dict(name=candidate.name, bytes=candidate.stat().st_size, sha256=sha(candidate),
                                    scope="optional artifact inventoried; group diagnostics independently recomputed from rollouts"))
    return dict(passed=True, arm=arm, run=str(run), audited_unix=time.time(),
                source_release_sha256=sha(experiment / "release.json"),
                source_files_checked=len(release["source_hashes"]), runtime_inputs_checked=len(release["runtime_inputs"]),
                recomputed=computed, actual_completion_tokens=tokens, overshoot_tokens=tokens-6000000,
                prm_scoring_seconds=prm_seconds, checkpoint=str(checkpoint),
                checkpoint_files_checked=len(hashes), adapter_tensors_checked=len(tensor_a),
                checkpoint_adapter_sha256=sha(adapter_a), final_adapter_sha256=sha(adapter_b),
                final_adapter_logical_tensors_match=True, optional_group_diagnostics=diagnostics,
                configuration=dict(model_revision=spec["model_revision"],
                                   data_manifest_sha256=spec["data_manifest_sha256"],
                                   initial_adapter_sha256=spec["actor_adapter_initial_sha256"],
                                   shared_arguments_sha256=hashlib.sha256(json.dumps(spec["shared_training_arguments"],
                                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()),
                training_wall_seconds=progress.get("elapsed_seconds"),
                pytorch_peak_allocated_GiB=progress.get("torch_peak_allocated_GiB"),
                limitations=LIMITATIONS,
                artifact_hashes={name:sha(run/name) for name in
                                 ("status.json", "run_config.json", "progress.json", "live_audits.json", "rollouts.jsonl")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("A", "B"))
    parser.add_argument("--output", type=Path, help="Exclusive new JSON file; never overwrites")
    args = parser.parse_args()
    try:
        result = audit(args.root, args.arm)
        code = 0
    except Exception as exc:
        result = dict(passed=False, arm=args.arm, error_type=type(exc).__name__, error=str(exc),
                      audited_unix=time.time(), limitations=LIMITATIONS)
        code = 1
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    print(payload, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
