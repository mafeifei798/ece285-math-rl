"""Synthetic, independently specified positive and corruption controls; CPU only."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit_completed_training", HERE / "audit_completed_training.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(arm, updates=1):
    rows = []
    for update in range(updates):
        for group in range(4):
            # Four independent groups: all correct, all wrong, mixed, half truncated.
            outcomes = ([1] * 8, [0] * 8, [1] * 4 + [0] * 4, [0] * 8)[group]
            for i, outcome in enumerate(outcomes):
                truncated = group == 3 and i < 4
                text = "First line.\r\n\r\nSecond line."
                row = dict(optimizer_step_before_update=update, problem_id=f"q{update}-{group}",
                           gold="2", completion=text, completion_tokens=2000,
                           terminated=not truncated, status="truncated" if truncated else "exact",
                           outcome_reward=outcome, step_scores=[], min_score=0, step_count=0,
                           prm_input_tokens=0, valid_for_process_reward=False,
                           advantage_rule="mixed_reward_original_group_std")
                if arm == "B" and not truncated:
                    minimum = (.25 if i < 4 else .75) if group == 1 else .5
                    row.update(step_scores=[minimum, 1.0], min_score=minimum, step_count=2,
                               prm_input_tokens=20 + i, valid_for_process_reward=True,
                               raw_sha256=hashlib.sha256(text.encode()).hexdigest(), crlf_replacements=2,
                               repair_version="reward-repair-v3-20261021")
                row["reward"] = outcome + .2 * row["min_score"]
                # Analytic sample-SD for four copies each of two levels: SD = gap*sqrt(2/7).
                if arm == "A":
                    gap = 1.0 if group == 2 else 0.0
                    sign = 1 if i < 4 else -1
                else:
                    gap = {0:0.0, 1:.1, 2:1.0, 3:.1}[group]
                    sign = (1 if i < 4 else -1) if group == 2 else (-1 if i < 4 else 1)
                row["advantage"] = sign * gap / 2 / (gap * math.sqrt(2 / 7) + 1e-4)
                rows.append(row)
    return rows


def tensor_file(path, value=1.25, reverse_metadata=False):
    payload = struct.pack("<f", value)
    metadata = {"lora_B.weight":{"dtype":"F32", "shape":[1], "data_offsets":[0,4]}}
    if reverse_metadata:
        metadata = {"__metadata__":{"format":"pt"}, **metadata}
    header = json.dumps(metadata).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def fixture(root, arm):
    experiment = root / "experiments" / audit_module.EXPERIMENT
    run_relative = f"runs/{arm}_synthetic_seed17_6M"
    run = root / run_relative
    run.mkdir(parents=True)
    rows = rows_for(arm, updates=94)  # 94*32*2000=6,016,000; prior update=5,952,000.
    with (run / "rollouts.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    # Independently specified totals, not obtained from audit_rows.
    all_counts = dict(generated_completion_tokens=6016000, generated_completions=3008,
                      generated_groups=376, mixed_reward_groups=(94 if arm == "A" else 282),
                      correct_completions=1128, truncated_completions=376,
                      prm_input_tokens=(626*94 if arm == "B" else 0),
                      prm_padded_tokens=(668*94 if arm == "B" else 0),
                      prm_scored_completions=(28*94 if arm == "B" else 0),
                      prm_forward_seconds=(12.0 if arm == "B" else 0.0),
                      reward_status_counts={"exact":2632,"truncated":376},
                      outcome_group_counts={"all_correct":94,"all_wrong":188,"mixed":94})
    # Three full valid groups (8*23.5 each), plus 4 valid rows of length24..27.
    if arm == "B":
        all_counts["prm_input_tokens"] = (3*188 + 102)*94
        all_counts["prm_padded_tokens"] = (3*(4*23+4*27)+4*27)*94
    last = {key:value//94 for key,value in all_counts.items()
            if key not in {"prm_forward_seconds", "reward_status_counts", "outcome_group_counts"}}
    last["prm_forward_seconds"] = 1.0 if arm == "B" else 0.0
    all_counts["last_batch"] = last
    checkpoint = run / "checkpoint-94"
    checkpoint.mkdir()
    write(checkpoint / "completion_accounting.json", all_counts)
    write(checkpoint / "trainer_state.json", {"global_step":94})
    write(checkpoint / "adapter_config.json", {"r":16,"task_type":"CAUSAL_LM"})
    for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]:
        (checkpoint / name).write_bytes(b"synthetic-not-loaded")
    tensor_file(checkpoint / "adapter_model.safetensors")
    final = run / "final_adapter"
    final.mkdir()
    tensor_file(final / "adapter_model.safetensors", reverse_metadata=True)
    write(final / "adapter_config.json", {"r":16,"task_type":"CAUSAL_LM"})
    plan = dict(seed=17,completion_token_budget_per_arm=6000000,group_size=8,process_coefficient=.2,
                runs={arm:run_relative})
    write(experiment / "plan.json", plan)
    write(experiment / "release.json", {"source_hashes":{"experiments/" + audit_module.EXPERIMENT + "/plan.json":digest(experiment/"plan.json")},
                                        "runtime_inputs":{},"unit_metadata_sha256":"synthetic-unit-hash"})
    write(root/"training/baseline_config.json", {"seed":17})
    write(root/"data/prepared/manifest.json", {"synthetic":True})
    write(root/"receipts/assets.json", {"model_revision":"synthetic-model-revision"})
    arguments = dict(seed=17, data_seed=17, num_generations=8,
                     gradient_accumulation_steps=32, per_device_train_batch_size=1,
                     max_completion_length=2048, max_prompt_length=2048,
                     loss_type="grpo", scale_rewards="group", num_iterations=1,
                     beta=.04, epsilon=.2, learning_rate=5e-6)
    write(run / "run_config.json", dict(arm=arm,seed=17,pilot=False,
          source_release_sha256=digest(experiment/"release.json"),unit_metadata_sha256="synthetic-unit-hash",
          verifier_version="reward-repair-v3-20261021",source_checkpoint=None,base_actor_initialization=True,
          target_completion_tokens=6000000,raw_vllm_engine_seed=0,
          advantage="original_TRL_mixed_reward_group_standardization",
          reward="outcome only" if arm=="A" else "outcome+0.2*min(native step probabilities)",
          shared_training_arguments=arguments,config_sha256=digest(root/"training/baseline_config.json"),
          data_manifest_sha256=digest(root/"data/prepared/manifest.json"),
          model_revision="synthetic-model-revision",actor_adapter_initial_sha256="synthetic-initial-adapter"))
    write(run / "status.json", dict(status="completed",arm=arm,pilot=False,budget=6000000,
          stop_reason="token_budget_reached",global_step=94,actor_adapter_changed=True,lora_B_abs_sum=1.25,
          prm_frozen_verified=(arm=="B"),accounting=all_counts,checkpoint=str(checkpoint),
          checkpoint_sha256={p.name:digest(p) for p in checkpoint.iterdir()}))
    write(run / "progress.json", dict(all_counts, global_step=94, elapsed_seconds=100,
                                       torch_peak_allocated_GiB=1.0))
    write(run / "live_audits.json", dict(passed=True,global_step=94,generation_batches=94,
          loss_rows=3008,loss_calls=3008,original_TRL_advantages_retained=True,
          loss_input_values_preserved=True,PRM_frozen=True if arm=="B" else None))
    return run


class TestAudit(unittest.TestCase):
    def test_analytic_group_advantages(self):
        for arm in ["A", "B"]:
            result = audit_module.audit_rows(rows_for(arm), arm)
            self.assertEqual(result["totals"]["generated_completion_tokens"], 64000)
            self.assertEqual(result["totals"]["outcome_group_counts"],
                             {"all_correct":1,"all_wrong":2,"mixed":1})

    def test_corrupt_reward_rejected(self):
        rows = rows_for("B"); rows[0]["reward"] += .01
        with self.assertRaisesRegex(audit_module.AuditError, "reward formula mismatch"):
            audit_module.audit_rows(rows, "B")

    def test_corrupt_advantage_rejected(self):
        rows = rows_for("B"); rows[0]["advantage"] += 1
        with self.assertRaisesRegex(audit_module.AuditError, "advantage inconsistent"):
            audit_module.audit_rows(rows, "B")

    def test_corrupt_group_or_step_rejected(self):
        rows = rows_for("A"); rows[0]["problem_id"] = "another-question"
        with self.assertRaisesRegex(audit_module.AuditError, "Group crosses"):
            audit_module.audit_rows(rows, "A")
        rows = rows_for("A"); rows[0]["optimizer_step_before_update"] = 1
        with self.assertRaisesRegex(audit_module.AuditError, "optimizer step"):
            audit_module.audit_rows(rows, "A")

    def test_completed_arms_and_tensor_equivalence(self):
        for arm in ["A", "B"]:
            with tempfile.TemporaryDirectory(dir=HERE, prefix="fixture_") as directory:
                fixture(Path(directory), arm)
                report = audit_module.audit(Path(directory), arm)
                self.assertTrue(report["passed"])
                self.assertEqual(report["overshoot_tokens"],16000)
                self.assertNotEqual(report["checkpoint_adapter_sha256"], report["final_adapter_sha256"])
                self.assertTrue(report["final_adapter_logical_tensors_match"])

    def test_corrupt_token_total_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix="fixture_") as directory:
            run = fixture(Path(directory), "A")
            rows = [json.loads(line) for line in (run/"rollouts.jsonl").read_text().splitlines()]
            rows[0]["completion_tokens"] -= 1
            (run/"rollouts.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(audit_module.AuditError, "accounting mismatch: generated_completion_tokens"):
                audit_module.audit(Path(directory), "A")

    def test_corrupt_final_adapter_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix="fixture_") as directory:
            run = fixture(Path(directory), "B")
            tensor_file(run/"final_adapter/adapter_model.safetensors", value=2.0)
            with self.assertRaisesRegex(audit_module.AuditError, "tensor content differs"):
                audit_module.audit(Path(directory), "B")

    def test_cli_exclusive_output(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix="fixture_") as directory:
            fixture(Path(directory), "A")
            output = Path(directory)/"audit.json"
            command = [sys.executable,str(HERE/"audit_completed_training.py"),"--root",directory,
                       "--arm","A","--output",str(output)]
            first = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            before = output.read_bytes()
            second = subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(before,output.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
