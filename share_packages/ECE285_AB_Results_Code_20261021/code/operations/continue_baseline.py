"""Continue the audited GRPO baseline, retaining fixed-token full checkpoints.

The original reward, GRPO implementation and accounting subclass are imported
unchanged. The saved GRPOConfig is reconstructed with only output paths changed.
This runner requires a full checkpoint and verifies the state actually restored.
"""
from collections import Counter
import dataclasses
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig
from common import atomic_json, file_hash, read_jsonl
from reward import outcome
from train_baseline import AuditedGRPO, BudgetCallback


def read(path):
    return json.loads(Path(path).read_text())


def equal_states(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_states(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal_states(a, b)
    else:
        assert actual == expected


class VerifiedContinuation(AuditedGRPO):
    def __init__(self, *args, source, expected_first_ids, **kwargs):
        self.source = Path(source)
        self.source_step = read(self.source / "trainer_state.json")["global_step"]
        self.expected_first_ids = expected_first_ids
        self.resume_audit = {"source_checkpoint": str(self.source), "source_global_step": self.source_step}
        self.first_generation = True
        super().__init__(*args, **kwargs)

    def audit(self, **values):
        self.resume_audit.update(values)
        atomic_json(self.run_dir / "resume_audit.json", self.resume_audit)

    def _load_from_checkpoint(self, checkpoint, model=None):
        super()._load_from_checkpoint(checkpoint, model)
        saved = load_file(str(Path(checkpoint) / "adapter_model.safetensors"), device="cpu")
        current = get_peft_model_state_dict(self.model if model is None else model)
        equal_states(current, saved)
        assert any(p.requires_grad for n, p in self.model.named_parameters() if "lora_" in n)
        self.probe_key = next(k for k in saved if "lora_B" in k)
        self.probe_before = saved[self.probe_key].clone()
        self.audit(adapter_restored_exactly=True, adapter_tensor_count=len(saved))

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        saved = torch.load(Path(checkpoint) / "optimizer.pt", map_location="cpu", weights_only=True)
        equal_states(self.optimizer.state_dict(), saved)
        steps = sorted({int(s["step"]) for s in saved["state"].values()})
        assert steps == [self.source_step]
        saved_scheduler = torch.load(Path(checkpoint) / "scheduler.pt", map_location="cpu", weights_only=True)
        equal_states(self.lr_scheduler.state_dict(), saved_scheduler)
        assert saved_scheduler["last_epoch"] == self.source_step
        self.audit(optimizer_restored_exactly=True, optimizer_state_count=len(saved["state"]),
                   optimizer_steps=steps, scheduler_restored_exactly=True,
                   learning_rates=[g["lr"] for g in self.optimizer.param_groups])

    def _load_rng_state(self, checkpoint):
        super()._load_rng_state(checkpoint)
        # These are our own trusted checkpoint files, not third-party pickles.
        saved = torch.load(Path(checkpoint) / "rng_state.pth", map_location="cpu", weights_only=False)
        assert random.getstate() == saved["python"]
        actual_numpy = np.random.get_state()
        assert actual_numpy[0] == saved["numpy"][0]
        assert np.array_equal(actual_numpy[1], saved["numpy"][1])
        assert actual_numpy[2:] == saved["numpy"][2:]
        equal_states(torch.random.get_rng_state(), saved["cpu"])
        equal_states(torch.cuda.get_rng_state(), saved["cuda"])
        self.audit(python_numpy_torch_cpu_cuda_rng_restored=True)

    def _move_model_to_vllm(self):
        super()._move_model_to_vllm()
        if self.first_generation:
            self.audit(first_vllm_weight_sync_global_step=self.state.global_step)

    def _generate_and_score_completions(self, inputs):
        first = self.first_generation
        if first:
            assert self.state.global_step == self.source_step
            for key in ["adapter_restored_exactly", "optimizer_restored_exactly",
                        "scheduler_restored_exactly", "python_numpy_torch_cpu_cuda_rng_restored"]:
                assert self.resume_audit.get(key), f"Missing actual restore check: {key}"
            actual = [row["problem_id"] for row in inputs]
            assert actual == self.expected_first_ids, "Resumed dataloader did not skip the original examples correctly"
            self.audit(first_rollout_global_step=self.state.global_step, data_progress_verified=True,
                       first_new_question_ids=list(dict.fromkeys(actual)))
        output = super()._generate_and_score_completions(inputs)
        if first:
            assert self.resume_audit["first_vllm_weight_sync_global_step"] == self.source_step
            self.first_generation = False
            self.audit(first_resumed_generation_completed=True)
        return output


class MilestoneCallback(BudgetCallback):
    def __init__(self, trainer, plan, initial_tokens):
        super().__init__(trainer, plan["wall_limit_hours"] * 3600)
        self.initial_tokens = initial_tokens
        self.targets = [t for t in plan["milestones"] if t > initial_tokens]
        self.pending = []

    def on_train_begin(self, args, state, control, **kwargs):
        assert state.global_step == self.trainer.source_step
        assert not control.should_training_stop
        assert not args.ignore_data_skip
        assert self.trainer.accounting["generated_completion_tokens"] == self.initial_tokens
        self.trainer.start_clock = time.monotonic()
        self.trainer.audit(train_begin_global_step=state.global_step,
                           initial_cumulative_completion_tokens=self.initial_tokens)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # Preserve the existing stop logic; correct throughput to count only
        # newly generated tokens in this continuation's elapsed time.
        control = super().on_step_end(args, state, control, **kwargs)
        trainer = self.trainer
        progress = read(trainer.run_dir / "progress.json")
        new_tokens = progress["generated_completion_tokens"] - self.initial_tokens
        progress.update(resumed_from_global_step=trainer.source_step,
                        source_completion_tokens=self.initial_tokens,
                        continuation_generated_completion_tokens=new_tokens,
                        effective_completion_tokens_per_second=new_tokens / max(progress["elapsed_seconds"], 1))
        atomic_json(trainer.run_dir / "progress.json", progress)
        for target in self.targets:
            if progress["generated_completion_tokens"] >= target and target not in self.pending:
                if not (trainer.run_dir / "milestones" / str(target) / "milestone.json").exists():
                    self.pending.append(target)
        if self.pending:
            control.should_save = True
        if state.global_step == trainer.source_step + 1:
            steps = sorted({int(s["step"]) for s in trainer.optimizer.state.values()})
            assert steps == [state.global_step]
            now = get_peft_model_state_dict(trainer.model)[trainer.probe_key].detach().cpu()
            delta = float((now - trainer.probe_before).abs().sum())
            assert delta > 0, "First resumed update did not change the checked adapter tensor"
            trainer.audit(first_resumed_optimizer_step=state.global_step,
                          optimizer_steps_after_first_update=steps, adapter_probe_change_l1=delta,
                          first_update_completion_tokens=progress["generated_completion_tokens"])
        return control

    def on_save(self, args, state, control, **kwargs):
        super().on_save(args, state, control, **kwargs)
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        for target in self.pending:
            destination = self.trainer.run_dir / "milestones" / str(target)
            staging = destination.with_name(f".{target}.tmp")
            assert not destination.exists() and not staging.exists()
            staging.mkdir(parents=True)
            shutil.copytree(checkpoint, staging / checkpoint.name)
            hashes = {p.name: file_hash(p) for p in checkpoint.iterdir() if p.is_file()}
            for name, digest in hashes.items():
                assert file_hash(staging / checkpoint.name / name) == digest
            tokens = self.trainer.accounting["generated_completion_tokens"]
            assert 0 <= tokens - target < args.generation_batch_size * args.max_completion_length
            receipt = {"target_completion_tokens": target, "actual_completion_tokens": tokens,
                       "overshoot": tokens - target, "global_step": state.global_step,
                       "checkpoint": str(destination / checkpoint.name), "checkpoint_sha256": hashes,
                       "evaluation_dir": str(destination / "evaluation"), "status": "saved_pending_evaluation",
                       "saved_unix": time.time()}
            atomic_json(staging / "milestone.json", receipt)
            staging.rename(destination)
            index = read(self.trainer.run_dir / "milestones.json")
            index["milestones"][str(target)] = receipt
            atomic_json(self.trainer.run_dir / "milestones.json", index)
            print(json.dumps({"event": "milestone_saved", **receipt}), flush=True)
        self.pending.clear()
        return control


def main():
    plan = read(ROOT / "operations/continuation_24m.json")
    source, run = ROOT / plan["source_checkpoint"], ROOT / plan["run_dir"]
    parent = ROOT / plan["origin_run"]
    lock = (run / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (run / "run_config.json").exists(), "Continuation already attempted; inspect before explicit recovery"
    origin = read(parent / "run_config.json")
    cfg_path = ROOT / "training/baseline_config.json"
    cfg, assets = read(cfg_path), read(ROOT / "receipts/assets.json")
    manifest_path = ROOT / "data/prepared/manifest.json"
    manifest = read(manifest_path)
    assert file_hash(cfg_path) == origin["config_sha256"] == manifest["config_sha256"]
    assert file_hash(manifest_path) == origin["data_manifest_sha256"]
    assert assets["model_revision"] == origin["model_revision"]
    for name, digest in origin["code_sha256"].items():
        # Only the launcher changed earlier to repair missing Python headers;
        # it is not used by this continuation. The algorithm/evaluator are pinned.
        if name != "launch_pipeline.py":
            assert file_hash(ROOT / "training" / name) == digest, f"Changed original source: {name}"
    versions = {p: importlib.metadata.version(p) for p in origin["packages"]}
    assert versions == origin["packages"]
    data_path = ROOT / "data/prepared/train.jsonl"
    assert file_hash(data_path) == manifest["files"][data_path.name]["sha256"]
    accounting = read(source / "completion_accounting.json")
    state = read(source / "trainer_state.json")
    assert state["global_step"] == plan["source_global_step"]
    assert accounting["generated_completion_tokens"] == plan["source_completion_tokens"]
    saved_args = torch.load(source / "training_args.bin", map_location="cpu", weights_only=False)
    kwargs = {f.name: getattr(saved_args, f.name) for f in dataclasses.fields(saved_args) if f.init}
    # GRPOConfig derives generation_batch_size from steps_per_generation.
    kwargs.update(output_dir=str(run), logging_dir=str(run / "tensorboard"), generation_batch_size=None)
    args = GRPOConfig(**kwargs)
    argument_changes = {k: [saved_args.to_dict().get(k), v] for k, v in args.to_dict().items()
                        if saved_args.to_dict().get(k) != v}
    assert set(argument_changes) == {"output_dir", "logging_dir"}
    assert args.num_iterations == 1 and args.steps_per_generation == args.gradient_accumulation_steps
    assert not args.restore_callback_states_from_checkpoint and not args.ignore_data_skip
    rows = read_jsonl(data_path)
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
    prior = read_jsonl(parent / "rollouts.jsonl")
    groups = accounting["generated_groups"]
    expected_prior = [rows[i]["problem_id"] for i in order[:groups] for _ in range(cfg["group_size"])]
    assert [r["problem_id"] for r in prior] == expected_prior
    next_groups = args.generation_batch_size // args.num_generations
    expected_first_ids = [rows[i]["problem_id"] for i in order[groups:groups + next_groups]
                          for _ in range(cfg["group_size"])]
    spec = {**origin, "continuation_started_unix": time.time(), "pid": os.getpid(),
            "origin_run": str(parent), "source_checkpoint": str(source), "continuation_plan": plan,
            "continuation_plan_sha256": file_hash(ROOT / "operations/continuation_24m.json"),
            "argument_changes": argument_changes,
            "continuation_code_sha256": file_hash(Path(__file__)),
            "original_training_configuration_retained": True,
            "effective_generated_token_budget": plan["target_completion_tokens"],
            "rng_limitation": plan["rng_limitation"]}
    atomic_json(run / "run_config.json", spec)
    status = {"status": "initializing", "pid": os.getpid(), "started_unix": time.time(),
              "source_checkpoint": str(source), "source_global_step": state["global_step"],
              "source_completion_tokens": plan["source_completion_tokens"]}
    atomic_json(run / "status.json", status)
    try:
        set_seed(cfg["seed"])
        tokenizer = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(assets["model_path"], local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation=cfg["attention_implementation"])
        model.config.use_cache = False
        holder = {}

        def outcome_accuracy(prompts, completions, completion_ids, answer, problem_id, **unused):
            values, statuses, truncations = [], Counter(), 0
            trainer = holder["trainer"]
            with (run / "rollouts.jsonl").open("a") as stream:
                for text, ids, gold, ident in zip(completions, completion_ids, answer, problem_id):
                    terminated = tokenizer.eos_token_id in ids
                    result = outcome(text, gold, terminated)
                    values.append(result["reward"])
                    statuses[result["status"]] += 1
                    truncations += not terminated
                    stream.write(json.dumps({"optimizer_step_before_update": trainer.state.global_step,
                        "problem_id": ident, "gold": gold, "completion": text, "completion_tokens": len(ids),
                        "terminated": terminated, **result}, ensure_ascii=False) + "\n")
            trainer.last_reward_values = values
            trainer.last_batch = {"status_counts": dict(statuses), "truncated_completions": int(truncations),
                                  "token_count": sum(map(len, completion_ids))}
            return values

        lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                         target_modules=cfg["lora_target_modules"], bias="none", task_type="CAUSAL_LM")
        trainer = VerifiedContinuation(model=model, processing_class=tokenizer, reward_funcs=outcome_accuracy,
            args=args, train_dataset=Dataset.from_list(rows), peft_config=lora, run_dir=run,
            budget=plan["target_completion_tokens"], resume_accounting=accounting,
            source=source, expected_first_ids=expected_first_ids)
        holder["trainer"] = trainer
        callback = MilestoneCallback(trainer, plan, plan["source_completion_tokens"])
        trainer.add_callback(callback)
        atomic_json(run / "status.json", {**status, "status": "restoring_and_training"})
        result = trainer.train(resume_from_checkpoint=str(source))
        trainer.save_model(str(run / "final_adapter"))
        tokenizer.save_pretrained(run / "final_adapter")
        trainer.save_state()
        completed = trainer.accounting["generated_completion_tokens"] >= trainer.budget
        final = {**status, "status": "completed" if completed else "incomplete", "finished_unix": time.time(),
                 "global_step": trainer.state.global_step, "accounting": trainer.accounting,
                 "budget": trainer.budget, "budget_overshoot": max(0, trainer.accounting["generated_completion_tokens"] - trainer.budget),
                 "stop_reason": callback.stop_reason, "continuation_metrics": result.metrics,
                 "rng_limitation": plan["rng_limitation"]}
        atomic_json(run / "status.json", final)
        assert completed, "Training stopped early; full checkpoint retained, do not restart from base"
        index = read(run / "milestones.json")
        for target in [12000000, 24000000]:
            assert index["milestones"][str(target)]["status"] == "saved_pending_evaluation"
        last = Path(index["milestones"]["24000000"]["checkpoint"])
        assert file_hash(run / "final_adapter/adapter_model.safetensors") == file_hash(last / "adapter_model.safetensors")
        print(json.dumps(final, indent=2), flush=True)
    except Exception as exc:
        atomic_json(run / "failure.json", {"error": str(exc), "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
