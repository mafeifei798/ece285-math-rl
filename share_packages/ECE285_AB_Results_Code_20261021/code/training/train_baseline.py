"""Pinned TRL GRPO baseline with completion-token accounting and safe checkpoints.

The TRL objective is unchanged. Subclassing only records rollouts, audits group
advantages, and counts actually generated completion tokens (including failures).
"""
import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer
from common import ROOT, atomic_json, file_hash, read_jsonl
from reward import outcome


class AuditedGRPO(GRPOTrainer):
    def __init__(self, *args, run_dir, budget, resume_accounting=None, **kwargs):
        self.run_dir = Path(run_dir)
        self.budget = budget
        self.accounting = resume_accounting or {
            "generated_completion_tokens": 0, "generated_completions": 0,
            "generated_groups": 0, "mixed_reward_groups": 0,
            "correct_completions": 0, "truncated_completions": 0, "reward_status_counts": {},
        }
        self.last_reward_values = None
        self.last_batch = {}
        self.start_clock = time.monotonic()
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        if self.accelerator.num_processes != 1:
            raise RuntimeError("This initial runner is single-GPU; do not silently change token accounting")
        output = super()._generate_and_score_completions(inputs)
        if not self.model.training:
            return output
        counts = output["completion_mask"].sum(dim=1)
        tokens = int(counts.sum().item())
        rewards = torch.tensor(self.last_reward_values, dtype=torch.float32, device=output["advantages"].device)
        grouped = rewards.reshape(-1, self.num_generations)
        expected = ((grouped - grouped.mean(dim=1, keepdim=True)) /
                    (grouped.std(dim=1, keepdim=True) + 1e-4)).reshape(-1)
        torch.testing.assert_close(output["advantages"], expected, atol=2e-5, rtol=2e-5)
        if not torch.isfinite(output["advantages"]).all():
            raise RuntimeError("Nonfinite group advantages")
        batch = {
            "generated_completion_tokens": tokens,
            "generated_completions": int(len(counts)), "generated_groups": int(len(grouped)),
            "mixed_reward_groups": int(((grouped.max(dim=1).values - grouped.min(dim=1).values) > 0).sum()),
            "correct_completions": int(rewards.sum()),
            "truncated_completions": self.last_batch["truncated_completions"],
        }
        assert tokens == self.last_batch["token_count"], "Completion accounting differs from reward input"
        for key, value in batch.items():
            self.accounting[key] += value
        statuses = Counter(self.accounting["reward_status_counts"])
        statuses.update(self.last_batch["status_counts"])
        self.accounting["reward_status_counts"] = dict(statuses)
        self.accounting["last_batch"] = batch
        self._metrics["train"]["project/generated_completion_tokens"] = [self.accounting["generated_completion_tokens"]]
        self._metrics["train"]["project/mixed_group_fraction"] = [batch["mixed_reward_groups"] / batch["generated_groups"]]
        return output


class BudgetCallback(TrainerCallback):
    def __init__(self, trainer, wall_limit):
        self.trainer = trainer
        self.wall_limit = wall_limit
        self.stop_reason = None

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        elapsed = time.monotonic() - trainer.start_clock
        progress = {
            **trainer.accounting, "global_step": state.global_step, "elapsed_seconds": elapsed,
            "target_completion_tokens": trainer.budget,
            "effective_completion_tokens_per_second": trainer.accounting["generated_completion_tokens"] / max(elapsed, 1),
            "torch_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "torch_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
        }
        if trainer.accounting["generated_completion_tokens"] >= trainer.budget:
            self.stop_reason = "token_budget_reached"
        elif elapsed >= self.wall_limit:
            self.stop_reason = "wall_time_limit_incomplete"
        if self.stop_reason:
            control.should_training_stop = True
            control.should_save = True
        if state.global_step >= state.max_steps:
            control.should_save = True
        atomic_json(trainer.run_dir / "progress.json", progress)
        print(json.dumps({"event": "optimizer_step", "step": state.global_step,
                          "completion_tokens": progress["generated_completion_tokens"],
                          "elapsed_seconds": round(elapsed, 1), "stop_reason": self.stop_reason}), flush=True)
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        atomic_json(checkpoint / "completion_accounting.json", self.trainer.accounting)

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = dict(logs or {})
        for key, value in values.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"Nonfinite training metric: {key}={value}")
        with (self.trainer.run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": state.global_step, **values}, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wall-limit-hours", type=float, default=48)
    cli = parser.parse_args()
    cfg_path = ROOT / "training/baseline_config.json"
    cfg = json.loads(cfg_path.read_text())
    assets = json.loads((ROOT / "receipts/assets.json").read_text())
    manifest = json.loads((ROOT / "data/prepared/manifest.json").read_text())
    assert file_hash(cfg_path) == manifest["config_sha256"]
    data_path = ROOT / "data/prepared/train.jsonl"
    assert file_hash(data_path) == manifest["files"]["train.jsonl"]["sha256"]
    run = Path(cli.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "run_config.json").exists() and not cli.resume:
        raise RuntimeError("Run already exists; inspect and explicitly resume instead of overwriting")
    if not cli.pilot_steps:
        gate = json.loads((ROOT / "receipts/pilot_gate.json").read_text())
        assert gate["passed"] and gate["config_sha256"] == file_hash(cfg_path)
        for name, digest in gate["critical_code_sha256"].items():
            assert file_hash(ROOT / "training" / name) == digest, f"Pilot source changed: {name}"
        # The immutable pre-RL policy may be evaluated concurrently on GPU 1.
        # Its scores never select training settings or checkpoints.
        initial_path = ROOT / "runs/initial_policy/evaluation_config.json"
        assert initial_path.is_file(), "Start the initial-checkpoint evaluation before formal training"
        initial = json.loads(initial_path.read_text())
        assert initial["config_sha256"] == file_hash(cfg_path) and initial["model_revision"] == assets["model_revision"]
    versions = {p: importlib.metadata.version(p) for p in ["torch", "transformers", "trl", "peft", "vllm", "datasets", "math-verify"]}
    assert versions["trl"] == "0.23.1"
    spec = {"configuration": cfg, "config_sha256": file_hash(cfg_path), "model_revision": assets["model_revision"],
            "data_manifest_sha256": file_hash(ROOT / "data/prepared/manifest.json"), "packages": versions,
            "code_sha256": {p.name: file_hash(p) for p in (ROOT / "training").glob("*.py")},
            "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pilot_steps": cli.pilot_steps, "started_unix": time.time(), "pid": os.getpid(),
            "raw_vllm_engine_seed": 0, "seed_scope": "actor adapter initialization, data shuffling, torch RNG; pinned TRL colocated vLLM engine seed is 0"}
    if cli.resume:
        old = json.loads((run / "run_config.json").read_text())
        for key in ["config_sha256", "model_revision", "data_manifest_sha256", "packages", "code_sha256"]:
            assert old[key] == spec[key], f"Resume provenance mismatch: {key}"
    else:
        atomic_json(run / "run_config.json", spec)
    status = {"status": "initializing", "pid": os.getpid(), "started_unix": time.time(), "pilot": bool(cli.pilot_steps)}
    atomic_json(run / "status.json", status)
    try:
        set_seed(cfg["seed"])
        tokenizer = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(assets["model_path"], local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation=cfg["attention_implementation"])
        model.config.use_cache = False
        data = Dataset.from_list(read_jsonl(data_path))
        trainer_holder = {}
        def outcome_accuracy(prompts, completions, completion_ids, answer, problem_id, **kwargs):
            values, statuses = [], Counter()
            truncations = 0
            trainer = trainer_holder["trainer"]
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
        params = dict(
            output_dir=str(run), seed=cfg["seed"], data_seed=cfg["seed"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            max_steps=cli.pilot_steps or 20000, learning_rate=cfg["learning_rate"],
            lr_scheduler_type=cfg["lr_scheduler_type"], warmup_steps=cfg["warmup_steps"],
            weight_decay=cfg["weight_decay"], max_grad_norm=cfg["max_grad_norm"],
            bf16=True, tf32=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, disable_dropout=True,
            num_generations=cfg["group_size"], max_prompt_length=cfg["max_prompt_tokens"],
            max_completion_length=cfg["max_new_tokens"], temperature=cfg["temperature"],
            top_p=cfg["top_p"], top_k=cfg["top_k"], beta=cfg["beta"], epsilon=cfg["epsilon"],
            loss_type=cfg["loss_type"], scale_rewards=cfg["scale_rewards"], num_iterations=cfg["num_iterations"],
            mask_truncated_completions=False, use_vllm=cfg["use_vllm"], vllm_mode="colocate",
            vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
            vllm_enable_sleep_mode=cfg["vllm_enable_sleep_mode"],
            vllm_importance_sampling_correction=cfg["vllm_importance_sampling_correction"],
            logging_steps=1, logging_first_step=True, logging_nan_inf_filter=False,
            save_strategy="steps", save_steps=cfg["save_steps"], save_total_limit=cfg["save_total_limit"],
            report_to="none", push_to_hub=False, remove_unused_columns=False, dataloader_num_workers=0,
            dataloader_pin_memory=True, disable_tqdm=True,
        )
        lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                          target_modules=cfg["lora_target_modules"], bias="none", task_type="CAUSAL_LM")
        resume_accounting = json.loads((Path(cli.resume) / "completion_accounting.json").read_text()) if cli.resume else None
        trainer = AuditedGRPO(model=model, processing_class=tokenizer, reward_funcs=outcome_accuracy,
                    args=GRPOConfig(**params), train_dataset=data, peft_config=lora, run_dir=run,
                    budget=cfg["generated_token_budget"], resume_accounting=resume_accounting)
        trainer_holder["trainer"] = trainer
        budget_callback = BudgetCallback(trainer, cli.wall_limit_hours * 3600)
        trainer.add_callback(budget_callback)
        atomic_json(run / "status.json", {**status, "status": "training"})
        result = trainer.train(resume_from_checkpoint=cli.resume)
        trainer.save_model(str(run / "final_adapter"))
        tokenizer.save_pretrained(run / "final_adapter")
        trainer.save_state()
        lora_b_abs_sum = sum(float(p.detach().float().abs().sum()) for n, p in trainer.model.named_parameters() if "lora_B" in n)
        complete = trainer.accounting["generated_completion_tokens"] >= cfg["generated_token_budget"]
        final = {**status, "status": "pilot_completed" if cli.pilot_steps else ("completed" if complete else "incomplete"),
                 "finished_unix": time.time(), "global_step": trainer.state.global_step,
                 "stop_reason": budget_callback.stop_reason or "max_steps", "accounting": trainer.accounting,
                 "budget": cfg["generated_token_budget"], "budget_overshoot": max(0, trainer.accounting["generated_completion_tokens"] - cfg["generated_token_budget"]),
                 "lora_B_abs_sum": lora_b_abs_sum, "metrics": result.metrics}
        atomic_json(run / "status.json", final)
        print(json.dumps(final, indent=2), flush=True)
        if not cli.pilot_steps and not complete:
            raise RuntimeError("Stopped before the declared token budget; checkpoint retained for explicit resume")
    except Exception as exc:
        atomic_json(run / "failure.json", {"error": str(exc), "traceback": traceback.format_exc(), "time": time.time()})
        raise

if __name__ == "__main__":
    main()
