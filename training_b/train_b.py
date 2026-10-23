"""B/17: unchanged GRPO and actor settings, frozen native-step PRM reward.

The A code and checkpoints are read-only. Exact checkpoint-restoration audits
and milestone preservation reuse the already-tested continuation helpers.
"""
import argparse
from collections import Counter
import dataclasses
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "operations"))
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer
from common import atomic_json, file_hash, read_jsonl
from reward import outcome
from train_baseline import AuditedGRPO, BudgetCallback
from continue_baseline import VerifiedContinuation, MilestoneCallback
from prm_reward import FrozenPRM, VALID_STATUSES, group_diagnostics, mixed_reward


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    paths = list((ROOT / "training_b").glob("*.py")) + [ROOT / "training_b/config.json"]
    paths += [ROOT / p for p in ["training/train_baseline.py", "training/common.py", "training/reward.py",
                               "training/evaluate_policy.py", "operations/continue_baseline.py",
                               "operations/run_continuation.py", "operations/lora_preflight.py"]]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


class BTrainer(AuditedGRPO):
    def __init__(self, *args, resume_accounting=None, **kwargs):
        counters = {k: 0 for k in ["generated_completion_tokens", "generated_completions", "generated_groups",
                    "mixed_reward_groups", "correct_completions", "truncated_completions",
                    "prm_input_tokens", "prm_padded_tokens", "prm_scored_completions", "native_steps",
                    "single_step_completions", "no_double_newline_completions"]}
        counters.update(prm_forward_seconds=0.0, reward_status_counts={}, outcome_group_counts={})
        super().__init__(*args, resume_accounting=resume_accounting or counters, **kwargs)

    def _generate_and_score_completions(self, inputs):
        assert self.accelerator.num_processes == 1
        # Deliberately bypass outcome-only accounting; leave TRL's algorithm untouched.
        output = GRPOTrainer._generate_and_score_completions(self, inputs)
        if not self.model.training:
            return output
        rows = self.last_records
        expected, groups = group_diagnostics(
            [r["outcome_reward"] for r in rows], self.last_reward_values,
            [r["valid_for_process_reward"] for r in rows], [r["min_score"] for r in rows], self.num_generations)
        torch.testing.assert_close(output["advantages"], expected.to(output["advantages"].device), atol=2e-5, rtol=2e-5)
        assert torch.isfinite(output["advantages"]).all()
        tokens = int(output["completion_mask"].sum())
        assert tokens == sum(r["completion_tokens"] for r in rows)
        batch = {"generated_completion_tokens": tokens, "generated_completions": len(rows),
                 "generated_groups": len(groups), "mixed_reward_groups": sum(not g["zero_reward_variance"] for g in groups),
                 "correct_completions": int(sum(r["outcome_reward"] for r in rows)),
                 "truncated_completions": sum(not r["terminated"] for r in rows),
                 "native_steps": sum(r["step_count"] for r in rows),
                 "single_step_completions": sum(r["step_count"] == 1 for r in rows),
                 "no_double_newline_completions": sum(r["raw_no_double_newline"] for r in rows), **self.last_prm_cost}
        for key, value in batch.items():
            self.accounting[key] += value
        statuses = Counter(self.accounting["reward_status_counts"])
        statuses.update(r["status"] for r in rows)
        self.accounting["reward_status_counts"] = dict(statuses)
        counts = Counter(self.accounting["outcome_group_counts"])
        for g in groups:
            counts[g["outcome_type"]] += 1
            counts[g["outcome_type"] + ("_all_valid" if g["all_valid"] else "_has_invalid")] += 1
        self.accounting["outcome_group_counts"] = dict(counts)
        self.accounting["last_batch"] = batch
        with (self.run_dir / "group_diagnostics.jsonl").open("a") as stream:
            for i, g in enumerate(groups):
                ids = {r["problem_id"] for r in rows[i * self.num_generations:(i + 1) * self.num_generations]}
                assert len(ids) == 1
                stream.write(json.dumps({"optimizer_step_before_update": self.state.global_step,
                                         "problem_id": ids.pop(), **g}) + "\n")
        self._metrics["train"]["project/generated_completion_tokens"] = [self.accounting["generated_completion_tokens"]]
        self._metrics["train"]["project/outcome_accuracy"] = [batch["correct_completions"] / len(rows)]
        self._metrics["train"]["project/prm_seconds"] = [self.last_prm_cost["prm_forward_seconds"]]
        return output


class BContinuation(VerifiedContinuation, BTrainer):
    """VerifiedContinuation's super() dispatches to B's nonbinary accounting."""


class BMilestones(MilestoneCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if hasattr(self.trainer, "source_step"):
            return super().on_train_begin(args, state, control, **kwargs)
        assert state.global_step == 0
        self.trainer.start_clock = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer.prm.assert_frozen()
        if hasattr(self.trainer, "source_step"):
            return super().on_step_end(args, state, control, **kwargs)
        control = BudgetCallback.on_step_end(self, args, state, control, **kwargs)
        tokens = self.trainer.accounting["generated_completion_tokens"]
        for target in self.targets:
            if tokens >= target and target not in self.pending:
                if not (self.trainer.run_dir / "milestones" / str(target) / "milestone.json").exists():
                    self.pending.append(target)
        if self.pending:
            control.should_save = True
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--prior-run")
    parser.add_argument("--pilot-max-steps", type=int, default=0)
    cli = parser.parse_args()
    bcfg = read(ROOT / "training_b/config.json")
    cfg = read(ROOT / "training/baseline_config.json")
    origin = read(ROOT / "runs/A_seed17/run_config.json")
    assets = read(ROOT / "receipts/assets.json")
    prm_assets = read(ROOT / "receipts/B_prm_assets.json")
    manifest_path = ROOT / "data/prepared/manifest.json"
    manifest = read(manifest_path)
    assert cfg["seed"] == bcfg["seed"] == 17
    assert manifest["config_sha256"] == origin["config_sha256"] == file_hash(ROOT / "training/baseline_config.json")
    assert origin["data_manifest_sha256"] == file_hash(manifest_path)
    assert assets["model_revision"] == origin["model_revision"]
    assert prm_assets["status"] == "completed" and prm_assets["model_revision"] == bcfg["prm_revision"]
    for name, data in prm_assets["files"].items():
        assert file_hash(Path(prm_assets["model_path"]) / name) == data["sha256"]
    packages = {p: importlib.metadata.version(p) for p in origin["packages"]}
    assert packages == origin["packages"] and packages["trl"] == "0.23.1"
    assert file_hash(ROOT / "data/prepared/train.jsonl") == manifest["files"]["train.jsonl"]["sha256"]
    for name in ["train_baseline.py", "common.py", "reward.py", "evaluate_policy.py"]:
        assert file_hash(ROOT / "training" / name) == origin["code_sha256"][name]
    run = Path(cli.run_dir).resolve()
    assert run.is_relative_to(ROOT / "runs") and run.name.startswith(("B_", "pilot_B_"))
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (run / "run_config.json").exists(), "Inspect existing run; do not overwrite or restart"
    hashes = source_hashes()
    if not cli.pilot_max_steps:
        gate = read(ROOT / "receipts/B_pilot_gate.json")
        assert gate["passed"] and gate["source_hashes"] == hashes
        assert gate["prm_revision"] == bcfg["prm_revision"]
    source = Path(cli.resume) if cli.resume else None
    accounting = read(source / "completion_accounting.json") if source else None
    initial_tokens = accounting["generated_completion_tokens"] if accounting else 0
    saved_args_path = source / "training_args.bin" if source else ROOT / "runs/A_seed17/checkpoint-280/training_args.bin"
    # Only our own previously generated trusted checkpoint, never downloaded pickles.
    saved_args = torch.load(saved_args_path, map_location="cpu", weights_only=False)
    params = {f.name: getattr(saved_args, f.name) for f in dataclasses.fields(saved_args) if f.init}
    params.update(output_dir=str(run), logging_dir=str(run / "tensorboard"), generation_batch_size=None)
    if cli.pilot_max_steps:
        params["max_steps"] = cli.pilot_max_steps
    args = GRPOConfig(**params)
    changes = {k: [saved_args.to_dict().get(k), v] for k, v in args.to_dict().items() if saved_args.to_dict().get(k) != v}
    assert set(changes) <= {"output_dir", "logging_dir", "max_steps"}
    if not cli.pilot_max_steps:
        assert "max_steps" not in changes
    assert args.generation_batch_size == 32 and args.num_generations == 8
    rows = read_jsonl(ROOT / "data/prepared/train.jsonl")
    extra = {}
    if source:
        prior = read(Path(cli.prior_run) / "run_config.json")
        assert prior["B_source_hashes"] == hashes
        assert prior["prm_revision"] == bcfg["prm_revision"]
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
        groups = accounting["generated_groups"]
        assert groups + 4 < len(rows), "Extend audited multi-epoch resume logic before continuing"
        prior_records = read_jsonl(Path(cli.prior_run) / "rollouts.jsonl")
        expected = [rows[i]["problem_id"] for i in order[:groups] for _ in range(8)]
        assert [r["problem_id"] for r in prior_records] == expected
        extra = {"source": source, "expected_first_ids": [rows[i]["problem_id"] for i in order[groups:groups + 4] for _ in range(8)]}
    spec = {"arm": bcfg["arm"], "configuration": cfg, "B_configuration": bcfg, "config_sha256": origin["config_sha256"],
            "model_revision": assets["model_revision"], "data_manifest_sha256": file_hash(manifest_path),
            "code_sha256": origin["code_sha256"], "B_source_hashes": hashes, "packages": packages,
            "prm_revision": bcfg["prm_revision"], "prm_assets_sha256": file_hash(ROOT / "receipts/B_prm_assets.json"),
            "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pilot_max_steps": cli.pilot_max_steps, "started_unix": time.time(), "pid": os.getpid(),
            "argument_changes": changes, "source_checkpoint": str(source) if source else None,
            "target_completion_tokens": cli.target, "raw_vllm_engine_seed": 0,
            "base_actor_initialization": not bool(source), "B_evaluation_request_name_note": "The unchanged evaluator's LoRARequest name A_seed17 is cosmetic; adapter path and SHA identify B."}
    atomic_json(run / "run_config.json", spec)
    atomic_json(run / "milestones.json", {"run": "B_seed17", "cumulative_completion_tokens": True, "milestones": {}})
    status = {"status": "initializing", "pid": os.getpid(), "started_unix": time.time(), "pilot": bool(cli.pilot_max_steps)}
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

        def outcome_and_process(prompts, completions, completion_ids, answer, problem_id, problem, **unused):
            trainer = holder["trainer"]
            records, valid_indices = [], []
            for i, (text, ids, gold, ident) in enumerate(zip(completions, completion_ids, answer, problem_id)):
                terminated = tokenizer.eos_token_id in ids
                judged = outcome(text, gold, terminated)
                valid = judged["status"] in VALID_STATUSES
                records.append({"optimizer_step_before_update": trainer.state.global_step, "problem_id": ident,
                    "gold": gold, "completion": text, "completion_tokens": len(ids), "terminated": terminated,
                    "status": judged["status"], "answer": judged["answer"], "outcome_reward": judged["reward"],
                    "valid_for_process_reward": valid, "step_scores": [], "min_score": 0.0, "step_count": 0,
                    "prm_input_tokens": 0, "raw_no_double_newline": "\n\n" not in text.strip()})
                if valid:
                    valid_indices.append(i)
            scores, cost = trainer.prm.score([problem[i] for i in valid_indices], [completions[i] for i in valid_indices])
            for i, scored in zip(valid_indices, scores):
                records[i].update(scored)
            values = []
            with (run / "rollouts.jsonl").open("a") as stream:
                for row in records:
                    row["reward"] = mixed_reward({"status": row["status"], "reward": row["outcome_reward"]},
                                                  row["step_scores"], bcfg["process_reward_lambda"])
                    values.append(row["reward"])
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            trainer.last_records, trainer.last_reward_values, trainer.last_prm_cost = records, values, cost
            return values

        lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                         target_modules=cfg["lora_target_modules"], bias="none", task_type="CAUSAL_LM")
        cls = BContinuation if source else BTrainer
        trainer = cls(model=model, processing_class=tokenizer, reward_funcs=outcome_and_process, args=args,
                      train_dataset=Dataset.from_list(rows), peft_config=lora, run_dir=run,
                      budget=cli.target, resume_accounting=accounting, **extra)
        holder["trainer"] = trainer
        # PRM loading must not consume actor/dropout/data RNG or change LoRA initialization.
        py_state, np_state = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            trainer.prm = FrozenPRM(prm_assets["model_path"], microbatch_size=bcfg["prm_microbatch_size"])
        random.setstate(py_state)
        np.random.set_state(np_state)
        actor_ids = {id(p) for p in trainer.model.parameters()}
        assert all(id(p) not in actor_ids for p in trainer.prm.model.parameters())
        plan = {"wall_limit_hours": bcfg["wall_limit_hours"],
                "milestones": [t for t in bcfg["milestones"] if initial_tokens < t <= cli.target]}
        callback = BMilestones(trainer, plan, initial_tokens)
        trainer.add_callback(callback)
        atomic_json(run / "status.json", {**status, "status": "training"})
        result = trainer.train(resume_from_checkpoint=str(source) if source else None)
        trainer.prm.assert_frozen()
        trainer.save_model(str(run / "final_adapter"))
        tokenizer.save_pretrained(run / "final_adapter")
        trainer.save_state()
        complete = trainer.accounting["generated_completion_tokens"] >= cli.target
        final = {**status, "status": "pilot_completed" if cli.pilot_max_steps else ("completed" if complete else "incomplete"),
                 "finished_unix": time.time(), "global_step": trainer.state.global_step, "accounting": trainer.accounting,
                 "budget": cli.target, "stop_reason": callback.stop_reason or "max_steps", "metrics": result.metrics,
                 "prm_frozen_verified": True, "initial_completion_tokens": initial_tokens,
                 "lora_B_abs_sum": sum(float(p.detach().float().abs().sum()) for n, p in trainer.model.named_parameters() if "lora_B" in n)}
        atomic_json(run / "status.json", final)
        print(json.dumps(final, indent=2), flush=True)
        assert cli.pilot_max_steps or complete, "Incomplete run retained; do not restart from base"
    except Exception as exc:
        atomic_json(run / "failure.json", {"error": str(exc), "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
