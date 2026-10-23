"""Independent Bprime run; original GRPO, data, verifier and B accounting reused."""
import argparse
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
import random
import sys
import time
import traceback

from support import HERE, ROOT, atomic, read, sha, source_hashes
for folder in ["training", "operations", "training_b", "experiments/reward_repair_v2_20261021"]:
    sys.path.insert(0, str(ROOT / folder))
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig
from common import read_jsonl
from reward import outcome
from train_b import BTrainer, BMilestones
from prm_reward import VALID_STATUSES, mixed_reward
from b_audit_patch import patched_b_trainer_class
from frozen_prm import FrozenQwenPRM


def adapter_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(get_peft_model_state_dict(model).items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    cli = parser.parse_args()
    plan, cfg = read(HERE / "protocol.json"), read(ROOT / "training/baseline_config.json")
    run = ROOT / plan["pilot" if cli.pilot else "run"]
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (run / "run_config.json").exists(), "Prior attempt exists; do not overwrite or restart"
    hashes = source_hashes()
    release = read(ROOT / plan["run"] / "execution_source_hashes.json")
    assert hashes == release
    if not cli.pilot:
        gate = read(ROOT / plan["run"] / "pilot_gate.json")
        assert gate["passed"] and gate["source_hashes"] == hashes
    origin = read(ROOT / "runs/A_seed17/run_config.json")
    assets = read(ROOT / "receipts/assets.json")
    prm_assets = read(ROOT / "audits/qwen7b_offline_20261021/model_assets.json")
    assert prm_assets["complete"] and prm_assets["revision"] == plan["prm_revision"]
    for record in prm_assets["files"]:
        assert sha(ROOT / plan["prm_path"] / record["name"]) == record["sha256"]
    manifest = read(ROOT / "data/prepared/manifest.json")
    assert cfg["seed"] == plan["seed"] == 17
    assert sha(ROOT / "training/baseline_config.json") == manifest["config_sha256"] == origin["config_sha256"]
    assert sha(ROOT / "data/prepared/manifest.json") == origin["data_manifest_sha256"]
    assert sha(ROOT / "data/prepared/train.jsonl") == manifest["files"]["train.jsonl"]["sha256"]
    assert assets["model_revision"] == origin["model_revision"]
    for name in ["train_baseline.py", "common.py", "reward.py", "evaluate_policy.py"]:
        assert sha(ROOT / "training" / name) == origin["code_sha256"][name]
    packages = {p: importlib.metadata.version(p) for p in origin["packages"]}
    assert packages == origin["packages"] and packages["trl"] == "0.23.1"
    saved_path = ROOT / "runs/A_seed17/checkpoint-280/training_args.bin"
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    params = {f.name: getattr(saved, f.name) for f in dataclasses.fields(saved) if f.init}
    params.update(output_dir=str(run), logging_dir=str(run / "tensorboard"), generation_batch_size=None)
    if cli.pilot:
        params["max_steps"] = plan["pilot_steps"]
    args = GRPOConfig(**params)
    changes = {k: [saved.to_dict().get(k), v] for k, v in args.to_dict().items()
               if saved.to_dict().get(k) != v}
    assert set(changes) <= {"output_dir", "logging_dir", "max_steps"}
    assert cli.pilot or "max_steps" not in changes
    assert args.generation_batch_size == 32 and args.num_generations == 8 and torch.cuda.device_count() == 1
    spec = {"arm": plan["experiment"], "plan": plan, "source_hashes": hashes,
        "model_revision": assets["model_revision"], "config_sha256": origin["config_sha256"],
        "data_manifest_sha256": origin["data_manifest_sha256"], "code_sha256": origin["code_sha256"],
        "packages": packages, "prm_revision": plan["prm_revision"], "argument_changes": changes,
        "source_checkpoint": None, "base_actor_initialization": True, "raw_vllm_engine_seed": 0,
        "target_completion_tokens": plan["target_completion_tokens"], "pilot": cli.pilot,
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(), "started_unix": time.time(), "trusted_args_source": str(saved_path)}
    atomic(run / "run_config.json", spec)
    atomic(run / "milestones.json", {"run": plan["experiment"], "cumulative_completion_tokens": True, "milestones": {}})
    status = {"status": "initializing", "pid": os.getpid(), "started_unix": time.time(), "pilot": cli.pilot}
    atomic(run / "status.json", status)
    try:
        set_seed(cfg["seed"])
        tokenizer = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(assets["model_path"], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation=cfg["attention_implementation"])
        model.config.use_cache = False
        rows = read_jsonl(ROOT / "data/prepared/train.jsonl")
        with (ROOT / "runs/B_seed17_6M/rollouts.jsonl").open() as stream:
            original_first = [json.loads(line) for line in itertools.islice(stream, 32)]
        holder = {}

        def outcome_and_process(prompts, completions, completion_ids, answer, problem_id, problem, **unused):
            trainer = holder["trainer"]
            records, valid_indices = [], []
            if trainer.state.global_step == 0:
                first = {"same_problem_ids": list(problem_id) == [r["problem_id"] for r in original_first],
                    "same_completions": list(completions) == [r["completion"] for r in original_first],
                    "same_token_counts": [len(ids) for ids in completion_ids] == [r["completion_tokens"] for r in original_first]}
                atomic(run / "initial_sampling_match.json", first)
                assert all(first.values()), "Initial actor sampling differs from original B; investigate before training"
            for i, (text, ids, gold, ident) in enumerate(zip(completions, completion_ids, answer, problem_id)):
                terminated = tokenizer.eos_token_id in ids
                judged = outcome(text, gold, terminated)
                valid = judged["status"] in VALID_STATUSES
                records.append({"optimizer_step_before_update": trainer.state.global_step,
                    "problem_id": ident, "gold": gold, "completion": text, "completion_tokens": len(ids),
                    "terminated": terminated, "status": judged["status"], "answer": judged["answer"],
                    "outcome_reward": judged["reward"], "valid_for_process_reward": valid,
                    "step_scores": [], "min_score": 0.0, "step_count": 0, "prm_input_tokens": 0,
                    "raw_no_double_newline": "\n\n" not in text.strip()})
                if valid:
                    valid_indices.append(i)
            scored, cost = trainer.prm.score([problem[i] for i in valid_indices], [completions[i] for i in valid_indices])
            assert len(scored) == len(valid_indices)
            for i, values in zip(valid_indices, scored):
                records[i].update(values)
            values = []
            with (run / "rollouts.jsonl").open("a") as stream:
                for row in records:
                    row["reward"] = mixed_reward({"status": row["status"], "reward": row["outcome_reward"]},
                        row["step_scores"], plan["process_reward_lambda"])
                    values.append(row["reward"])
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            trainer.last_records, trainer.last_reward_values, trainer.last_prm_cost = records, values, cost
            return values

        lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_target_modules"], bias="none", task_type="CAUSAL_LM")
        cls = patched_b_trainer_class(BTrainer, audit_directory=run / "advantage_audits")
        trainer = cls(model=model, processing_class=tokenizer, reward_funcs=outcome_and_process, args=args,
            train_dataset=Dataset.from_list(rows), peft_config=lora, run_dir=run,
            budget=plan["target_completion_tokens"])
        holder["trainer"] = trainer
        initial_digest = adapter_digest(trainer.model)
        py_state, np_state = random.getstate(), np.random.get_state()
        cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            trainer.prm = FrozenQwenPRM(ROOT / plan["prm_path"], receipt_dir=run,
                microbatch_size=plan["prm_microbatch_size"])
        random.setstate(py_state)
        np.random.set_state(np_state)
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
        assert initial_digest == adapter_digest(trainer.model)
        actor_ids = {id(p) for p in trainer.model.parameters()}
        assert all(id(p) not in actor_ids for p in trainer.prm.model.parameters())
        atomic(run / "initialization_audit.json", {"actor_adapter_sha256": initial_digest,
            "prm_load_preserved_actor_adapter": True, "python_numpy_torch_rng_restored": True,
            "actor_prm_parameters_disjoint": True, "numeric_audit": cls.numeric_patch_receipt,
            "PRM_frozen": True, "single_gpu": True, "argument_changes": changes})
        callback = BMilestones(trainer, {"wall_limit_hours": plan["wall_limit_hours"],
            "milestones": [] if cli.pilot else [plan["target_completion_tokens"]]}, 0)
        trainer.add_callback(callback)
        atomic(run / "status.json", {**status, "status": "training"})
        result = trainer.train()
        trainer.prm.assert_frozen()
        assert adapter_digest(trainer.model) != initial_digest, "Actor adapter did not update"
        trainer.save_model(str(run / "final_adapter"))
        tokenizer.save_pretrained(run / "final_adapter")
        trainer.save_state()
        complete = trainer.accounting["generated_completion_tokens"] >= plan["target_completion_tokens"]
        final = {**status, "status": "pilot_completed" if cli.pilot else ("completed" if complete else "incomplete"),
            "finished_unix": time.time(), "global_step": trainer.state.global_step,
            "accounting": trainer.accounting, "budget": plan["target_completion_tokens"],
            "stop_reason": callback.stop_reason or "pilot_max_steps", "metrics": result.metrics,
            "prm_frozen_verified": True, "actor_adapter_changed": True,
            "lora_B_abs_sum": sum(float(p.detach().float().abs().sum()) for n, p in trainer.model.named_parameters() if "lora_B" in n)}
        atomic(run / "status.json", final)
        print(json.dumps(final, indent=2), flush=True)
        assert cli.pilot or complete, "Incomplete run retained; do not restart from base"
    except Exception as exc:
        atomic(run / "failure.json", {"error": str(exc), "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
