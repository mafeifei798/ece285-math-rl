"""Authorized corrected-only A/B runner, same original GRPO and actor settings."""
import argparse
from collections import Counter
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
import traceback
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig
from pair_utils import HERE, ROOT, read, atomic, sha, read_jsonl, verify_sources

sys.path.insert(0, str(ROOT / "experiments/b_baseline_corrected_20261021"))
from trainer import trainer_class
from rewards import RewardLedger
sys.path.insert(0, str(ROOT / "experiments/reward_repair_v3_20261021"))
from repairs import repaired_outcome, valid_for_prm
sys.path.insert(0, str(ROOT / "experiments/b_qwen7b_6m_20261021"))
from frozen_prm import FrozenQwenPRM
from train_baseline import BudgetCallback


def adapter_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(get_peft_model_state_dict(model).items()):
        t = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class OutcomeLedger:
    """Exactly the shared repaired verifier; no process scoring for arm A."""
    def __init__(self, units):
        self.units, self.records = units, []
        self.cost = dict(prm_input_tokens=0, prm_padded_tokens=0,
                         prm_scored_completions=0, prm_forward_seconds=0.0)

    def callback(self, eos, owner):
        def outcome_only(prompts, completions, completion_ids, answer, problem_id, problem, **unused):
            self.records = []
            for text, ids, gold, ident, question in zip(completions, completion_ids, answer, problem_id, problem):
                terminated = eos in ids
                result = repaired_outcome(text, gold, terminated, question=question,
                    problem_id=ident, unit_metadata=self.units)
                self.records.append(dict(optimizer_step_before_update=owner["trainer"].state.global_step,
                    problem_id=ident, gold=gold, completion=text, completion_tokens=len(ids),
                    terminated=terminated, status=result["status"], answer=result["answer"],
                    outcome_reward=result["reward"], reward=result["reward"],
                    valid_output=valid_for_prm(result), valid_for_process_reward=False,
                    step_scores=[], min_score=0.0, step_count=0, prm_input_tokens=0,
                    raw_no_double_newline="\n\n" not in text.strip()))
            return [r["reward"] for r in self.records]
        return outcome_only


class LiveCallback(BudgetCallback):
    def on_step_end(self, args, state, control, **kwargs):
        t = self.trainer
        if getattr(t, "prm", None) is not None:
            t.prm.assert_frozen()
        assert t.baseline_loss_rows == t.accounting["generated_completions"], "Loss coverage differs from rollouts"
        assert all(c["advantage_tensor_retained"] for c in t.baseline_generation_checks)
        atomic(t.run_dir / "live_audits.json", dict(passed=True, global_step=state.global_step,
            loss_calls=t.baseline_loss_calls, loss_rows=t.baseline_loss_rows,
            generation_batches=len(t.baseline_generation_checks),
            original_TRL_advantages_retained=True, loss_input_values_preserved=True,
            PRM_frozen=True if getattr(t, "prm", None) else None))
        return super().on_step_end(args, state, control, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["A", "B"], required=True)
    p.add_argument("--pilot", action="store_true")
    cli = p.parse_args()
    plan, release = read(HERE / "plan.json"), verify_sources()
    run = ROOT / plan["pilots" if cli.pilot else "runs"][cli.arm]
    assert not run.exists(), "Attempt exists; inspect it rather than restarting"
    if not cli.pilot:
        gate = read(ROOT / plan["controller"] / "pilot_gate.json")
        assert gate["passed"] and gate["release_sha256"] == sha(HERE / "release.json")
    run.mkdir(parents=True)
    lock = (run / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(run / "status.json", dict(status="initializing", pid=os.getpid(), arm=cli.arm, pilot=cli.pilot))
    try:
        cfg = read(ROOT / "training/baseline_config.json")
        assets = read(ROOT / "receipts/assets.json")
        origin = read(ROOT / "runs/A_seed17/run_config.json")
        versions = {p: importlib.metadata.version(p) for p in origin["packages"]}
        assert versions == origin["packages"]
        assert assets["model_revision"] == origin["model_revision"] == "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
        assert cfg["seed"] == 17 and torch.cuda.device_count() == 1
        saved = torch.load(ROOT / "runs/A_seed17/checkpoint-280/training_args.bin", map_location="cpu", weights_only=False)
        params = {f.name: getattr(saved, f.name) for f in dataclasses.fields(saved) if f.init}
        params.update(output_dir=str(run), logging_dir=str(run / "tensorboard"), generation_batch_size=None)
        if cli.pilot:
            params["max_steps"] = 2
        args = GRPOConfig(**params)
        changes = {k:[saved.to_dict().get(k),v] for k,v in args.to_dict().items() if saved.to_dict().get(k)!=v}
        assert set(changes) <= {"output_dir", "logging_dir", "max_steps"}
        assert cli.pilot or "max_steps" not in changes
        assert args.generation_batch_size == 32 and args.num_generations == 8
        cls = trainer_class()
        set_seed(17)
        tokenizer = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(assets["model_path"], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation=cfg["attention_implementation"])
        model.config.use_cache = False
        units = read(ROOT / "audits/reward_repair_v3_20261021/unit_metadata.json")
        ledger = OutcomeLedger(units) if cli.arm == "A" else RewardLedger(None, units)
        owner = {}
        base_callback = ledger.callback(tokenizer.eos_token_id, owner)

        def rewards(prompts, completions, completion_ids, answer, problem_id, problem, **kw):
            result = base_callback(prompts=prompts, completions=completions, completion_ids=completion_ids,
                answer=answer, problem_id=problem_id, problem=problem, **kw)
            if owner["trainer"].state.global_step == 0:
                signature = [dict(problem_id=i, completion=c, completion_tokens=len(t))
                             for i,c,t in zip(problem_id,completions,completion_ids)]
                atomic(run / "initial_sampling.json", signature)
                old = read_jsonl(ROOT / "runs/B_Qwen7B_seed17_6M/rollouts.jsonl")[:32]
                assert list(problem_id) == [r["problem_id"] for r in old]
                atomic(run / "initial_sampling_match.json", dict(problem_ids=True,
                    historical_completions=list(completions)==[r["completion"] for r in old]))
            return result

        lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_target_modules"], bias="none", task_type="CAUSAL_LM")
        trainer = cls(model=model, processing_class=tokenizer, args=args,
            train_dataset=Dataset.from_list(read_jsonl(ROOT / "data/prepared/train.jsonl")),
            peft_config=lora, reward_funcs=rewards, baseline_ledger=ledger,
            run_dir=run, budget=6000000)
        owner["trainer"] = trainer
        trainer.prm = None
        initial_digest = adapter_digest(trainer.model)
        if cli.arm == "B":
            py_state, np_state = random.getstate(), np.random.get_state()
            cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                trainer.prm = FrozenQwenPRM(ROOT / plan["prm_path"], receipt_dir=run, microbatch_size=4)
            random.setstate(py_state); np.random.set_state(np_state)
            assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state())
            assert adapter_digest(trainer.model) == initial_digest
            actor_ids = {id(p) for p in trainer.model.parameters()}
            assert all(id(p) not in actor_ids for p in trainer.prm.model.parameters())
            ledger.scorer.frozen_scorer = trainer.prm
        spec = dict(arm=cli.arm, seed=17, pilot=cli.pilot, source_checkpoint=None,
            base_actor_initialization=True, actor_adapter_initial_sha256=initial_digest,
            packages=versions, model_revision=assets["model_revision"],
            config_sha256=sha(ROOT / "training/baseline_config.json"),
            data_manifest_sha256=sha(ROOT / "data/prepared/manifest.json"), code_sha256=origin["code_sha256"],
            argument_changes=changes, target_completion_tokens=6000000,
            shared_training_arguments={k:v for k,v in args.to_dict().items() if k not in {"output_dir","logging_dir"}},
            verifier_version="reward-repair-v3-20261021", unit_metadata_sha256=release["unit_metadata_sha256"],
            reward="outcome only" if cli.arm=="A" else "outcome+0.2*min(native step probabilities)",
            advantage="original_TRL_mixed_reward_group_standardization", raw_vllm_engine_seed=0,
            gpu=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            source_release_sha256=sha(HERE / "release.json"), started_unix=time.time(), pid=os.getpid())
        atomic(run / "run_config.json", spec)
        callback = LiveCallback(trainer, 12*3600)
        trainer.start_clock = time.monotonic()
        trainer.add_callback(callback)
        atomic(run / "status.json", dict(status="training", pid=os.getpid(), arm=cli.arm, pilot=cli.pilot))
        result = trainer.train()
        if trainer.prm:
            trainer.prm.assert_frozen()
        changed = adapter_digest(trainer.model) != initial_digest
        assert changed, "Actor adapter did not update"
        trainer.save_model(str(run / "final_adapter")); tokenizer.save_pretrained(run / "final_adapter")
        trainer.save_state()
        checkpoint = run / ("checkpoint-" + str(trainer.state.global_step))
        assert all((checkpoint/f).is_file() for f in ["optimizer.pt","scheduler.pt","rng_state.pth","completion_accounting.json"])
        tokens = trainer.accounting["generated_completion_tokens"]
        complete = tokens >= 6000000
        if not cli.pilot:
            assert complete and tokens-6000000 < 32*2048
        assert not any(r["status"].startswith("verifier_error") for r in read_jsonl(run / "rollouts.jsonl"))
        final = dict(status="pilot_completed" if cli.pilot else "completed", arm=cli.arm, pilot=cli.pilot,
            global_step=trainer.state.global_step, accounting=trainer.accounting, budget=6000000,
            stop_reason=callback.stop_reason or "pilot_max_steps", actor_adapter_changed=changed,
            lora_B_abs_sum=sum(float(p.detach().float().abs().sum()) for n,p in trainer.model.named_parameters() if "lora_B" in n),
            prm_frozen_verified=bool(trainer.prm), metrics=result.metrics, finished_unix=time.time(),
            checkpoint=str(checkpoint), checkpoint_sha256={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()})
        atomic(run / "status.json", final)
        print(json.dumps(final, indent=2), flush=True)
    except Exception as exc:
        atomic(run / "failure.json", dict(error=str(exc), traceback=traceback.format_exc(), time=time.time()))
        atomic(run / "status.json", dict(status="failed", arm=cli.arm, pilot=cli.pilot, error=str(exc), pid=os.getpid()))
        raise


if __name__ == "__main__":
    main()
