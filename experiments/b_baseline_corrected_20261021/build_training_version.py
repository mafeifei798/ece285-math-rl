"""Construct the isolated live training version on explicit future invocation.

There is intentionally no launch CLI, .train(), optimizer.step(), queue, or
background job in this module. The current authorized action is offline replay.
"""
import dataclasses
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build(run_dir, *, root=ROOT):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig
    from trainer import trainer_class
    from rewards import RewardLedger
    for folder in ["training", "experiments/b_qwen7b_6m_20261021"]:
        sys.path.insert(0, str(root / folder))
    from frozen_prm import FrozenQwenPRM
    from train_baseline import BudgetCallback
    import numpy as np
    import random
    import time

    run = Path(run_dir).resolve()
    if not run.is_relative_to(root / "runs") or not run.name.startswith("Bbaseline_corrected_"):
        raise ValueError("Use a new runs/Bbaseline_corrected_* directory; never overwrite original A/B")
    if run.exists():
        raise ValueError("Run directory exists; inspect it before an explicit future resume")
    config = json.loads((root / "training/baseline_config.json").read_text())
    assets = json.loads((root / "receipts/assets.json").read_text())
    expected = json.loads((HERE / "release.json").read_text())
    import hashlib
    for name, digest in expected["source_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Dependency changed: " + name)
    for name, digest in expected["trusted_runtime_inputs"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Trusted runtime input changed: " + name)
    saved = torch.load(root / "runs/A_seed17/checkpoint-280/training_args.bin",
                       map_location="cpu", weights_only=False)
    params = {f.name: getattr(saved, f.name) for f in dataclasses.fields(saved) if f.init}
    params.update(output_dir=str(run), logging_dir=str(run / "tensorboard"), generation_batch_size=None)
    args = GRPOConfig(**params)
    if torch.cuda.device_count() != 1 or args.generation_batch_size != 32:
        raise RuntimeError("Future runner must expose one GPU and preserve batch32")
    cls = trainer_class()  # Source pin checked before model load.
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(assets["model_path"], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation=config["attention_implementation"])
    model.config.use_cache = False
    if assets["model_revision"] != "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2":
        raise RuntimeError("Actor revision changed")
    unit_path = root / "execution/reward_repair_v3_20261021/unit_metadata.json"
    if not unit_path.exists():
        unit_path = root / "audits/reward_repair_v3_20261021/unit_metadata.json"
    if hashlib.sha256(unit_path.read_bytes()).hexdigest() != expected["replay_input_hashes"]["unit_metadata.json"]:
        raise RuntimeError("Frozen verifier unit metadata changed")
    run.mkdir(parents=True)
    ledger = RewardLedger(None, json.loads(unit_path.read_text()))
    rows = [json.loads(s) for s in (root / "data/prepared/train.jsonl").open() if s.strip()]
    owner = {}
    lora = LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"], bias="none", task_type="CAUSAL_LM")
    instance = cls(model=model, processing_class=tokenizer, args=args, train_dataset=Dataset.from_list(rows),
        peft_config=lora, reward_funcs=ledger.callback(tokenizer.eos_token_id, owner),
        baseline_ledger=ledger, run_dir=run, budget=6000000)
    owner["trainer"] = instance
    # Preserve the original actor/vLLM/adapter initialization order and RNG.
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        scorer = FrozenQwenPRM(root / "models/qwen25_math_prm_7b", receipt_dir=run, microbatch_size=4)
    random.setstate(py_state)
    np.random.set_state(np_state)
    ledger.scorer.frozen_scorer = scorer
    instance.prm = scorer
    instance.start_clock = time.monotonic()
    instance.add_callback(BudgetCallback(instance, 12 * 3600))
    (run / "independent_version.json").write_text(json.dumps({
        "version": HERE.name, "source_hashes": expected["source_hashes"],
        "advantage_rule": "mixed_reward_original_group_std", "started_training": False,
        "historical_AB_untouched": True}, indent=2) + "\n")
    return instance
