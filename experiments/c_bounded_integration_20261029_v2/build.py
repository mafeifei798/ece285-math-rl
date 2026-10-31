"""Build the real independent trainer with formal-training entry points locked."""
import dataclasses,sys,time,random
from pathlib import Path
from audit_common import ROOT,HERE,read,rows,atomic,verify_release

def build_no_update(run,questions):
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import LoraConfig
    from trl import GRPOConfig
    from c_trainer import trainer_class
    from c_rewards import RewardLedger
    from c_prm import FrozenCScorer
    verify_release();cfg=read(ROOT/'training/baseline_config.json');assets=read(ROOT/'receipts/assets.json')
    saved=torch.load(ROOT/'runs/A_seed17/checkpoint-280/training_args.bin',map_location='cpu',weights_only=False)
    values={f.name:getattr(saved,f.name) for f in dataclasses.fields(saved) if f.init}
    values.update(output_dir=str(run),logging_dir=str(run/'tensorboard'),generation_batch_size=None,max_steps=1,
        report_to=[],save_strategy='no',logging_strategy='no',disable_tqdm=True,vllm_gpu_memory_utilization=.15)
    args=GRPOConfig(**values)
    assert args.generation_batch_size==32 and args.steps_per_generation==32 and args.num_generations==8
    assert args.max_completion_length==2048 and args.temperature==1. and args.beta==.04
    set_seed(17)
    tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(assets['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation=cfg['attention_implementation'])
    model.config.use_cache=False
    lora=LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],lora_dropout=0.,
        target_modules=cfg['lora_target_modules'],bias='none',task_type='CAUSAL_LM')
    ledger=RewardLedger(None,read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json'));owner={}
    cls=trainer_class()
    trainer=cls(model=model,processing_class=tok,args=args,train_dataset=Dataset.from_list(questions),peft_config=lora,
        reward_funcs=ledger.callback(tok.eos_token_id,owner),c_ledger=ledger)
    owner['trainer']=trainer
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainer.model.enable_input_require_grads();trainer.model.train()
    assert all(m.p==0 for m in trainer.model.modules() if isinstance(m,torch.nn.Dropout))
    py_state,np_state=random.getstate(),np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):scorer=FrozenCScorer(receipt_dir=run)
    random.setstate(py_state);np.random.set_state(np_state)
    ledger.scorer.frozen_scorer=scorer;trainer.prm=scorer
    trainer.current_gradient_accumulation_steps=32
    assert trainer.optimizer is None and trainer.lr_scheduler is None
    atomic(run/'independent_version.json',dict(version=HERE.name,source_release_sha256=verify_release(),
        formal_RL_authorized=False,optimizer_constructed=False,actor_revision=assets['model_revision'],
        C_adapter_sha256=read(ROOT/'runs/CD_PRM7B_pair_seed17_20261028/C/completion.json')['adapter_sha256'],
        audit_overrides={k:values[k] for k in ['max_steps','report_to','save_strategy','logging_strategy','vllm_gpu_memory_utilization']},
        overrides_reason='One no-update batch with lower vLLM cache footprint on shared GPU; reward/verifier/GRPO clipping/KL unchanged',
        sampling=dict(group_size=8,questions=len(questions),temperature=1.,cap=2048,seed=17)))
    return trainer,tok,ledger,scorer
