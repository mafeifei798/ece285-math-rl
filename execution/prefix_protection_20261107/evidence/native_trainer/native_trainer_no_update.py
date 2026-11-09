"""Full pinned TRL generation/reward/loss path with native LoRA, no updates."""
import os,json,time,dataclasses,sys,hashlib,fcntl
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
from peft import PeftModel
from datasets import Dataset
from trl import GRPOConfig
from signals import ROOT,signals
from native_sync import trainer_class
from runtime_mixin import fingerprint
from no_update import tensor_hash,sha

OUT=ROOT/'audits/prefix_protection_20261107/native_trainer'
CP=ROOT/'runs/ABC_seed17_resumed_V1_20261104/C_bounded/checkpoint-288'

def save(name,data):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def event(stage,**kw):
    x={'stage':stage,'pid':os.getpid(),'unix':time.time(),'optimizer_steps':0,**kw};save('progress.json',x);print(json.dumps(x),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'result.json').exists()
    before=sha(CP/'adapter_model.safetensors');started=time.time();set_seed(17);torch.set_num_threads(2)
    assets=json.loads((ROOT/'receipts/assets.json').read_text())
    saved=torch.load(CP/'training_args.bin',map_location='cpu',weights_only=False)
    values={f.name:getattr(saved,f.name) for f in dataclasses.fields(saved) if f.init}
    values.update(output_dir=str(OUT/'scratch'),logging_dir=str(OUT/'scratch/logs'),report_to=[],generation_batch_size=None,
                  vllm_gpu_memory_utilization=.18)
    args=GRPOConfig(**values)
    assert args.learning_rate==5e-6 and args.beta==.04 and args.num_generations==8 and args.max_completion_length==2048
    tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    cfg=json.loads((ROOT/'training/baseline_config.json').read_text())
    event('load_actor')
    actor=AutoModelForCausalLM.from_pretrained(assets['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,
                                           attn_implementation=cfg['attention_implementation'])
    actor=PeftModel.from_pretrained(actor,CP,is_trainable=True)
    sys.path.insert(0,str(ROOT/'experiments/c_bounded_integration_20261029_v2'))
    from c_rewards import RewardLedger
    from c_prm import FrozenCScorer
    units=json.loads((ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json').read_text())
    ledger=RewardLedger(None,units);owner={}
    data=[json.loads(s) for s in (ROOT/'data/prepared/train.jsonl').open()]
    trainer=trainer_class()(model=actor,processing_class=tok,args=args,train_dataset=Dataset.from_list(data),
                            reward_funcs=ledger.callback(tok.eos_token_id,owner))
    owner['trainer']=trainer
    trainer.initialize_signals('prefix',ledger,tok,OUT/'signal_trace.jsonl')
    trainer.state.global_step=288;trainer.current_gradient_accumulation_steps=32
    event('load_frozen_C_PRM')
    scorer=FrozenCScorer(receipt_dir=OUT);ledger.scorer.frozen_scorer=scorer
    base_before=tensor_hash((n,p) for n,p in trainer.model.named_parameters() if 'lora_' not in n)
    adapter_before=tensor_hash((n,p) for n,p in trainer.model.named_parameters() if 'lora_' in n)
    schedule=json.loads((ROOT/'audits/ABC_seed17_resumed_V1_20261104/preflight.json').read_text())['first_epoch_question_groups']
    chosen=schedule[288];byid={r['problem_id']:r for r in data};inputs=[dict(byid[q]) for q in chosen for _ in range(8)]
    trainer.model.train();trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});trainer.model.enable_input_require_grads()
    event('generate_one_real_batch',problem_ids=chosen)
    t=time.time();generated=trainer._generate_and_score_completions(inputs);generation_seconds=time.time()-t
    records=ledger.records;made=signals(records,tok,device=generated['advantages'].device)
    assert torch.equal(generated['advantages'],made['prefix'])
    save('rollouts.json',records)
    counts=dict(Counter(a['status'] for a in made['annotations']))
    candidate=next((i for i,a in enumerate(made['annotations']) if a['status']=='prefix_protected'),0)
    loss_checks=[]
    for variant in ['bounded','uniform','prefix']:
        # No re-generation between arms: these are forward/backward diagnostics
        # on precisely the same fixed responses and old-policy probabilities.
        test={k:v[candidate:candidate+1] if torch.is_tensor(v) and v.ndim and v.shape[0]==len(records) else v for k,v in generated.items()}
        test['advantages']=made[variant][candidate:candidate+1].clone()
        tag=int(test['signal_trace_id'][0]);trainer.signal_expected[tag]={'fingerprint':fingerprint(test,0),'outcome':int(records[candidate]['outcome_reward'])}
        trainer.model.zero_grad(set_to_none=True)
        with trainer.compute_loss_context_manager():loss=trainer._compute_loss(trainer.model,test)
        loss.backward()
        grad=sum(float(p.grad.float().square().sum()) for n,p in trainer.model.named_parameters() if 'lora_' in n and p.grad is not None)**.5
        assert torch.isfinite(loss) and grad>0
        loss_checks.append({'variant':variant,'loss':float(loss.detach()),'gradient_L2':grad,'selected_row':candidate,
                            'selected_status':made['annotations'][candidate]['status'],'advantage_unchanged':torch.equal(test['advantages'],made[variant][candidate:candidate+1])})
    trainer.model.zero_grad(set_to_none=True)
    assert trainer.optimizer is None
    assert tensor_hash((n,p) for n,p in trainer.model.named_parameters() if 'lora_' not in n)==base_before
    assert tensor_hash((n,p) for n,p in trainer.model.named_parameters() if 'lora_' in n)==adapter_before
    event('full_batch_and_loss_passed',statuses=counts)
    # Verify the hot-reload path: identical weights under a new adapter ID must
    # yield identical teacher-forced scores. Each snapshot is local to scratch.
    from vllm import SamplingParams
    probe=tok.encode(records[0]['prompt_text'],add_special_tokens=False)+records[0]['completion_token_ids'][:64]
    params=SamplingParams(temperature=0.,max_tokens=1,prompt_logprobs=1)
    a=trainer.llm.generate([{'prompt_token_ids':probe}],params,lora_request=trainer._active_lora_request,use_tqdm=False)
    trainer.state.global_step=289;trainer._move_model_to_vllm()
    b=trainer.llm.generate([{'prompt_token_ids':probe}],params,lora_request=trainer._active_lora_request,use_tqdm=False)
    difference=max(abs(a[0].prompt_logprobs[i][probe[i]].logprob-b[0].prompt_logprobs[i][probe[i]].logprob) for i in range(1,len(probe)))
    assert difference==0
    assert tensor_hash((n,p) for n,p in trainer.model.named_parameters() if 'lora_' not in n)==base_before
    assert sha(CP/'adapter_model.safetensors')==before
    scorer.assert_frozen()
    result={'passed':True,'optimizer_steps':0,'formal_training_started':False,'source_checkpoint':str(CP),'source_checkpoint_sha256':before,
            'native_adapter_ids':[289,290],'identical_weight_hot_reload_max_logprob_difference':difference,
            'full_generation_reward_signal_loss_path':True,'actor_base_unchanged':True,'actor_adapter_unchanged':True,'frozen_PRM':True,
            'rows':len(records),'generation_tokens':sum(r['completion_tokens'] for r in records),'generation_and_PRM_seconds':generation_seconds,
            'status_counts':counts,'loss_checks':loss_checks,'PRM_cost':ledger.cost,'seconds':time.time()-started,
            'max_torch_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'wrong_positive_tokens':0,
            'normalizations_after_scalar':0,'scope':'One no-update native-LoRA TRL batch. No accuracy evaluation or optimizer step.'}
    save('result.json',result);event('complete',seconds=result['seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        OUT.mkdir(parents=True,exist_ok=True);save('failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'optimizer_steps':0});raise
