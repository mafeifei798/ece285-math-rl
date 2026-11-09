"""Paired length-reduction-only branches; same policy/optimizer/RNG source."""
from common import *
import sys,dataclasses,random,fcntl,traceback,threading,subprocess,math,argparse
from collections import Counter
sys.path.insert(0,str(VALIDATED))
import torch
import numpy as np
from native_sync import trainer_class
from signals import signals
from no_update import tensor_hash
from transformers import AutoModelForCausalLM,AutoTokenizer,TrainerCallback,set_seed
from peft import PeftModel
from datasets import Dataset
from trl import GRPOConfig
sys.path.insert(0,str(ROOT/'experiments/c_bounded_integration_20261029_v2'))
from c_rewards import RewardLedger
from c_prm import FrozenCScorer

class AuditMixin:
    def _generate_and_score_completions(self,inputs):
        step=self.state.global_step
        assert START<=step<TARGET
        assert not (self.run/f'raw_step_{step}.pt').exists(),'Uncommitted generation exists; do not resample'
        before=time.monotonic();output=super()._generate_and_score_completions(inputs)
        records=self.signal_ledger.records
        assert len(records)==32
        torch.save({k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in output.items()},self.run/f'raw_step_{step}.pt')
        for row in records:append(self.run/'rollouts.jsonl',row)
        actual_order=[r['problem_id'] for r in records[::8]]
        assert actual_order==self.schedule[step-START],('Question order mismatch',step,actual_order)
        if step==START and not self.is_no_update:
            signature=[{'problem_id':r['problem_id'],'token_ids':r['completion_token_ids'],'outcome':r['outcome_reward']} for r in records]
            reference=RUN/'common_first_generation.json'
            if reference.exists():
                ref=read(reference)
                assert [r['problem_id'] for r in ref]==[r['problem_id'] for r in signature]
                atomic(self.run/'initial_sampling_audit.json',{
                    'same_initial_checkpoint':True,'same_questions':True,'same_engine_seed':0,
                    'token_identical_responses':sum(x['token_ids']==y['token_ids'] for x,y in zip(ref,signature)),
                    'outcome_identical_responses':sum(x['outcome']==y['outcome'] for x,y in zip(ref,signature)),
                    'note':'Identical stochastic draws are descriptive, not an execution gate. Native LoRA/eager and all sampling settings are shared.'})
            else:
                assert self.experiment_arm=='grpo';atomic(reference,signature)
        made=signals(records,self.signal_tokenizer,device=output['advantages'].device)
        assert self.signal_variant=='bounded'
        assert torch.equal(output['advantages'],made['bounded'])
        self.used_tags=set();self.last_generation={'step_before_update':step,'question_ids':actual_order,
            'tokens':sum(r['completion_tokens'] for r in records),'rows':len(records),'seconds':time.monotonic()-before,
            'statuses':dict(Counter(a['status'] for a in made['annotations'])),
            'protected_tokens':sum(a['protected_tokens'] for a in made['annotations']),
            'correct':sum(r['outcome_reward'] for r in records),
            'empty':sum(not r['completion'].strip() for r in records),
            'short_wrong':sum(r['completion_tokens']<=32 and not r['outcome_reward'] for r in records),
            'truncated':sum(not r['terminated'] for r in records),'answer_statuses':dict(Counter(r['status'] for r in records)),
            'group_types':dict(Counter('all_wrong' if sum(r['outcome_reward'] for r in records[i:i+8])==0 else
                          'all_correct' if sum(r['outcome_reward'] for r in records[i:i+8])==8 else 'mixed' for i in range(0,32,8))),
            'PRM_cost':self.signal_ledger.cost,'wrong_positive_tokens':0,'post_scalar_normalizations':0,
            'arm':self.experiment_arm,'loss_type':self.loss_type}
        append(self.run/'generation_checks.jsonl',self.last_generation)
        return output

    def _compute_loss(self,model,inputs):
        tags=set(inputs['signal_trace_id'].tolist());assert not self.used_tags.intersection(tags)
        loss=super()._compute_loss(model,inputs)
        assert torch.isfinite(loss);self.used_tags.update(tags)
        self.loss_rows+=len(tags)
        append(self.run/'loss_reductions.jsonl',{
            'step':self.state.global_step,'arm':self.experiment_arm,'loss_type':self.loss_type,
            'effective_lengths':inputs['completion_mask'].sum(1).detach().cpu().tolist(),
            'constant_length':self.max_completion_length,'gradient_accumulation_steps':self.current_gradient_accumulation_steps,
            'post_scalar_normalizations':0,'loss':float(loss.detach())})
        return loss

    def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
        assert Path(resume_from_checkpoint)==CP
        super()._load_from_checkpoint(resume_from_checkpoint,model=model)
        assert adapter_digest(self.model)==read(CP/'pilot_checkpoint.json')['actor_adapter_digest']
        self.resume_receipt.update(adapter_exact=True,source_step=START)

    def _load_optimizer_and_scheduler(self,checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        assert Path(checkpoint)==CP
        assert tree_digest(self.optimizer.state_dict())==tree_digest(torch.load(CP/'optimizer.pt',map_location='cpu',weights_only=False))
        assert tree_digest(self.lr_scheduler.state_dict())==tree_digest(torch.load(CP/'scheduler.pt',map_location='cpu',weights_only=False))
        self.resume_receipt.update(optimizer_exact=True,scheduler_exact=True)

    def _load_rng_state(self,checkpoint):
        super()._load_rng_state(checkpoint)
        saved=torch.load(CP/'rng_state.pth',map_location='cpu',weights_only=False)
        assert torch.equal(torch.get_rng_state(),saved['cpu'])
        cuda=saved.get('cuda')
        if isinstance(cuda,list):assert all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),cuda))
        elif cuda is not None:assert torch.equal(torch.cuda.get_rng_state(),cuda)
        self.resume_receipt.update(torch_cpu_rng_exact=True,torch_cuda_rng_exact=True)

class Checks(TrainerCallback):
    def __init__(self,t):self.t=t;self.started=time.monotonic()
    def on_train_begin(self,args,state,control,**kw):
        t=self.t;assert state.global_step==START and all(t.resume_receipt.get(k) for k in ['adapter_exact','optimizer_exact','scheduler_exact'])
        trainable={id(p) for p in t.model.parameters() if p.requires_grad}
        optimizer={id(p) for g in t.optimizer.param_groups for p in g['params']}
        assert trainable and trainable==optimizer
        assert not trainable.intersection(id(p) for p in t.prm.model.parameters())
        t.base_hash=tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n)
        atomic(t.run/'resume_loaded.json',t.resume_receipt)
    def on_pre_optimizer_step(self,args,state,control,**kw):
        t=self.t;assert t.used_tags==set(t.signal_expected) and len(t.used_tags)==32
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in t.model.parameters() if p.requires_grad)
        t.prm.assert_frozen()
    def on_step_end(self,args,state,control,**kw):
        t=self.t;count=state.global_step-START
        assert t.loss_rows==count*32 and state.global_step==t.last_generation['step_before_update']+1
        if count==1:
            assert t.resume_receipt.get('torch_cpu_rng_exact') and t.resume_receipt.get('torch_cuda_rng_exact')
            assert adapter_digest(t.model)!=read(CP/'pilot_checkpoint.json')['actor_adapter_digest']
            atomic(t.run/'first_update_acceptance.json',{'passed':True,**t.resume_receipt,'correct_question_order':True,
                'advantage_exact_at_loss':True,'actual_optimizer_update':True,'vllm_engine_seed':0,'no_historical_vllm_rng_exact_claim':True})
        if state.global_step==TARGET:control.should_training_stop=True;control.should_save=True
        assert state.global_step<=TARGET
        prior=read(t.run/'progress.json') if (t.run/'progress.json').exists() else {}
        out={'arm':t.experiment_arm,'global_step':state.global_step,'additional_updates':count,'target_step':TARGET,
            'generated_tokens':prior.get('generated_tokens',0)+t.last_generation['tokens'],
            'loss_rows':t.loss_rows,'elapsed_seconds':time.monotonic()-self.started,'pid':os.getpid(),'unix':time.time(),
            'latest_generation':t.last_generation,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
        atomic(t.run/'progress.json',out);append(t.run/'update_audits.jsonl',out)
        print(json.dumps({'event':'update','arm':t.experiment_arm,'new_step':count,'target':48,'tokens':out['generated_tokens']}),flush=True)
    def on_save(self,args,state,control,**kw):
        t=self.t;p=t.run/f'checkpoint-{state.global_step}'
        required=['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
        assert all((p/n).exists() for n in required)
        atomic(p/'short_pilot_checkpoint.json',{'arm':t.experiment_arm,'global_step':state.global_step,'loss_rows':t.loss_rows,
            'files_sha256':{n:sha(p/n) for n in required},'source_release_sha256':verify()})
        atomic(t.run/'last_checkpoint.json',{'path':str(p),'global_step':state.global_step})
    def on_log(self,args,state,control,logs=None,**kw):
        for v in (logs or {}).values():
            if isinstance(v,(int,float)):assert math.isfinite(v)
        append(self.t.run/'metrics.jsonl',{'step':state.global_step,**(logs or {})})

def build(arm,run):
    assert arm in ARMS
    set_seed(17);torch.set_num_threads(2)
    saved=torch.load(CP/'training_args.bin',map_location='cpu',weights_only=False)
    values={f.name:getattr(saved,f.name) for f in dataclasses.fields(saved) if f.init}
    values.update(output_dir=str(run),logging_dir=str(run/'logs'),report_to=[],generation_batch_size=None,
                  vllm_gpu_memory_utilization=.18,save_steps=8,save_total_limit=2,
                  loss_type='grpo' if arm=='grpo' else 'dr_grpo')
    args=GRPOConfig(**values)
    assert args.seed==args.data_seed==17 and args.max_steps==20000 and args.generation_batch_size==32 and args.steps_per_generation==32
    assert args.learning_rate==5e-6 and args.beta==.04 and args.max_completion_length==2048 and args.num_generations==8
    assert args.per_device_train_batch_size==1 and args.gradient_accumulation_steps==32
    assert args.loss_type==('grpo' if arm=='grpo' else 'dr_grpo')
    assert args.scale_rewards=='group' and not args.mask_truncated_completions
    assets=read(ROOT/'receipts/assets.json');cfg=read(ROOT/'training/baseline_config.json')
    tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(assets['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation=cfg['attention_implementation'])
    model.config.use_cache=False;model=PeftModel.from_pretrained(model,CP,is_trainable=True)
    assert adapter_digest(model)==read(CP/'pilot_checkpoint.json')['actor_adapter_digest']
    ledger=RewardLedger(None,read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json'));owner={}
    class Trainer(AuditMixin,trainer_class()):pass
    t=Trainer(model=model,processing_class=tok,args=args,train_dataset=Dataset.from_list(rows(ROOT/'data/prepared/train.jsonl')),
              reward_funcs=ledger.callback(tok.eos_token_id,owner));owner['trainer']=t
    t.initialize_signals('bounded',ledger,tok,run/'signal_trace.jsonl')
    t.experiment_arm=arm;t.is_no_update=False
    t.run=run;t.resume_receipt={};t.loss_rows=0;t.used_tags=set();t.schedule=read(AUDIT/'preflight.json')['question_groups']
    py,ns=random.getstate(),np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):prm=FrozenCScorer(receipt_dir=run)
    random.setstate(py);np.random.set_state(ns);t.prm=prm;ledger.scorer.frozen_scorer=prm
    assert type(t.llm.llm_engine).__module__.startswith('vllm.v1.')
    atomic(run/'build.json',{'arm':arm,'settings':args.to_dict(),'source_checkpoint':str(CP),'release_sha256':verify(),
        'source_adapter_sha256':sha(CP/'adapter_model.safetensors'),
        'C_adapter_sha256':'4917be38966c8438f58e283eb06c7fa818fed2bb13b726152078c19af013c356','native_LoRA':True,'enforce_eager':True,'engine_seed':0})
    return t

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True);a=parser.parse_args()
    guard();verify();source_unchanged();assert read(AUDIT/'preflight.json')['passed']
    assert read(AUDIT/'no_update/result.json')['passed'],'Real-model zero-update gate required'
    run=RUN/a.arm;lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'intent.json').exists(),'No automatic restart of an existing attempt'
    atomic(run/'intent.json',{'pid':os.getpid(),'unix':time.time(),'source_checkpoint':str(CP),'start':START,'target':TARGET,'release_sha256':verify()})
    started=time.time();stop=threading.Event()
    def monitor():
        import psutil
        while not stop.is_set():
            try:
                owned={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)}
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
                mem=sum(int(s.split(',')[1]) for s in raw.splitlines() if int(s.split(',')[0]) in owned)
                append(run/'resources.jsonl',{'unix':time.time(),'own_total_MiB':mem})
            except Exception as exc:append(run/'resources.jsonl',{'monitor_error':str(exc),'unix':time.time()})
            stop.wait(15)
    threading.Thread(target=monitor,daemon=True).start()
    try:
        t=build(a.arm,run);prm=t.prm
        t.add_callback(Checks(t));result=t.train(resume_from_checkpoint=str(CP))
        assert t.state.global_step==TARGET and t.loss_rows==48*32 and read(run/'first_update_acceptance.json')['passed']
        assert tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n)==t.base_hash
        prm.assert_frozen();source_unchanged();verify()
        p=Path(read(run/'last_checkpoint.json')['path']);assert p.name=='checkpoint-336'
        atomic(run/'result.json',{'passed':True,'arm':a.arm,'global_step':TARGET,'additional_updates':48,'loss_rows':t.loss_rows,
            'checkpoint':str(p),'adapter_sha256':sha(p/'adapter_model.safetensors'),'source_checkpoint_unchanged':True,
            'actor_base_unchanged':True,'PRM_frozen':True,'seconds':time.time()-started,'progress':read(run/'progress.json'),'metrics':result.metrics})
    except Exception as exc:
        atomic(run/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'unix':time.time(),'automatic_retry':False});raise
    finally:stop.set()
if __name__=='__main__':main()
