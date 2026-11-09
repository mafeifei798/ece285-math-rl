from common import *
import argparse,fcntl,shutil,threading,subprocess,traceback,math
from build import build,legacy,VALIDATED
from transformers import TrainerCallback
sys.path.insert(0,str(VALIDATED))
from no_update import tensor_hash
import torch

class Acceptance(TrainerCallback):
    def __init__(self,t,cp,start,target):self.t=t;self.cp=cp;self.start=start;self.target=target;self.started=time.monotonic();self.reason=None
    def on_train_begin(self,args,state,control,**kw):
        t=self.t;assert state.global_step==self.start
        assert all(t.resume_receipt.get(k) for k in ['adapter_exact','optimizer_exact','scheduler_exact'])
        trainable={id(p) for p in t.model.parameters() if p.requires_grad}
        assert trainable=={id(p) for g in t.optimizer.param_groups for p in g['params']} and trainable
        t.base_hash=tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n)
        atomic(t.run/f'optimizer_acceptance_{self.target}.json',{'passed':True,'source':str(self.cp),**t.resume_receipt})
    def on_pre_optimizer_step(self,args,state,control,**kw):
        t=self.t;assert len(t.used_tags)==32 and t.used_tags==set(t.expected)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in t.model.parameters() if p.requires_grad)
        if t.prm:t.prm.assert_frozen()
    def on_step_end(self,args,state,control,**kw):
        t=self.t;a=t.accounting
        assert a['loss_rows']==a['generated_completions']==state.global_step*32
        assert a['loss_wrong_positive']==0 and state.global_step==t.last_generation['step_before_update']+1
        if state.global_step==self.start+1:
            assert t.resume_receipt.get('torch_cpu_rng_exact') and t.resume_receipt.get('torch_cuda_rng_exact')
            assert adapter_digest(t.model)!=read(self.cp/'pilot_checkpoint.json')['actor_adapter_digest']
            atomic(t.run/f'first_update_acceptance_{self.target}.json',{'passed':True,**t.resume_receipt,
                'first_optimizer_step':state.global_step,'question_order_checked':True,'advantage_fingerprint_at_loss':True,
                'vllm_sampling_RNG_recreated_not_bitwise_restored':True})
        if state.global_step>=self.target:self.reason='common_update_target';control.should_training_stop=True;control.should_save=True
        elif time.monotonic()-self.started>=18*3600:self.reason='wall_limit_incomplete';control.should_training_stop=True;control.should_save=True
        v={'arm':t.arm,'global_step':state.global_step,'stage_target':self.target,'stage_start':self.start,
           'stage_new_updates':state.global_step-self.start,'target_total_updates':576,'accounting':a,
           'seconds':time.monotonic()-self.started,'unix':time.time(),'pid':os.getpid(),'latest_generation':t.last_generation}
        atomic(t.run/'progress.json',v);append(t.run/'update_audits.jsonl',v)
        print(json.dumps({'event':'update','arm':t.arm,'step':state.global_step,'target':self.target,'tokens':a['generated_completion_tokens']}),flush=True)
    def on_save(self,args,state,control,**kw):
        t=self.t;p=t.run/f'checkpoint-{state.global_step}';names=['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
        assert all((p/n).exists() for n in names)
        atomic(p/'completion_accounting.json',t.accounting)
        atomic(p/'pilot_checkpoint.json',{'arm':t.arm,'global_step':state.global_step,'actor_adapter_digest':adapter_digest(t.model),
            'source_release_sha256':verify(),'complete_generation_boundary':t.accounting['loss_rows']==t.accounting['generated_completions'],
            'accounting_sha256':sha(p/'completion_accounting.json'),'files_sha256':{n:sha(p/n) for n in names}})
        atomic(t.run/'last_checkpoint.json',{'path':str(p),'global_step':state.global_step})
    def on_log(self,args,state,control,logs=None,**kw):
        for v in (logs or {}).values():
            if isinstance(v,(int,float)):assert math.isfinite(v)
        append(self.t.run/'metrics.jsonl',{'step':state.global_step,**(logs or {})})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--target',type=int,choices=[432,576],required=True);a=ap.parse_args()
    guard();verify();source_unchanged();assert read(AUDIT/'no_update'/a.arm/'result.json')['passed']
    run=RUN/a.arm;run.mkdir(exist_ok=True);lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    intent=run/f'stage_{a.target}_intent.json';assert not intent.exists(),'Inspect prior attempt; no automatic retry or resampling'
    start=288 if a.target==432 else 432
    cp=SOURCE/a.arm/'checkpoint-288' if start==288 else run/'milestones/checkpoint-432'
    contract=read(cp/'pilot_checkpoint.json');assert contract['global_step']==start
    for n,h in contract['files_sha256'].items():assert sha(cp/n)==h
    atomic(intent,{'pid':os.getpid(),'unix':time.time(),'source':str(cp),'start_step':start,'target':a.target})
    begun=time.time();stop=threading.Event()
    def resources():
        import psutil
        while not stop.is_set():
            try:
                owned={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)}
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
                append(run/'resources.jsonl',{'unix':time.time(),'stage_target':a.target,'own_MiB':sum(int(x.split(',')[1]) for x in raw.splitlines() if int(x.split(',')[0]) in owned)})
            except Exception as e:append(run/'resources.jsonl',{'monitor_error':str(e),'unix':time.time()})
            stop.wait(20)
    thread=threading.Thread(target=resources,daemon=True);thread.start()
    try:
        t,tok=build(a.arm,run,cp,a.target);prm_hash=legacy.parameter_hash(t.prm.model) if t.prm else None
        check=Acceptance(t,cp,start,a.target);t.add_callback(check);metrics=t.train(resume_from_checkpoint=str(cp))
        assert check.reason=='common_update_target' and t.state.global_step==a.target
        assert tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n)==t.base_hash
        if t.prm:t.prm.assert_frozen();assert legacy.parameter_hash(t.prm.model)==prm_hash
        last=Path(read(run/'last_checkpoint.json')['path']);assert read(last/'pilot_checkpoint.json')['global_step']==a.target
        milestone=run/f'milestones/checkpoint-{a.target}';milestone.parent.mkdir(exist_ok=True);shutil.copytree(last,milestone)
        verify();source_unchanged();protected_unchanged()
        atomic(run/f'stage_{a.target}_result.json',{'passed':True,'arm':a.arm,'start_step':start,'global_step':a.target,
            'accounting':t.accounting,'checkpoint':str(milestone),'adapter_sha256':sha(milestone/'adapter_model.safetensors'),
            'actor_base_unchanged':True,'PRM_frozen':True,'seconds':time.time()-begun,'metrics':metrics.metrics})
    except Exception as e:atomic(run/f'failure_{a.target}.json',{'error':str(e),'traceback':traceback.format_exc(),'no_automatic_retry':True});raise
    finally:stop.set();thread.join(timeout=10)

if __name__=='__main__':main()
