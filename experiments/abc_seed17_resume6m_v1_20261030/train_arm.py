"""Exact state continuation to common fixed milestones; no new reward/loss."""
from pilot_common import *
from runtime import build,parameter_hash
import argparse,fcntl,shutil,traceback,math,threading,subprocess
import torch
from transformers import TrainerCallback

class Acceptance(TrainerCallback):
    def __init__(self,t,start,target,source_digest):self.t=t;self.start=start;self.target=target;self.source_digest=source_digest;self.begun=time.monotonic();self.stop_reason=None
    def on_train_begin(self,args,state,control,**kw):
        t=self.t;assert state.global_step==self.start and t.resume_receipt['source_step']==self.start
        assert all(t.resume_receipt.get(k) for k in ['adapter_exact','optimizer_exact','scheduler_exact'])
        actor={id(p) for p in t.model.parameters() if p.requires_grad};opt={id(p) for g in t.optimizer.param_groups for p in g['params']}
        assert actor==opt and actor
        atomic(t.run/f'optimizer_acceptance_{self.target}.json',{'passed':True,'start_step':state.global_step,'trainable_parameters':sum(p.numel() for p in t.model.parameters() if p.requires_grad),'PRM_in_optimizer':False,'resume':t.resume_receipt})
    def on_pre_optimizer_step(self,args,state,control,**kw):
        t=self.t;grads=[p.grad for p in t.model.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert len(t.used_tags)==32 and t.used_tags==set(t.expected)
        if t.prm:t.prm.assert_frozen()
    def on_step_end(self,args,state,control,**kw):
        t=self.t;a=t.accounting;assert a['loss_rows']==a['generated_completions']==state.global_step*32
        assert state.global_step==t.last_generation['step_before_update']+1
        if t.arm!='C_mix':assert a['loss_wrong_positive']==0
        if state.global_step==self.start+1:
            assert t.resume_receipt.get('torch_cpu_rng_exact') and t.resume_receipt.get('torch_cuda_rng_exact')
            assert adapter_digest(t.model)!=self.source_digest
            atomic(t.run/f'resume_acceptance_{self.target}.json',{'passed':True,**t.resume_receipt,'first_new_optimizer_step':state.global_step,'prompt_order_checked':True,'loss_input_exact':True,'actor_updated':True,'vllm_sampling_RNG_not_bitwise_resumed':True})
        elapsed=time.monotonic()-self.begun
        if state.global_step>=self.target:self.stop_reason='common_update_target_reached'
        elif elapsed>=18*3600 or time.time()>float(os.environ.get('ABC_PIPELINE_DEADLINE','inf')):self.stop_reason='wall_limit_incomplete'
        if self.stop_reason:control.should_training_stop=True;control.should_save=True
        p=dict(arm=t.arm,stage_target=self.target,global_step=state.global_step,**a,elapsed_seconds=elapsed,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            latest_generation=t.last_generation,stop_reason=self.stop_reason,unix=time.time(),pid=os.getpid(),engine='V1')
        atomic(t.run/'progress.json',p);append(t.run/'update_audits.jsonl',p)
        print(json.dumps({'event':'optimizer_update','arm':t.arm,'step':state.global_step,'tokens':a['generated_completion_tokens'],'wrong_positive':a['loss_wrong_positive'],'target':self.target}),flush=True)
    def on_save(self,args,state,control,**kw):
        t=self.t;p=t.run/f'checkpoint-{state.global_step}';required=['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
        assert all((p/n).is_file() for n in required)
        atomic(p/'completion_accounting.json',t.accounting)
        atomic(p/'pilot_checkpoint.json',{'arm':t.arm,'global_step':state.global_step,'actor_adapter_digest':adapter_digest(t.model),'accounting_sha256':sha(p/'completion_accounting.json'),
            'source_release_sha256':verify(),'complete_generation_boundary':t.accounting['loss_rows']==t.accounting['generated_completions'],
            'files_sha256':{n:sha(p/n) for n in required}})
        atomic(t.run/'last_checkpoint.json',{'path':str(p),'global_step':state.global_step,'checkpoint_contract_sha256':sha(p/'pilot_checkpoint.json')})
    def on_log(self,args,state,control,logs=None,**kw):
        for k,v in (logs or {}).items():
            if isinstance(v,(float,int)):assert math.isfinite(v),(k,v)
        append(self.t.run/'metrics.jsonl',{'step':state.global_step,'stage_target':self.target,**(logs or {})})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True);parser.add_argument('--target',type=int,choices=[50,144,288],required=True);a=parser.parse_args()
    guard();release=verify();unchanged();assert read(AUDIT/'preflight.json')['passed']
    run=PAIR/a.arm;lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/f'stage_{a.target}_intent.json').exists(),'Existing attempt; inspect checkpoint and raw batches, never resample silently'
    start={50:48,144:50,288:144}[a.target];checkpoint=run/f'milestones/checkpoint-{start}';cp=read(checkpoint/'pilot_checkpoint.json')
    assert cp['global_step']==start
    for n,h in cp['files_sha256'].items():assert sha(checkpoint/n)==h
    accounting=read(checkpoint/'completion_accounting.json');assert accounting['loss_rows']==start*32
    if a.target!=50:assert read(PAIR/'acceptance_gate.json')['passed']
    atomic(run/f'stage_{a.target}_intent.json',{'unix':time.time(),'pid':os.getpid(),'source_checkpoint':str(checkpoint),'start_step':start,'target':a.target,'source_release_sha256':release})
    started=time.time();stop=threading.Event()
    def sampler():
        import psutil
        while not stop.is_set():
            try:
                owned={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)}
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
                owned_memory=[int(s.split(',')[1]) for s in raw.splitlines() if int(s.split(',')[0]) in owned]
                append(run/'resource_samples.jsonl',{'unix':time.time(),'target':a.target,'own_total_MiB':sum(owned_memory)})
            except Exception as exc:append(run/'resource_samples.jsonl',{'unix':time.time(),'monitor_error':str(exc)})
            stop.wait(20)
    monitor=threading.Thread(target=sampler,daemon=True);monitor.start()
    try:
        t,tok=build(a.arm,run,accounting,target_step=a.target)
        prm_before=parameter_hash(t.prm.model) if t.prm else None
        callback=Acceptance(t,start,a.target,cp['actor_adapter_digest']);t.add_callback(callback)
        result=t.train(resume_from_checkpoint=str(checkpoint))
        assert callback.stop_reason=='common_update_target_reached' and t.state.global_step==a.target
        assert read(run/f'resume_acceptance_{a.target}.json')['passed']
        if t.prm:t.prm.assert_frozen();assert parameter_hash(t.prm.model)==prm_before
        last=Path(read(run/'last_checkpoint.json')['path']);assert int(last.name.split('-')[-1])==a.target
        milestone=run/f'milestones/checkpoint-{a.target}';shutil.copytree(last,milestone)
        for n,h in read(milestone/'pilot_checkpoint.json')['files_sha256'].items():assert sha(milestone/n)==h
        assert adapter_digest(t.model)!=cp['actor_adapter_digest'];unchanged()
        out={'passed':True,'arm':a.arm,'start_step':start,'global_step':a.target,'accounting':t.accounting,'checkpoint':str(milestone),
            'adapter_sha256':sha(milestone/'adapter_model.safetensors'),'PRM_frozen':True,'engine':'V1','source_release_sha256':release,
            'seconds':time.time()-started,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'metrics':result.metrics}
        atomic(run/f'stage_{a.target}_result.json',out)
        if a.target==288:atomic(run/'final_result.json',out)
    except Exception as exc:
        atomic(run/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'unix':time.time(),'target':a.target,'no_automatic_retry':True});raise
    finally:stop.set();monitor.join(timeout=10)
if __name__=='__main__':main()
