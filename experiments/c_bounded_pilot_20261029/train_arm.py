"""Authorized actual two-update acceptance, then checkpoint continuation to 1M."""
import argparse,fcntl,os,time,traceback,shutil,math
from pathlib import Path
import torch
from transformers import TrainerCallback
from pilot_common import *
from runtime import build,protect,parameter_hash

class Acceptance(TrainerCallback):
    def __init__(self,t,phase):self.t=t;self.phase=phase;self.start=time.monotonic();self.stop_reason=None
    def on_train_begin(self,args,state,control,**kw):
        t=self.t
        assert t.optimizer is not None and t.lr_scheduler is not None
        actor_ids={id(p) for p in t.model.parameters() if p.requires_grad}
        opt_ids={id(p) for g in t.optimizer.param_groups for p in g['params']}
        assert actor_ids==opt_ids and actor_ids
        if self.phase=='continue':
            assert state.global_step==2 and t.resume_receipt['source_step']==2
            assert all(t.resume_receipt.get(k) for k in ['adapter_exact','optimizer_exact','scheduler_exact'])
        atomic(t.run/(self.phase+'_optimizer_acceptance.json'),dict(optimizer_constructed=True,
            optimizer_trainable_parameters=sum(p.numel() for p in t.model.parameters() if p.requires_grad),
            PRM_in_optimizer=False,global_step=state.global_step,resume=t.resume_receipt))
    def on_pre_optimizer_step(self,args,state,control,**kw):
        t=self.t;grads=[p.grad for p in t.model.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert len(t.used_tags)==32 and t.used_tags==set(t.expected),'Not every sampled row reached loss exactly once'
        if t.prm:t.prm.assert_frozen()
    def on_step_end(self,args,state,control,**kw):
        t=self.t;a=t.accounting
        assert a['loss_rows']==a['generated_completions']==state.global_step*32
        assert state.global_step==t.last_generation['step_before_update']+1
        if t.arm!='C_mix':assert a['loss_wrong_positive']==0
        if self.phase=='continue' and state.global_step==3:
            assert t.resume_receipt.get('torch_cpu_rng_exact') and t.resume_receipt.get('torch_cuda_rng_exact')
            assert t.last_generation['step_before_update']==2
            atomic(t.run/'resume_acceptance.json',dict(passed=True,**t.resume_receipt,
                continued_optimizer_step=3,prompt_order_checked=True,loss_input_preserved=True,
                actor_updated_since_smoke=adapter_digest(t.model)!=t.smoke_adapter_digest,
                vllm_RNG_limitation=read(HERE/'protocol.json')['resume_limitation']))
            assert read(t.run/'resume_acceptance.json')['actor_updated_since_smoke']
        elapsed=time.monotonic()-self.start
        if self.phase=='smoke' and state.global_step>=2:self.stop_reason='two_real_updates_complete'
        elif a['generated_completion_tokens']>=1000000:self.stop_reason='token_budget_reached'
        elif elapsed>=18*3600:self.stop_reason='wall_limit_incomplete'
        if self.stop_reason:control.should_training_stop=True;control.should_save=True
        progress=dict(arm=t.arm,phase=self.phase,global_step=state.global_step,**a,elapsed_seconds=elapsed,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            latest_generation=t.last_generation,stop_reason=self.stop_reason,unix=time.time(),pid=os.getpid())
        atomic(t.run/'progress.json',progress);append(t.run/'update_audits.jsonl',progress)
        print(json.dumps(dict(event='optimizer_update',arm=t.arm,step=state.global_step,tokens=a['generated_completion_tokens'],
            wrong_positive=a['loss_wrong_positive'],stop=self.stop_reason)),flush=True)
    def on_save(self,args,state,control,**kw):
        t=self.t;p=t.run/f'checkpoint-{state.global_step}'
        required=['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
        assert all((p/n).is_file() for n in required)
        atomic(p/'completion_accounting.json',t.accounting)
        atomic(p/'pilot_checkpoint.json',dict(arm=t.arm,global_step=state.global_step,actor_adapter_digest=adapter_digest(t.model),
            accounting_sha256=sha(p/'completion_accounting.json'),source_release_sha256=verify(),
            complete_generation_boundary=t.accounting['loss_rows']==t.accounting['generated_completions'],
            files_sha256={n:sha(p/n) for n in required}))
        atomic(t.run/'last_checkpoint.json',dict(path=str(p),global_step=state.global_step,checkpoint_contract_sha256=sha(p/'pilot_checkpoint.json')))
    def on_log(self,args,state,control,logs=None,**kw):
        for k,v in (logs or {}).items():
            if isinstance(v,(int,float)):assert math.isfinite(v),('Nonfinite training log',k)
        append(self.t.run/'metrics.jsonl',dict(step=state.global_step,**(logs or {})))

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--phase',choices=['smoke','continue'],required=True)
    a=p.parse_args();guard();release=verify();assert read(AUDIT/'preflight.json')['passed']
    run=PAIR/a.arm;run.mkdir(parents=True,exist_ok=True)
    lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.phase=='smoke':
        assert not (run/'run_config.json').exists(),'Existing attempt must be inspected, never restart from base'
        accounting=None;checkpoint=None
    else:
        assert read(PAIR/'smoke_gate.json')['passed']
        assert not (run/'resume_build.json').exists(),'Continuation already attempted; inspect checkpoint, do not redispatch'
        smoke=read(run/'smoke_result.json');assert smoke['passed'] and smoke['global_step']==2
        checkpoint=Path(smoke['checkpoint']);accounting=read(checkpoint/'completion_accounting.json')
        assert accounting['loss_rows']==64
        for n,h in read(checkpoint/'pilot_checkpoint.json')['files_sha256'].items():assert sha(checkpoint/n)==h
    atomic(run/'status.json',dict(status='initializing',phase=a.phase,pid=os.getpid(),arm=a.arm,unix=time.time()))
    started=time.time()
    try:
        before=protect();atomic(run/(a.phase+'_protected_before.json'),before)
        t,tok=build(a.arm,run,accounting)
        initial=adapter_digest(t.model)
        prm_before=parameter_hash(t.prm.model) if t.prm else None
        if checkpoint:t.smoke_adapter_digest=read(checkpoint/'pilot_checkpoint.json')['actor_adapter_digest']
        cb=Acceptance(t,a.phase);t.add_callback(cb)
        atomic(run/'status.json',dict(status='training',phase=a.phase,pid=os.getpid(),arm=a.arm,unix=time.time()))
        result=t.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        assert adapter_digest(t.model)!=initial
        if t.prm:
            t.prm.assert_frozen();assert parameter_hash(t.prm.model)==prm_before
        assert protect()==before
        last=Path(read(run/'last_checkpoint.json')['path']);assert int(last.name.split('-')[-1])==t.state.global_step
        tokens=t.accounting['generated_completion_tokens']
        if a.phase=='smoke':
            assert t.state.global_step==2 and tokens<1000000
            archive=run/'smoke_checkpoint';shutil.copytree(last,archive)
            out=dict(passed=True,arm=a.arm,phase=a.phase,global_step=2,accounting=t.accounting,checkpoint=str(archive),
                adapter_changed=True,PRM_frozen=True,originals_preserved=True,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,source_release_sha256=release,seconds=time.time()-started)
            atomic(run/'smoke_result.json',out)
        else:
            assert cb.stop_reason=='token_budget_reached' and 1000000<=tokens<1065536
            assert read(run/'resume_acceptance.json')['passed']
            t.save_model(str(run/'final_adapter'));tok.save_pretrained(run/'final_adapter');t.save_state()
            out=dict(passed=True,arm=a.arm,phase=a.phase,global_step=t.state.global_step,accounting=t.accounting,
                target_tokens=1000000,budget_overshoot=tokens-1000000,checkpoint=str(last),adapter=str(run/'final_adapter'),
                adapter_sha256=sha(run/'final_adapter/adapter_model.safetensors'),PRM_frozen=True,originals_preserved=True,
                source_release_sha256=release,seconds=time.time()-started,metrics=result.metrics)
            atomic(run/'final_result.json',out)
        atomic(run/'status.json',dict(status='smoke_complete' if a.phase=='smoke' else 'complete',pid=os.getpid(),unix=time.time(),**out))
    except Exception as e:
        atomic(run/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),unix=time.time(),phase=a.phase))
        atomic(run/'status.json',dict(status='failed',error=str(e),phase=a.phase,pid=os.getpid(),unix=time.time()));raise

if __name__=='__main__':main()
