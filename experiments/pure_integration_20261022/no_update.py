"""One bounded real-generation batch; both PURE losses backward, zero updates.

There is deliberately no optimizer construction, optimizer.step, checkpoint
save, resumable training loop or formal training CLI in this entry point.
"""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import threading
import time
import traceback
import torch
from common import ROOT, HERE, read, read_jsonl, write_jsonl, atomic, sha, verify_release, frozen_verifier
from trainer import PureIntegration, parameter_hash, adapter_hash
from alignment import align_actor

class MemoryMonitor:
    def __init__(self, run):
        from vllm.third_party import pynvml
        self.nv=pynvml; pynvml.nvmlInit()
        # Physical index is recorded by launcher; CUDA logical index is always 0.
        self.handle=pynvml.nvmlDeviceGetHandleByIndex(int(os.environ['PURE_PHYSICAL_GPU']))
        self.pid=os.getpid(); self.run=run; self.stop=threading.Event()
        self.samples=[]; self.stage='starting'; self.peaks={}
        self.thread=threading.Thread(target=self.poll,daemon=True);self.thread.start()
    def poll(self):
        while not self.stop.is_set():
            try:
                procs=self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
                own=sum(p.usedGpuMemory for p in procs if p.pid==self.pid)
                total=self.nv.nvmlDeviceGetMemoryInfo(self.handle).used
                stat=self.peaks.setdefault(self.stage,dict(own_process_bytes=0,device_used_bytes=0,samples=0))
                stat['own_process_bytes']=max(stat['own_process_bytes'],own)
                stat['device_used_bytes']=max(stat['device_used_bytes'],total);stat['samples']+=1
                self.samples.append(dict(time=time.time(),stage=self.stage,own_process_bytes=own,
                    device_used_bytes=total,processes=[dict(pid=p.pid,bytes=p.usedGpuMemory) for p in procs]))
            except Exception as exc:
                self.samples.append(dict(time=time.time(),error=str(exc)))
            self.stop.wait(.5)
    def close(self):
        self.stop.set();self.thread.join(timeout=2)
        write_jsonl(self.run/'nvml_samples.jsonl',self.samples)
        atomic(self.run/'nvml_peaks.json',self.peaks)

def protected_snapshot():
    paths=[]
    for folder in ['training','training_b','experiments/ab_corrected_6m_seed17_20261022',
                   'experiments/b_baseline_corrected_20261021','experiments/b_qwen7b_6m_20261021']:
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    inventory={};hashed={}
    for name in ['A_corrected_seed17_6M_20261022','B_corrected_seed17_6M_20261022',
                 'A_seed17','B_seed17_6M','B_Qwen7B_seed17_6M']:
        path=ROOT/'runs'/name
        if path.exists():
            for p in path.rglob('*'):
                if p.is_file():
                    st=p.stat();inventory[str(p.relative_to(ROOT))]=[st.st_size,st.st_mtime_ns]
                    if p.suffix in ['.json','.jsonl','.safetensors']:paths.append(p)
    for p in sorted(set(paths)):hashed[str(p.relative_to(ROOT))]=sha(p)
    return dict(sha256=hashed,inventory=inventory)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--i-understand-no-updates',action='store_true',required=True)
    p.add_argument('--reuse-exact-rollouts',type=Path)
    args=p.parse_args();run=args.run.resolve()
    assert run.parent == ROOT/'runs' and run.name.startswith('PURE_no_update_')
    assert not run.exists(), 'Run exists: inspect receipt/PID instead of duplicating it'
    release=verify_release()
    run.mkdir()
    lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(run/'status.json',dict(status='initializing',pid=os.getpid(),started=time.time(),optimizer_steps=0))
    monitor=None;stages={};started=time.time()
    def stage(name):
        @contextmanager
        def wrapped():
            monitor.stage=name;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            atomic(run/'status.json',dict(status='running',stage=name,pid=os.getpid(),time=time.time(),optimizer_steps=0))
            print('STAGE '+name,flush=True);t=time.perf_counter()
            yield
            torch.cuda.synchronize()
            stages[name]=dict(seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            atomic(run/'stages.json',stages)
        return wrapped()
    try:
        monitor=MemoryMonitor(run)
        before=protected_snapshot();atomic(run/'originals_before.json',before)
        assert sha(ROOT/'data/prepared/manifest.json')=='7efe2833ba0d0d0df322fbb96f0ee377003783aead450d6c09585efa24ca7459'
        assert sha(ROOT/'data/prepared/train.jsonl')=='91931fa7ee301c03ce6bb705c728c8f5f45eedecbb7a8ca17481fd06eca6270e'
        lookup={r['problem_id']:r for r in read_jsonl(ROOT/'data/prepared/train.jsonl')}
        # Frozen same first four IDs as the historical A/B initial generation,
        # determined before observing any new rollout or advantage.
        historical=read_jsonl(ROOT/'runs/B_corrected_seed17_6M_20261022/rollouts.jsonl')[:32]
        ids=[historical[i]['problem_id'] for i in range(0,32,8)]
        questions=[lookup[i] for i in ids]
        assert len(set(ids))==4
        atomic(run/'run_config.json',dict(mode='real_forward_backward_no_update',release_sha256=sha(HERE/'release.json'),
            questions=ids,group_size=8,batch=32,completion_cap=2048,return_width=2048,
            physical_gpu=os.environ['PURE_PHYSICAL_GPU'],cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            formal_training=False,optimizer_constructed=False,approved_plan=read(HERE/'approved_plan.json')))
        with stage('actor_and_vllm_load'):
            engine=PureIntegration(run,questions)
        with stage('prm_load'):
            engine.load_prm()
        with stage('parameter_hash_before'):
            actor_before=parameter_hash(engine.model);prm_before=parameter_hash(engine.prm.model)
        if args.reuse_exact_rollouts:
            source=args.reuse_exact_rollouts.resolve()
            assert source == ROOT/'runs/PURE_no_update_20261022_v2/generated_exact_ids.jsonl'
            previous=read(source.parent/'result.json')
            assert previous['passed'] and previous['optimizer_steps']==0
            with stage('reuse_saved_generation'):
                rows=read_jsonl(source)
                assert [r['problem_id'] for r in rows[::8]]==ids
                assert len(rows)==32
                generation={**previous['generation'],'reused_exact_ids':True,
                    'measurement_from':str(source.parent/'result.json'),'source_sha256':sha(source)}
                write_jsonl(run/'generated_exact_ids.jsonl',rows)
        else:
            with stage('generation'):
                rows,generation=engine.generate(questions)
                write_jsonl(run/'generated_exact_ids.jsonl',rows)
        units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
        verify=frozen_verifier()
        for r in rows:
            outcome=verify(r['completion'],r['gold'],r['terminated'],question=r['question'],
                           problem_id=r['problem_id'],unit_metadata=units)
            r['outcome']=outcome['reward'];r['outcome_details']=outcome
        with stage('native_prm_scoring'):
            scored,prm_cost=engine.prm.score([r['question'] for r in rows],[r['completion'] for r in rows])
        mappings=[]
        for r,s in zip(rows,scored):
            mappings.append(align_actor(engine.tokenizer,r['completion_ids'],r['completion'],s['spans'],
                empty_terminal_step=s['empty_terminal_step']))
        write_jsonl(run/'scored_alignment.jsonl',[dict(row=i,problem_id=r['problem_id'],outcome=r['outcome'],
            terminated=r['terminated'],native=s,alignment=m) for i,(r,s,m) in enumerate(zip(rows,scored,mappings))])
        # Actual native PRM forward on fixed edge inputs (not RL training data).
        edge_texts=['',' \r\n\t ','x = 2.\r\n\r\nTherefore x = 2.','Answer without a box.']
        with stage('native_edge_forwards'):
            edge_scores,edge_cost=engine.prm.score(['Solve x+1=3.']*len(edge_texts),edge_texts)
        atomic(run/'native_edge_forwards.json',dict(texts=edge_texts,scores=edge_scores,cost=edge_cost))
        with stage('old_and_reference_logps'):
            old,ref=engine.old_and_reference(rows)
        # Persist actual old/ref values. Initial zero LoRA implies zero KL, but
        # CPU tests separately exercise nonzero KL and clipped PPO ratios.
        outputs={};tensor_dump={};positions=None
        with stage('signals'):
            for arm in ['P_VR','P_PRM']:
                signals,positions=engine.prepare(arm,rows,scored,old,ref)
                outputs[arm]=signals
                tensor_dump[arm]={k:getattr(signals,k).detach().cpu() for k in
                    ['mask','task_rewards','kl','rewards','returns','advantages','old_logps']}
            permuted=[dict(r,outcome=1-r['outcome']) for r in rows]
            perm_prm,_=engine.prepare('P_PRM',permuted,scored,old,ref)
            perm_vr,_=engine.prepare('P_VR',permuted,scored,old,ref)
            assert torch.equal(perm_prm.advantages,outputs['P_PRM'].advantages)
            assert not torch.equal(perm_vr.advantages,outputs['P_VR'].advantages)
            del perm_prm,perm_vr
            torch.save(dict(signals=tensor_dump,reference_logps=ref.cpu()),run/'signals.pt')
        results={}
        for arm in ['P_VR','P_PRM']:
            with stage(arm+'_backward'):
                results[arm]=engine.backward_no_update(outputs[arm],rows)
                atomic(run/(arm+'_loss_audit.json'),results[arm])
                engine.prm.assert_frozen()
        with stage('parameter_hash_after'):
            actor_after=parameter_hash(engine.model);prm_after=parameter_hash(engine.prm.model)
            assert actor_before==actor_after and prm_before==prm_after
            assert adapter_hash(engine.model)==engine.initial_adapter
        after=protected_snapshot();assert before==after,'Original A/B files changed'
        atomic(run/'preservation.json',dict(passed=True,hashed_files=len(before['sha256']),
            inventoried_files=len(before['inventory']),actor_before=actor_before,actor_after=actor_after,
            prm_before=prm_before,prm_after=prm_after,actor_and_prm_parameters_unchanged=True))
        atomic(run/'result.json',dict(passed=True,status='completed_no_update',optimizer_steps=0,
            actor_and_prm_parameters_unchanged=True,original_AB_preserved=True,
            generation=generation,prm_cost=prm_cost,stages=stages,
            generation_tokens_per_second=generation['tokens']/generation['seconds'],
            prm_input_tokens_per_second=prm_cost['prm_input_tokens']/prm_cost['seconds'],
            total_seconds=time.time()-started,exact_token_alignment_rows=len(mappings),
            exact_mapped_steps=sum(m['native_steps'] for m in mappings),
            outcome_counts=dict(correct=sum(r['outcome'] for r in rows),total=len(rows)),
            truncated=sum(not r['terminated'] for r in rows),
            scored_all_responses=len(scored)==len(rows),outcome_independence_passed=True,
            arms={k:{a:b for a,b in v.items() if a!='loss_trace'} for k,v in results.items()},
            limitations=['One batch at original base/zero LoRA; not training convergence or accuracy.',
                'GPU shared with other processes; timings are observed shared-device measurements.',
                'No optimizer state allocated; memory is no-update working set, not certified training peak.',
                'P_VR measured while frozen PRM remains resident for matched validation; standalone P_VR uses less memory.',
                'No checkpoint-resume or multi-update lifecycle acceptance was attempted.']))
        atomic(run/'status.json',dict(status='completed_no_update',passed=True,pid=os.getpid(),optimizer_steps=0,finished=time.time()))
    except Exception as exc:
        atomic(run/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
        atomic(run/'status.json',dict(status='failed',pid=os.getpid(),error=str(exc),optimizer_steps=0))
        raise
    finally:
        if monitor is not None:
            monitor.close()

if __name__=='__main__':main()
