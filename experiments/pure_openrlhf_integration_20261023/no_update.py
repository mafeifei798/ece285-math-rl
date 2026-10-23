"""One exact-token real-model OpenRLHF integration probe, zero optimizer steps."""
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
from engine import OpenRLHFIntegration, parameter_hash, adapter_hash
from signals import TENSOR_FIELDS

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
                   'experiments/b_baseline_corrected_20261021','experiments/b_qwen7b_6m_20261021',
                   'experiments/pure_integration_20261022','experiments/pure_official_prm_offline_20261022']:
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    inventory={};hashed={}
    for name in ['A_corrected_seed17_6M_20261022','B_corrected_seed17_6M_20261022',
                 'A_seed17','B_seed17_6M','B_Qwen7B_seed17_6M','PURE_no_update_20261022_v2','PURE_no_update_20261022_v3']:
        path=ROOT/'runs'/name
        if path.exists():
            for p in path.rglob('*'):
                if p.is_file():
                    st=p.stat();inventory[str(p.relative_to(ROOT))]=[st.st_size,st.st_mtime_ns]
                    if p.suffix in ['.json','.jsonl','.safetensors']:paths.append(p)
    for p in sorted(set(paths)):hashed[str(p.relative_to(ROOT))]=sha(p)
    return dict(sha256=hashed,inventory=inventory)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--authorized-no-update',action='store_true',required=True)
    args=parser.parse_args();run=args.run.resolve()
    assert run.parent==ROOT/'runs' and run.name.startswith('PURE_OpenRLHF_no_update_20261023_')
    assert not run.exists(),'Inspect existing status/receipt instead of repeating this probe'
    verify_release();run.mkdir()
    lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(run/'status.json',dict(status='initializing',pid=os.getpid(),optimizer_steps=0,started=time.time()))
    monitor=None;stages={};started=time.time()
    def stage(name):
        @contextmanager
        def context():
            monitor.stage=name;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            atomic(run/'status.json',dict(status='running',stage=name,pid=os.getpid(),optimizer_steps=0,time=time.time()))
            print('STAGE '+name,flush=True);t=time.perf_counter()
            yield
            torch.cuda.synchronize()
            stages[name]=dict(seconds=time.perf_counter()-t,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            atomic(run/'stages.json',stages)
        return context()
    try:
        monitor=MemoryMonitor(run)
        before=protected_snapshot();atomic(run/'originals_before.json',before)
        assert sha(ROOT/'data/prepared/manifest.json')=='7efe2833ba0d0d0df322fbb96f0ee377003783aead450d6c09585efa24ca7459'
        assert sha(ROOT/'data/prepared/train.jsonl')=='91931fa7ee301c03ce6bb705c728c8f5f45eedecbb7a8ca17481fd06eca6270e'
        lookup={r['problem_id']:r for r in read_jsonl(ROOT/'data/prepared/train.jsonl')}
        with (ROOT/'runs/B_corrected_seed17_6M_20261022/rollouts.jsonl').open() as f:
            import json
            historical=[json.loads(next(f)) for _ in range(32)]
        ids=[historical[i]['problem_id'] for i in range(0,32,8)]
        questions=[lookup[i] for i in ids];assert len(set(ids))==4
        atomic(run/'run_config.json',dict(mode='OpenRLHF_real_forward_backward_no_update',
            release_sha256=sha(HERE/'release.json'),protocol=read(HERE/'protocol.json'),
            physical_gpu=os.environ['PURE_PHYSICAL_GPU'],questions=ids,groups=4,responses=32,
            completion_cap=2048,seed=17,optimizer_constructed=False,
            original_verifier_retained=True,formal_training=False))
        with stage('actor_and_vllm_load'):
            engine=OpenRLHFIntegration(run,questions)
        with stage('tokenizer_checks'):
            from tokenizer_checks import run_tokenizer_checks
            token_tests=run_tokenizer_checks(engine.tokenizer,builder=engine.builder)
            atomic(run/'tokenizer_checks.json',token_tests)
            assert token_tests['passed']
        with stage('prm_load'):
            engine.load_prm()
        with stage('parameter_hash_before'):
            actor_before=parameter_hash(engine.model);prm_before=parameter_hash(engine.prm.model)
        with stage('generation'):
            rows,generation=engine.generate(questions)
            write_jsonl(run/'generated_exact_ids.jsonl',rows)
        verify=frozen_verifier();units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
        for row in rows:
            # Keep the archived verifier and real termination status. An EOS
            # inserted by official process_sequences never certifies completion.
            result=verify(row['emitted_completion'],row['gold'],row['original_terminated'],
                question=row['question'],problem_id=row['problem_id'],unit_metadata=units)
            row['outcome']=result['reward'];row['outcome_details']=result
        write_jsonl(run/'verified_rows.jsonl',rows)
        alignments=[r['alignment'] for r in rows]
        with stage('official_prm_scoring'):
            scored=engine.prm.score(alignments)
        assert len(scored)==32
        write_jsonl(run/'scored_alignment.jsonl',[dict(index=i,problem_id=r['problem_id'],outcome=r['outcome'],
            original_terminated=r['original_terminated'],alignment=r['alignment'],prm=s)
            for i,(r,s) in enumerate(zip(rows,scored))])
        edge_texts=['',' \r\n\t ','x = 2.\r\n\r\nTherefore x = 2.',
                    'x+1=3.\n\nx=2.\n\n','Thank you.','x=2.','No separator answer.']
        edge_prompt=engine.tokenizer.encode('Solve x+1=3.',add_special_tokens=False)
        edge_inputs=[]
        for i,text in enumerate(edge_texts):
            emitted=engine.tokenizer.encode(text,add_special_tokens=False)
            if i!=5:emitted.append(engine.tokenizer.eos_token_id)
            edge_inputs.append(engine.builder.build(edge_prompt,emitted))
        with stage('official_edge_forwards'):
            edge_scores=engine.prm.score(edge_inputs)
        atomic(run/'edge_forwards.json',dict(fixed_inputs_not_generated=True,texts=edge_texts,
            alignments=edge_inputs,scores=edge_scores,training_batch_unchanged=True))
        with stage('old_and_reference_logps'):
            old,ref=engine.old_and_reference(rows)
        batches={}
        with stage('signals'):
            for arm in ('P_VR','P_PRM'):
                batches[arm]=engine.prepare(arm,rows,scored,old,ref)
                atomic(run/(arm+'_signal_receipt.json'),batches[arm].receipt)
            flipped=[dict(r,outcome=1-r['outcome']) for r in rows]
            pp=engine.prepare('P_PRM',flipped,scored,old,ref)
            pv=engine.prepare('P_VR',flipped,None,old,ref)
            assert torch.equal(pp.advantages,batches['P_PRM'].advantages)
            assert not torch.equal(pv.advantages,batches['P_VR'].advantages)
            torch.save(dict(signals={arm:{k:getattr(b,k).detach().cpu() for k in TENSOR_FIELDS}
                                     for arm,b in batches.items()},reference_logps=ref),run/'signals.pt')
            atomic(run/'outcome_independence.json',dict(passed=True,PRM_unchanged=True,VR_changed=True,
                calls_are_separate_validation_branches=True,normalization_per_branch=1))
            del pp,pv
        audits={}
        for arm in ('P_VR','P_PRM'):
            with stage(arm+'_backward'):
                audits[arm]=engine.backward_no_update(batches[arm],rows)
                atomic(run/(arm+'_loss_audit.json'),audits[arm])
                engine.prm.verify_frozen()
        with stage('parameter_hash_after'):
            actor_after=parameter_hash(engine.model);prm_after=parameter_hash(engine.prm.model)
            assert actor_before==actor_after and prm_before==prm_after
            assert adapter_hash(engine.model)==engine.initial_adapter
        after=protected_snapshot();assert before==after,'Protected previous files changed'
        atomic(run/'preservation.json',dict(passed=True,hashed_files=len(before['sha256']),
            inventoried_files=len(before['inventory']),actor_before=actor_before,actor_after=actor_after,
            prm_before=prm_before,prm_after=prm_after,actor_and_prm_parameters_unchanged=True,
            original_AB_and_verl_preserved=True))
        prm_tokens=sum(len(a['prm_input_ids']) for a in alignments)
        result=dict(passed=True,status='completed_no_update',optimizer_constructed=False,optimizer_steps=0,
            original_AB_and_verl_preserved=True,actor_and_prm_parameters_unchanged=True,
            fresh_generated_responses=32,generation=generation,stages=stages,total_seconds=time.time()-started,
            emitted_tokens_per_second=generation['emitted_tokens']/generation['seconds'],
            prm_input_tokens=prm_tokens,prm_responses_per_second=32/stages['official_prm_scoring']['seconds'],
            prm_input_tokens_per_second=prm_tokens/stages['official_prm_scoring']['seconds'],
            exact_token_alignment_rows=len(alignments),mapped_boundaries=sum(len(a['actor_boundary_positions']) for a in alignments),
            actor_processed_rows_changed=sum(a['process_sequences_changed'] for a in alignments),
            original_truncated=sum(not r['original_terminated'] for r in rows),
            outcomes=dict(correct=sum(r['outcome'] for r in rows),total=32),
            arms={arm:{k:v for k,v in audit.items() if k!='loss_trace'} for arm,audit in audits.items()},
            signal_receipts={arm:batch.receipt for arm,batch in batches.items()},
            limitations=['One fixed four-question batch; not convergence or accuracy evaluation.',
                'Shared A100 timings; no optimizer state allocation, so not a certified training memory peak.',
                'P-VR measured with frozen PRM resident for joint testing; standalone P-VR requires less memory.',
                'Original data/prompt/verifier, single-card LoRA and batch32 are explicit resource/protocol adaptations.',
                'Truncated output retains archived verifier failure status despite official inserted EOS.',
                'No resume or multi-update lifecycle test; no new reward or length/empty penalty.'])
        atomic(run/'result.json',result)
        atomic(run/'status.json',dict(status='completed_no_update',passed=True,pid=os.getpid(),optimizer_steps=0,finished=time.time()))
    except Exception as exc:
        atomic(run/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
        atomic(run/'status.json',dict(status='failed',pid=os.getpid(),error=str(exc),optimizer_steps=0))
        raise
    finally:
        if monitor is not None:monitor.close()


if __name__=='__main__':main()

