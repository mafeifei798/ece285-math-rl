"""Finite user-authorized pipeline; locks, no duplicate job or automatic retries."""
import fcntl,os,sys,subprocess,time,traceback
from pilot_common import *

def gpu_headroom():
    waited=0
    while True:
        p=subprocess.run(['nvidia-smi','-i','1','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        free=float(p.stdout.split(',')[0]);atomic(PAIR/'resource_snapshot.json',dict(free_MiB=free,raw=p.stdout.strip(),unix=time.time()))
        if free>=42000:return
        if waited>=18*3600:raise RuntimeError('Insufficient GPU1 headroom; no other process touched')
        event('waiting_GPU1_memory',free_MiB=free);time.sleep(30);waited+=30
def run(script,*args,stage):
    gpu_headroom() if script in ['train_arm.py','evaluate.py'] else None
    event(stage,command=[script,*args]);log=PAIR/'logs'/f'{stage}.log';log.parent.mkdir(exist_ok=True)
    assert not log.exists(),('Stage already attempted; inspect it before recovery',stage)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        VLLM_CACHE_ROOT=str(ROOT/'cache/pilot_v2/vllm'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'cache/pilot_v2/inductor'),
        TRITON_CACHE_DIR=str(ROOT/'cache/pilot_v2/triton'),XDG_CACHE_HOME=str(ROOT/'cache/pilot_v2/xdg'),
        HF_HOME=str(ROOT/'cache/pilot_v2/huggingface'),HF_MODULES_CACHE=str(ROOT/'cache/pilot_v2/hf_modules'))
    with log.open('x') as f:
        child=subprocess.Popen([sys.executable,'-B',str(HERE/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        atomic(PAIR/'child.json',dict(pid=child.pid,stage=stage,log=str(log),unix=time.time()))
        code=child.wait()
    if code:raise RuntimeError(f'{stage} exited {code}; no automatic retry; evidence {log}')
def main():
    guard();verify();PAIR.mkdir(parents=True,exist_ok=True)
    lock=(PAIR/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (PAIR/'started.json').exists(),'Controller already dispatched; inspect receipts/PIDs first'
    atomic(PAIR/'started.json',dict(pid=os.getpid(),unix=time.time(),authorization=read(HERE/'protocol.json'),source_release_sha256=verify()))
    try:
        run('preflight.py',stage='CPU_preflight')
        for a in ARMS:run('train_arm.py','--arm',a,'--phase','smoke',stage=a+'_two_updates')
        smoke={a:read(PAIR/a/'smoke_result.json') for a in ARMS}
        assert all(s['passed'] and s['global_step']==2 for s in smoke.values())
        specs=[read(PAIR/a/'run_config.json') for a in ARMS]
        assert len({s['initial_adapter_digest'] for s in specs})==1
        normalized=[{k:v for k,v in s['settings'].items() if k not in ['output_dir','logging_dir','run_name']} for s in specs]
        assert all(s==normalized[0] for s in normalized)
        # Identical initial policy, generation protocol and restored scorer RNG
        # should yield identical first-batch samples across arms.
        initial=[[(r['problem_id'],r['completion_token_ids']) for r in rows(PAIR/a/'rollouts.jsonl')[:32]] for a in ARMS]
        assert all(v==initial[0] for v in initial)
        atomic(PAIR/'smoke_gate.json',dict(passed=True,source_release_sha256=verify(),all_three_real_updates=smoke,
            matched_initial_adapter=True,matched_training_arguments=True,matched_initial_raw_generation=True))
        run('evaluate.py','--arm','initial',stage='initial_validation')
        for a in ARMS:
            run('train_arm.py','--arm',a,'--phase','continue',stage=a+'_continue_to_1M')
            run('evaluate.py','--arm',a,stage=a+'_validation')
        run('analyze.py',stage='paired_analysis')
        event('complete',comparison=str(PAIR/'comparison.json'),no_6M_started=True,D_adaptation=False)
    except Exception as e:
        atomic(PAIR/'controller_failure.json',dict(error=str(e),traceback=traceback.format_exc(),unix=time.time()))
        event('failed_stopped',error=str(e),no_automatic_retry=True);raise

if __name__=='__main__':main()
