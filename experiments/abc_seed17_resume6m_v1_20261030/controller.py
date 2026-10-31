"""Finite authorized seed17 continuation; no retries, extra seeds or budget expansion."""
from pilot_common import *
import fcntl,subprocess,sys,traceback

DEADLINE=None
def run(script,*args,label,gpu=False):
    if time.time()>DEADLINE:raise RuntimeError('Finite pipeline wall budget reached; preserve checkpoints')
    threshold=42000 if script=='train_arm.py' else 20000
    waited=time.monotonic()
    while gpu:
        free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free>=threshold:break
        if time.monotonic()-waited>18*3600 or time.time()>DEADLINE:raise RuntimeError('GPU1 memory unavailable; no other process touched')
        event('waiting_GPU1_memory',next_stage=label,free_MiB=free,required_MiB=threshold);time.sleep(30)
    log=PAIR/'logs'/f'{label}.log';assert not log.exists(),('Stage already attempted; inspect, never rerun silently',label)
    header=read(ROOT/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json');assert header['passed']
    cache=ROOT/'cache/abc_seed17_resume6m_v1_20261030'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='1',
        CPATH=header['CPATH'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
        XDG_CACHE_HOME=str(cache/'xdg'),HF_HOME=str(cache/'huggingface'),HF_MODULES_CACHE=str(cache/'hf_modules'),
        ABC_PIPELINE_DEADLINE=str(DEADLINE))
    started=time.time()
    with log.open('x') as f:
        child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/script),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        event(label,child_pid=child.pid,log=str(log),started_unix=started,deadline_unix=DEADLINE)
        code=child.wait()
    append(PAIR/'stages.jsonl',{'stage':label,'started_unix':started,'finished_unix':time.time(),'exit_code':code})
    assert code==0,(label,code,'No automatic retry; preserve committed raw rollouts and checkpoints')
    unchanged()

def main():
    global DEADLINE
    guard();verify();assert read(AUDIT/'preflight.json')['passed'];DEADLINE=time.time()+48*3600
    locks=[]
    for p in [PAIR/'controller.lock',ROOT/'receipts/abc_seed17_resume_v1_gpu1.lock']:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert not (PAIR/'started.json').exists(),'Already started; inspect PID/checkpoint before any action'
    atomic(PAIR/'started.json',{'pid':os.getpid(),'unix':time.time(),'deadline_unix':DEADLINE,'protocol':read(HERE/'protocol.json'),'source_release_sha256':verify()})
    (PAIR/'logs').mkdir(exist_ok=True)
    try:
        for arm in ARMS:run('train_arm.py','--arm',arm,'--target','50',label=arm+'_resume48_to50',gpu=True)
        results={arm:read(PAIR/arm/'stage_50_result.json') for arm in ARMS}
        assert all(r['passed'] and r['global_step']==50 and r['engine']=='V1' for r in results.values())
        specs=[read(PAIR/arm/'resume_build_50.json') for arm in ARMS]
        args=[{k:v for k,v in s['settings'].items() if k not in ['output_dir','logging_dir','run_name']} for s in specs]
        assert all(v==args[0] for v in args)
        for arm in ARMS:
            assert read(PAIR/arm/'resume_acceptance_50.json')['passed']
            assert len(rows(PAIR/arm/'rollouts.jsonl'))==1600
        atomic(PAIR/'acceptance_gate.json',{'passed':True,'all_three_true_continuations':True,'common_step':50,'engine':'V1','common_training_settings':True,'no_reward_loss_changes':True,'results':results,'unix':time.time()})
        for arm in ARMS:run('train_arm.py','--arm',arm,'--target','144',label=arm+'_to144',gpu=True)
        run('evaluate.py','--step','144',label='common_validation144',gpu=True)
        run('analyze.py','144',label='CPU_acceptance144')
        for arm in ('C_mix','C_bounded','A'):run('train_arm.py','--arm',arm,'--target','288',label=arm+'_to288',gpu=True)
        run('evaluate.py','--step','288',label='common_MATH500_GSM8K288',gpu=True)
        run('analyze.py','288',label='CPU_final_acceptance')
        assert read(PAIR/'comparison.json')['passed']
        event('complete',comparison=str(PAIR/'comparison.json'),seed=17,extra_seeds=False)
    except Exception as exc:
        atomic(PAIR/'controller_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'unix':time.time(),'no_automatic_retry':True})
        event('failed_stopped',error=str(exc));raise
if __name__=='__main__':main()
