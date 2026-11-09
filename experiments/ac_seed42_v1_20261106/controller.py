"""One finite resumed pipeline. No retries, extra seeds, or budget changes."""
from pilot_common import *
import fcntl,subprocess,sys,traceback,psutil

DEADLINE=None
def run(script,*args,label,gpu=False):
    if time.time()>DEADLINE:raise RuntimeError('Finite pipeline wall budget reached; preserve checkpoints')
    device=read(HERE/'protocol.json')['gpu_uuid'];threshold=42000 if script=='train_arm.py' else 20000
    waited=time.monotonic()
    while gpu:
        free=int(subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free>=threshold:break
        if time.monotonic()-waited>18*3600 or time.time()>DEADLINE:raise RuntimeError('GPU memory unavailable; no other process touched')
        event('waiting_GPU_memory',next_stage=label,free_MiB=free,required_MiB=threshold,gpu_uuid=device);time.sleep(30)
    log=PAIR/'logs'/f'{label}.log';assert not log.exists(),('Existing stage; do not resample silently',label)
    header=read(ROOT/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json');assert header['passed']
    cache=ROOT/'cache/ac_seed42_v1_20261106'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(read(HERE/'protocol.json')['gpu']),VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='1',
        CPATH=header['CPATH'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
        XDG_CACHE_HOME=str(cache/'xdg'),HF_HOME=str(cache/'huggingface'),HF_MODULES_CACHE=str(cache/'hf_modules'),ABC_PIPELINE_DEADLINE=str(DEADLINE))
    started=time.time()
    with log.open('x') as f:
        child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/script),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        event(label,child_pid=child.pid,child_create_time=psutil.Process(child.pid).create_time(),log=str(log),started_unix=started,deadline_unix=DEADLINE,gpu_uuid=device)
        code=child.wait()
    append(PAIR/'stages.jsonl',{'stage':label,'started_unix':started,'finished_unix':time.time(),'exit_code':code})
    assert code==0,(label,code,'No automatic retry; preserve checkpoints and committed generation')
    unchanged()

def verify_common_144():
    results={a:read(PAIR/a/'stage_144_result.json') for a in ARMS}
    assert all(v['passed'] and v['global_step']==144 and v['engine']=='V1' for v in results.values())
    specs=[read(PAIR/a/'resume_build_144.json') for a in ARMS]
    args=[{k:v for k,v in s['settings'].items() if k not in ['output_dir','logging_dir','run_name']} for s in specs]
    assert all(v==args[0] for v in args)
    for arm in ARMS:
        assert read(PAIR/arm/'resume_acceptance_144.json')['passed']
        assert len(rows(PAIR/arm/'rollouts.jsonl'))==144*32
    atomic(PAIR/'common144_gate.json',{'passed':True,'common_update':144,'same_training_settings':True,'results':results,'unix':time.time()})

def main():
    global DEADLINE
    guard();verify();assert read(AUDIT/'preflight.json')['passed'];DEADLINE=time.time()+48*3600
    locks=[]
    for p in [PAIR/'controller.lock',ROOT/'receipts/abc_seed17_resume_v1_gpu1.lock']:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert not (PAIR/'started.json').exists(),'Already started; inspect receipt/PID/checkpoints'
    atomic(PAIR/'started.json',{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'unix':time.time(),
        'deadline_unix':DEADLINE,'protocol':read(HERE/'protocol.json'),'source_release_sha256':verify()})
    (PAIR/'logs').mkdir(exist_ok=True)
    try:
        for arm in ARMS:run('train_arm.py','--arm',arm,'--target','2',label=arm+'_fresh_to2',gpu=True)
        specs=[read(PAIR/a/'resume_build_2.json') for a in ARMS]
        assert len({s['initial_adapter_digest'] for s in specs})==1
        settings=[{k:v for k,v in s['settings'].items() if k not in ['output_dir','logging_dir','run_name']} for s in specs]
        assert settings[0]==settings[1] and all(s['seed']==42 for s in specs)
        for arm in ARMS:
            assert read(PAIR/arm/'stage_2_result.json')['passed']
            assert read(PAIR/arm/'resume_acceptance_2.json')['actor_updated']
            assert read(PAIR/arm/'optimizer_acceptance_2.json')['PRM_in_optimizer']==False
            assert len(rows(PAIR/arm/'rollouts.jsonl'))==64
        atomic(PAIR/'acceptance_gate.json',{'passed':True,'common_update':2,'same_fresh_initialization':True,
            'same_training_settings':True,'actual_optimizer_updates_both_arms':True,'no_reward_loss_change':True,'unix':time.time()})
        for arm in ARMS:run('train_arm.py','--arm',arm,'--target','144',label=arm+'_resume_to144',gpu=True)
        verify_common_144()
        run('evaluate.py','--step','144',label='common_validation144',gpu=True)
        run('analyze.py','144',label='CPU_acceptance144')
        for arm in ('C_bounded','A'):run('train_arm.py','--arm',arm,'--target','288',label=arm+'_to288',gpu=True)
        run('evaluate.py','--step','288',label='common_MATH500_GSM8K288',gpu=True)
        run('analyze.py','288',label='CPU_final_acceptance')
        assert read(PAIR/'comparison.json')['passed']
        event('complete',comparison=str(PAIR/'comparison.json'),seed=42,extra_seeds=False)
    except Exception as exc:
        atomic(PAIR/'controller_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'unix':time.time(),'no_automatic_retry':True})
        event('failed_stopped',error=str(exc));raise

if __name__=='__main__':main()
