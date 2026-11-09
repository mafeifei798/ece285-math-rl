"""Single guarded launch; CPU and zero-update gates precede both training arms."""
from prepare_remote import *


def main():
    prepared=json.loads((HERE/'prepared.json').read_text());assert prepared['prepared']
    code=r'''
from pathlib import Path
import os,json,subprocess,psutil,time,fcntl,hashlib
r=Path('/path/to/ece285_math_rl_project');run=r/'runs/Length_norm_pair_seed17_48_20261107';src=r/'experiments/length_norm_pair_seed17_48_20261107'
lock=(run/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not any((run/n).exists() for n in ['controller.json','launch_receipt.json','dispatch_started.json']),'Already dispatched; inspect existing state'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
for p in psutil.process_iter(['pid','cmdline']):
    try:
        c=p.info['cmdline'] or []
        assert p.pid==os.getpid() or not (any(str(r) in str(x) for x in c) and any(any(n in str(x) for n in ['controller.py','train_arm.py','evaluate.py']) for x in c)),('Existing project process',p.pid)
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
command=[str(r/'.venv/bin/python'),'-B','-u',str(src/'controller.py')]
with (run/'controller.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
receipt={'pid':child.pid,'create_time':psutil.Process(child.pid).create_time(),'unix':time.time(),'command':command,
         'release_sha256':hashlib.sha256((src/'release.json').read_bytes()).hexdigest(),'gpu_uuid':'GPU-9fa4fe23-1ce5-d116-daab-3271dcbaa0ac',
         'authorized_updates_per_arm':48,'seed':17,'CPU_and_real_model_no_update_gates_required':True,'automatic_expansion':False}
(run/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
    r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=code,text=True,capture_output=True,timeout=40)
    if r.returncode:raise RuntimeError(r.stderr[-5000:])
    x=json.loads(r.stdout);(HERE/'launch_receipt.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))


if __name__=='__main__':main()
