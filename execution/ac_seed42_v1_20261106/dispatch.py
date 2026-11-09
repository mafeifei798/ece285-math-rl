"""Dispatch exactly one authorized continuation, never retry an existing intent."""
from inspect_remote import SSH,HERE
import subprocess,json

REMOTE=r'''
from pathlib import Path
import os,json,time,fcntl,subprocess,hashlib,psutil
r=Path('/path/to/ece285_math_rl_project');s=r/'experiments/ac_seed42_v1_20261106';p=r/'runs/AC_seed42_V1_20261106';a=r/'audits/AC_seed42_V1_20261106'
lock=(p/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((a/'preflight.json').read_text())['passed']
assert not any((p/n).exists() for n in ['dispatch.json','dispatch_intent.json','started.json']),'Prior dispatch; inspect instead'
with (r/'receipts/abc_seed17_resume_v1_gpu1.lock').open('r') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
for process in psutil.process_iter(['pid','cmdline']):
 try:
  cmd=' '.join(process.info['cmdline'] or [])
  assert process.pid==os.getpid() or not (str(r) in cmd and any(x in cmd for x in ['controller.py','train_arm.py','evaluate.py'])),('Project job already exists',process.pid)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
protocol=json.loads((s/'protocol.json').read_text())
assert subprocess.check_output(['nvidia-smi','-i',str(protocol['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==protocol['gpu_uuid']
header=json.loads((r/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json').read_text());assert header['passed']
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(protocol['gpu']),VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='1',CPATH=header['CPATH'],PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[str(r/'.venv/bin/python'),'-B','-u',str(s/'controller.py')]
with (p/'dispatch_intent.json').open('x') as f:json.dump({'unix':time.time(),'command':command,'seed':42,'start_steps':protocol['start_steps'],'final_step':288},f)
with (p/'controller.log').open('x') as log:
 child=subprocess.Popen(command,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d={'pid':child.pid,'create_time':psutil.Process(child.pid).create_time(),'unix':time.time(),'command':command,'run':str(p),'source':str(s),
 'source_release_sha256':hashlib.sha256((s/'release.json').read_bytes()).hexdigest(),
 'seed':42,'start_steps':protocol['start_steps'],'milestones':[2,144,288],'VLLM_USE_V1':'1','gpu_index':protocol['gpu'],'gpu_uuid':protocol['gpu_uuid'],
 'reward_loss_changed':False,'requested_new_seed':True,'new_PRM_adaptation':False,'no_automatic_retry':True,'existing_results_preserved':True}
with (p/'dispatch.json').open('x') as f:json.dump(d,f,indent=2)
print(json.dumps(d))
'''
if __name__=='__main__':
    r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=REMOTE,text=True,capture_output=True)
    if r.returncode:raise RuntimeError((r.stdout,r.stderr))
    d=json.loads(r.stdout);(HERE/'dispatch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
