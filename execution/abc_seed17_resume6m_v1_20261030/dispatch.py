"""Exactly one finite continuation controller; inspect any prior intent instead of retry."""
from inspect_remote import SSH,HERE
import subprocess,json
remote=r'''
from pathlib import Path
import os,json,time,fcntl,subprocess,hashlib
r=Path('/path/to/ece285_math_rl_project');s=r/'experiments/abc_seed17_resume6m_v1_20261030';p=r/'runs/ABC_seed17_resume6M_V1_20261030';a=r/'audits/ABC_seed17_resume6M_V1_20261030'
lock=(p/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((a/'preflight.json').read_text())['passed']
assert not (p/'dispatch.json').exists() and not (p/'dispatch_intent.json').exists(),'Already attempted; inspect only'
header=json.loads((r/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json').read_text());assert header['passed']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='1',CPATH=header['CPATH'],PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[str(r/'.venv/bin/python'),'-B','-u',str(s/'controller.py')]
with (p/'dispatch_intent.json').open('x') as f:json.dump({'unix':time.time(),'command':command,'seed':17,'origin_step':48,'final_step':288},f)
with (p/'controller.log').open('a') as log:child=subprocess.Popen(command,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d={'pid':child.pid,'unix':time.time(),'command':command,'run':str(p),'source':str(s),'source_release_sha256':hashlib.sha256((s/'release.json').read_bytes()).hexdigest(),
 'seed':17,'arms':['A','C_mix','C_bounded'],'start_step':48,'milestones':[50,144,288],'VLLM_USE_V1':'1','reward_loss_changed':False,'extra_seed':False,'new_PRM_adaptation':False,'no_automatic_retry':True}
with (p/'dispatch.json').open('x') as f:json.dump(d,f,indent=2)
print(json.dumps(d))
'''
r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=remote,text=True,capture_output=True)
assert r.returncode==0,(r.stdout,r.stderr)
d=json.loads(r.stdout);(HERE/'dispatch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
