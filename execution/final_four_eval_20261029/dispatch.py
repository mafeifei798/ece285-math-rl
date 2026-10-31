"""Launch the user-authorized finite evaluation exactly once; no new training."""
from pathlib import Path
import subprocess,json
HERE=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import json,subprocess,os,time,hashlib,fcntl
r=Path('/srv/research/ece285_math_rl_project');s=r/'experiments/final_four_eval_20261029';p=r/'runs/Final_four_eval_seed17_1M_20261029';a=r/'audits/Final_four_eval_seed17_1M_20261029'
lock=(p/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((a/'preflight.json').read_text())['passed']
assert not (p/'dispatch.json').exists() and not (p/'dispatch_intent.json').exists(),'Already dispatched or ambiguous prior outcome; inspect only'
assert json.loads((p/'status.json').read_text())['stage']=='admitted_CPU_only'
free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=20000,free
header=json.loads((r/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json').read_text());assert header['passed']
cache=r/'cache/final_four_eval_20261029'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CPATH=header['CPATH'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
    VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONUNBUFFERED='1')
command=[str(r/'.venv/bin/python'),'-B','-u',str(s/'controller.py')]
with (p/'dispatch_intent.json').open('x') as f:json.dump({'unix':time.time(),'command':command,'training':False},f)
with (p/'controller.log').open('a') as out:
    child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
d={'pid':child.pid,'unix':time.time(),'source':str(s),'run':str(p),'command':command,'GPU':1,'GPU_free_MiB_before':free,
   'source_release_sha256':hashlib.sha256((s/'release.json').read_bytes()).hexdigest(),
   'contract_sha256':hashlib.sha256((p/'contract.json').read_bytes()).hexdigest(),'generations':7276,'new_training':False,'new_PRM_inference':False,
   'environment_only_project_cache':str(cache),'no_automatic_retry':True}
with (p/'dispatch.json').open('x') as f:json.dump(d,f,indent=2)
print(json.dumps(d))
'''
cmd=['ssh','-p','11032','-i','/home/researcher/.ssh/research_server_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','researcher@server.example.invalid','/srv/research/ece285_math_rl_project/.venv/bin/python -B -']
r=subprocess.run(cmd,input=REMOTE,text=True,capture_output=True);assert r.returncode==0,(r.stdout,r.stderr)
d=json.loads(r.stdout);(HERE/'dispatch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
