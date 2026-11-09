"""Install one independent resumed version and perform CPU-only admission."""
from pathlib import Path
import ast,base64,hashlib,json,subprocess,zlib
from inspect_remote import SSH,HERE
SOURCE=HERE.parents[1]/'experiments/abc_seed17_resumed_v1_20261104'

def main():
    files={p.name:p.read_text() for p in SOURCE.iterdir() if p.suffix in ['.py','.json'] and p.name!='release.json'}
    for n,s in files.items():
        if n.endswith('.py'):ast.parse(s,filename=n)
    release={'files':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()},
        'dependencies':json.loads(files['dependency_hashes.json']),
        'scope':'Explicitly authorized2026-11-04 resume from A115/C50/Cbounded50; seed17 only, common144/288 milestones, frozen reward/loss and V1 evaluation'}
    (SOURCE/'release.json').write_text(json.dumps(release,indent=2)+'\n');files['release.json']=(SOURCE/'release.json').read_text()
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    remote=r'''
from pathlib import Path
import os, json, base64, zlib, subprocess, fcntl, psutil
r=Path('/path/to/ece285_math_rl_project');s=r/'experiments/abc_seed17_resumed_v1_20261104';p=r/'runs/ABC_seed17_resumed_V1_20261104';a=r/'audits/ABC_seed17_resumed_V1_20261104'
assert not any(x.exists() for x in [s,p,a]),'Existing resumed version; inspect, do not repeat preparation'
for process in psutil.process_iter(['pid','cmdline']):
 try:
  cmd=' '.join(process.info['cmdline'] or [])
  assert process.pid==os.getpid() or not (str(r) in cmd and any(x in cmd for x in ['controller.py','train_arm.py','evaluate.py'])),('Existing project job',process.pid)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
with (r/'receipts/abc_seed17_resume_v1_gpu1.lock').open('r') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)));s.mkdir();p.mkdir();a.mkdir()
 for n,v in files.items():
  assert Path(n).name==n;(s/n).write_text(v)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
 # Release guard before the child independently checks the same lock.
 fcntl.flock(f,fcntl.LOCK_UN)
 result=subprocess.run([str(r/'.venv/bin/python'),'-B',str(s/'preflight.py')],capture_output=True,text=True,env=env)
 (a/'preflight_stdout.log').write_text(result.stdout);(a/'preflight_stderr.log').write_text(result.stderr)
 assert result.returncode==0,{'returncode':result.returncode,'stdout':result.stdout[-2000:],'stderr':result.stderr[-5000:]}
 d=json.loads((a/'preflight.json').read_text());print(json.dumps(d))
'''.replace('PAYLOAD',repr(payload))
    result=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=remote,text=True,capture_output=True)
    (HERE/'prepare_stdout.txt').write_text(result.stdout);(HERE/'prepare_stderr.txt').write_text(result.stderr)
    if result.returncode:raise RuntimeError((result.returncode,result.stdout[-1500:],result.stderr[-5000:]))
    d=json.loads(result.stdout);(HERE/'admission.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:d[k] for k in ['passed','start_steps','protected_files','CPU_reference_rows','runtime_bytes_identical','advantages_bytes_identical','evaluation_bytes_identical','Acceptance_callback_AST_identical','CUDA_initialized','gpu_identity','PRM_shards_present','versions']},indent=2))

if __name__=='__main__':main()
