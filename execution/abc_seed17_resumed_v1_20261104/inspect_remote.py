"""Read-only resume inventory; never starts a model or changes remote state."""
from pathlib import Path
import subprocess, json
HERE = Path(__file__).resolve().parent
SSH = ['ssh', '-p', '11032', '-i', '/home/USER/.ssh/fym_125_76_110_84_ed25519',
       '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ConnectTimeout=15', 'USER@GPU_HOST']
REMOTE = r'''
from pathlib import Path
import os, json, time, hashlib, subprocess, psutil, fcntl
r=Path('/path/to/ece285_math_rl_project')
old=r/'runs/ABC_seed17_resume6M_V1_20261030'
new=r/'runs/ABC_seed17_resumed_V1_20261104'
def read(p): return json.loads(p.read_text()) if p.exists() else None
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
d={'unix':time.time(),'old_controller':read(old/'controller.json'),'pause':read(old/'pause_receipt.json'),
   'new_controller':read(new/'controller.json'),'new_dispatch':read(new/'dispatch.json'),
   'new_failure':read(new/'controller_failure.json'),'new_exists':new.exists(),'arms':{},'processes':[], 'instructions':{}}
for p in psutil.process_iter(['pid','ppid','cmdline','create_time','status','username']):
 try:
  cmd=' '.join(p.info['cmdline'] or [])
  if p.pid!=os.getpid() and str(r) in cmd:d['processes'].append(p.info)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
for arm, v in d['pause']['arms'].items():
 p=Path(v['resume_checkpoint']); checks={n:(p/n).is_file() and sha(p/n)==h for n,h in v['files_sha256'].items()}
 cp=read(p/'pilot_checkpoint.json');account=read(p/'completion_accounting.json')
 d['arms'][arm]={'checkpoint_exists':p.is_dir(),'saved_checks':checks,'contract_hash_matches':sha(p/'pilot_checkpoint.json')==v['checkpoint_contract_sha256'],
  'checkpoint_contract':cp,'accounting':account,'old_progress':read(old/arm/'progress.json'),
  'old_stage144':read(old/arm/'stage_144_result.json'),'old_stage288':read(old/arm/'stage_288_result.json'),
  'new_progress':read(new/arm/'progress.json'),'new_failure':read(new/arm/'failure.json')}
d['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
d['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
d['disk']=subprocess.check_output(['df','-h',str(r)],text=True)
for p in [Path('/AGENTS.md'),Path('/yimin/AGENTS.md'),Path('/yimin/lzx/AGENTS.md'),Path('/path/to/project_parent/AGENTS.md'),r/'AGENTS.md',r/'experiments/AGENTS.md',r/'runs/AGENTS.md',r/'audits/AGENTS.md']:
 if p.exists():d['instructions'][str(p)]=p.read_text()
d['locks']={}
for p in [old/'controller.lock',r/'receipts/abc_seed17_resume_v1_gpu1.lock']:
 if p.exists():
  with p.open('r') as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);d['locks'][str(p)]='available';fcntl.flock(f,fcntl.LOCK_UN)
   except BlockingIOError:d['locks'][str(p)]='held'
d['assets']=read(r/'receipts/assets.json')
d['prm_receipts']={str(p.relative_to(r)):read(p) for p in (r/'receipts').glob('*qwen*prm*.json')}
d['model_paths']=[{'path':str(p),'bytes':sum(f.stat().st_size for f in p.rglob('*.safetensors'))} for p in (r/'models').iterdir() if p.is_dir() and ('Qwen' in p.name)]
print(json.dumps(d))
'''
if __name__ == '__main__':
    result=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'], input=REMOTE,text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError((result.returncode,result.stderr[-3500:],result.stdout[-1000:]))
    d=json.loads(result.stdout)
    (HERE/'latest_snapshot.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    if not (HERE/'before_resume_snapshot.json').exists():
        (HERE/'before_resume_snapshot.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:d[k] for k in ['new_exists','new_controller','new_failure','processes','gpu','gpu_processes','disk','instructions','locks','model_paths']}|{'checkpoints':{a:{'step':v['checkpoint_contract']['global_step'],'tokens':v['accounting']['generated_completion_tokens'],'all_hashes_match':all(v['saved_checks'].values()) and v['contract_hash_matches'],'new_progress':v['new_progress']} for a,v in d['arms'].items()}},ensure_ascii=False,indent=2))
