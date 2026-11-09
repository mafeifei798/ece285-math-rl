"""Lightweight read-only status for the already dispatched finite controller."""
from pathlib import Path
import json
import subprocess
from inspect_remote import SSH

HERE = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import json, os, time, subprocess, psutil
root=Path('/path/to/ece285_math_rl_project')
run=root/'runs/AC_seed42_V1_20261106'
def read(p):
 if not p.exists():return None
 try:return json.loads(p.read_text())
 except Exception as e:return {'read_error':str(e)}
def tail(p,limit=9000):
 if not p.exists():return None
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-limit));return f.read().decode('utf-8','replace')
controller=read(run/'controller.json')
d={'unix':time.time(),'controller':controller,'dispatch':read(run/'dispatch.json'),
 'controller_failure':read(run/'controller_failure.json'),'comparison':read(run/'comparison.json'),
 'arms':{},'processes':[],'log_tail':None,'common_initialization':read(run/'common_initialization.json'),'acceptance_gate':read(run/'acceptance_gate.json')}
for arm in ('A','C_bounded'):
 p=run/arm
 d['arms'][arm]={n[:-5]:read(p/n) for n in ('progress.json','last_checkpoint.json','failure.json',
  'optimizer_acceptance_2.json','resume_acceptance_2.json','stage_2_result.json','resume_build_2.json','optimizer_acceptance_144.json','resume_acceptance_144.json','stage_144_result.json',
  'optimizer_acceptance_288.json','resume_acceptance_288.json','stage_288_result.json')}
if controller and controller.get('log'):
 d['log_tail']=tail(Path(controller['log']))
else:
 logs=sorted((run/'logs').glob('*.log'),key=lambda p:p.stat().st_mtime)
 if logs:d['log_tail']=tail(logs[-1])
for p in psutil.process_iter(['pid','ppid','cmdline','create_time','status']):
 try:
  if p.pid!=os.getpid() and str(root) in ' '.join(p.info['cmdline'] or []):d['processes'].append(p.info)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
d['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
d['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
print(json.dumps(d))
'''

if __name__ == '__main__':
    r = subprocess.run(SSH + ['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],
                       input=REMOTE, text=True, capture_output=True)
    if r.returncode:
        raise RuntimeError((r.returncode, r.stderr[-3500:], r.stdout[-1000:]))
    d = json.loads(r.stdout)
    (HERE / 'live_status.json').write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(d, ensure_ascii=False, indent=2))
