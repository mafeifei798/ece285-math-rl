"""Read-only snapshot; never launches, retries or changes remote files."""
from pathlib import Path
import subprocess,json,datetime
HERE=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import json,os,time
r=Path('/path/to/ece285_math_rl_project/runs/Final_four_eval_seed17_1M_20261029')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (ValueError,TypeError,ProcessLookupError):return False
d={n:read(r/(n+'.json')) for n in ['dispatch','controller','status','progress','generation_complete','acceptance','comparison','controller_failure','generation_failure']}
d['unix']=time.time();d['controller_alive']=alive((d['dispatch'] or {}).get('pid'));d['child_alive']=alive((d['controller'] or {}).get('child_pid'))
log=Path((d['controller'] or {}).get('log',str(r/'controller.log')));d['log_tail']=log.read_text(errors='replace')[-2400:] if log.exists() else ''
print(json.dumps(d,ensure_ascii=False))
'''
if __name__=='__main__':
 cmd=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','USER@GPU_HOST','/path/to/ece285_math_rl_project/.venv/bin/python -B -']
 r=subprocess.run(cmd,input=REMOTE,text=True,capture_output=True,check=True);d=json.loads(r.stdout)
 (HERE/'latest_snapshot.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:d[k] for k in ['unix','controller','progress','controller_alive','child_alive','controller_failure','generation_failure']},ensure_ascii=False,indent=2))
 if d.get('comparison'):print(json.dumps({'summary':d['comparison']['summary'],'paired':d['comparison']['paired'],'cost':d['comparison']['cost']},ensure_ascii=False,indent=2))
 print(d['log_tail'])
