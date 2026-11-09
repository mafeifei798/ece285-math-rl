"""Read-only status snapshot of the already dispatched finite triplet."""
from pathlib import Path
import subprocess,runpy,json
ROOT=Path(__file__).resolve().parents[2]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
REMOTE="""from pathlib import Path
import json,psutil,subprocess,time
r=Path('/path/to/ece285_math_rl_project');run=r/'runs/Prefix_triplet_seed17_48_20261107'
def read(p):return json.loads(p.read_text()) if p.exists() else None
x={'unix':time.time(),'controller':read(run/'controller.json'),'launch':read(run/'launch_receipt.json'),'arms':{}}
for a in ['bounded','uniform','prefix']:
 p=run/a;x['arms'][a]={n:read(p/(n+'.json')) for n in ['progress','first_update_acceptance','result','failure','last_checkpoint']}
x['evaluation']=read(run/'evaluation/progress.json');x['evaluation_summary']=read(run/'evaluation/summary.json');x['comparison']=read(run/'comparison.json')
c=x['controller'] or {};pid=c.get('pid');x['controller_alive']=bool(pid and psutil.pid_exists(pid) and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE)
x['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
if c.get('stage')=='failed':
 x['failed_log_tail']=(run/'controller.log').read_text()[-2500:]
 children=c.get('children') or []
 if children:
  p=run/(children[-1]['stage']+'.log')
  if p.exists():x['child_log_tail']=p.read_text()[-6500:]
print(json.dumps(x))
"""
def snapshot():
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python','-B','-'],input=REMOTE,text=True,capture_output=True,timeout=40)
    if p.returncode:raise RuntimeError(p.stderr)
    result=json.loads(p.stdout)
    here=Path(__file__).resolve().parent;tmp=here/'status.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(here/'status.json')
    return result
if __name__=='__main__':
    x=snapshot();compact={'controller_stage':(x['controller'] or {}).get('stage'),'controller_alive':x['controller_alive'],'gpu':x['gpu'],
        'arms':{a:{'step':(v['progress'] or {}).get('additional_updates'),'tokens':(v['progress'] or {}).get('generated_tokens'),
                  'first_update_passed':bool(v['first_update_acceptance']),'finished':bool(v['result']),'failure':v['failure']} for a,v in x['arms'].items()},
        'evaluation':x['evaluation'],'comparison':x['comparison']}
    if 'child_log_tail' in x:compact['failure_log']=x['child_log_tail']
    print(json.dumps(compact,ensure_ascii=False,indent=2))
