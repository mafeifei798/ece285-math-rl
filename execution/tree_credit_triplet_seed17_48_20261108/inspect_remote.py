"""Read-only status of this one finite paired run; never dispatches."""
from pathlib import Path
import subprocess,runpy,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
CODE=r'''
from pathlib import Path
import json,psutil,subprocess,time
root=Path('/path/to/ece285_math_rl_project');run=root/'runs/Tree_credit_triplet_seed17_48_20261108';audit=root/'audits/Tree_credit_triplet_seed17_48_20261108'
def read(p):return json.loads(p.read_text()) if p.exists() else None
x={'unix':time.time(),'controller':read(run/'controller.json'),'launch':read(run/'launch_receipt.json'),
   'recovery_launch':read(run/'recovery_launch_receipt.json'),'preflight':read(audit/'preflight.json'),'cpu_check':read(audit/'cpu_check.json'),
   'no_update':{n:read(audit/'no_update'/(n+'.json')) for n in ['progress','result','failure']},'arms':{}}
for a in ['chain_cb','tree_vr','tree_cb']:
    p=run/a;x['arms'][a]={n:read(p/(n+'.json')) for n in ['progress','first_update_acceptance','result','failure','last_checkpoint']}
x['evaluation']=read(run/'evaluation/progress.json');x['evaluation_summary']=read(run/'evaluation/summary.json');x['comparison']=read(run/'comparison.json')
c=x['controller'] or {};receipt=x['recovery_launch'] or x['launch'] or {};pid=receipt.get('pid')
x['controller_alive']=bool(pid and psutil.pid_exists(pid) and abs(psutil.Process(pid).create_time()-receipt['create_time'])<.1 and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE)
x['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
if c.get('stage')=='failed':
    x['controller_log_tail']=(run/'controller.log').read_text()[-2500:]
    children=c.get('children') or []
    if children:
        p=run/(children[-1]['stage']+'.log')
        if p.exists():x['child_log_tail']=p.read_text()[-8500:]
print(json.dumps(x))
'''


def snapshot():
    r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=CODE,text=True,capture_output=True,timeout=45)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    x=json.loads(r.stdout);p=HERE/'status.tmp';p.write_text(json.dumps(x,indent=2)+'\n');p.replace(HERE/'status.json')
    return x


def compact(x):
    d={'controller_stage':(x['controller'] or {}).get('stage'),'controller_alive':x['controller_alive'],'gpu':x['gpu'],
       'CPU_passed':bool(x['cpu_check'] and x['cpu_check'].get('passed')),
       'no_update_passed':bool(x['no_update']['result'] and x['no_update']['result'].get('passed')),
       'no_update_failure':x['no_update']['failure'],
       'arms':{a:{'updates':(v['progress'] or {}).get('additional_updates'),'tokens':(v['progress'] or {}).get('generated_tokens'),
                   'first_update_passed':bool(v['first_update_acceptance']),'finished':bool(v['result']),'failure':v['failure']} for a,v in x['arms'].items()},
       'evaluation':x['evaluation'],'comparison':x['comparison']}
    if 'child_log_tail' in x:d['failure_tail']=x['child_log_tail']
    return d


if __name__=='__main__':print(json.dumps(compact(snapshot()),ensure_ascii=False,indent=2))
