"""Read-only status/evidence retrieval for the already dispatched PRM pair."""
from pathlib import Path
import json
import subprocess
import datetime

LOCAL = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import os,json,subprocess,time
root=Path('/path/to/ece285_math_rl_project')
out=root/'runs/CD_PRM7B_pair_seed17_20261028'
d={'checked_unix':time.time(),'remote_run':str(out)}
for name in ['controller.json','dispatch.json','progress.json','cpu_checks.json','selection.json','anchor_manifest.json','buffer_manifest.json','model_preflight.json','paired_wiring.json']:
 p=out/name
 if p.exists():d[name]=json.loads(p.read_text())
for name in ['batch_manifest.json']:
 p=out/name
 if p.exists():
  v=json.loads(p.read_text());d[name]={k:v.get(k) for k in ['steps','counts','input_tokens','anchors_distinct_used','excluded']}
for arm in ['C','D']:
 data={}
 for name in ['initialization.json','latest_checkpoint.json','completion.json']:
  p=out/arm/name
  if p.exists():data[name]=json.loads(p.read_text())
 p=out/arm/'metrics.jsonl'
 if p.exists():
  rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()];data['last_metrics']=rows[-3:]
 d[arm]=data
ctrl=d.get('controller.json',{})
post=out/'wiring_recovery/postprocess.json'
if post.exists():
 pc=json.loads(post.read_text());d['postprocess']=pc
 active=pc.get('active_child')
 if active:
  d['postprocess_child_alive']=(Path('/proc')/str(active['pid'])).exists()
  p=out/'wiring_recovery'/(active['stage']+'.log')
  if p.exists():d['postprocess_log_tail']=p.read_text(errors='replace')[-2000:]
 p=out/'wiring_recovery/postprocess_controller.log'
 if pc['status']=='failed' and p.exists():d['postprocess_failure_log']=p.read_text(errors='replace')[-2500:]
child=ctrl.get('active_child') or {}
if child:
 p=Path('/proc')/str(child['pid']);d['child_alive']=p.exists()
 p=Path(child['log'])
 if p.exists():d['child_log_tail']=p.read_text(errors='replace')[-4500:]
if ctrl.get('status')=='failed':
 d['controller_log_tail']=(out/'controller.log').read_text(errors='replace')[-4500:]
for a in ['B','C','D']:
 p=out/f'scoring_{a}.json'
 if p.exists():d[f'scoring_{a}.json']=json.loads(p.read_text())
if (out/'summary.json').exists():
 d['completed_summary_available']=True
 v=json.loads((out/'summary.json').read_text());d['summary']={k:v[k] for k in ['status','processbench','D_minus_C_answer_correct_AUROC_CI95','ranking_changes','groups','preservation','actor_optimizer_updates']}
d['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(d,ensure_ascii=False,allow_nan=False))
'''

if __name__=='__main__':
    command=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
             'USER@GPU_HOST','/path/to/ece285_math_rl_project/.venv/bin/python -']
    result=subprocess.run(command,input=REMOTE,text=True,capture_output=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stderr[-1500:])
    data=json.loads(result.stdout);LOCAL.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (LOCAL/f'snapshot_{stamp}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (LOCAL/'latest_snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data,ensure_ascii=False,indent=2))
