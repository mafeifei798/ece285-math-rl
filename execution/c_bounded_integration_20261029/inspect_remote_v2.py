"""Read-only inspection; never dispatch or train."""
from pathlib import Path
import subprocess,json,datetime
BASE=Path(__file__).resolve().parent
CODE=r'''
from pathlib import Path
import json,time,subprocess
r=Path('/path/to/ece285_math_rl_project');run=r/'runs/C_bounded_no_update_20261029_v2';dispatch=r/'audits/C_bounded_integration_20261029_dispatch_v2'
d={'checked_unix':time.time()}
for folder,names in [(dispatch,['dispatch.json']),(run,['status.json','stages.json','result.json','failure.json','generation_checks.json','C_loading.json','independent_version.json','score_parity.json','generation_reuse.json'])]:
 for n in names:
  p=folder/n
  if p.exists():d[n]=json.loads(p.read_text())
p=dispatch/'runner.log'
if p.exists():d['log_tail']=p.read_text(errors='replace')[-4200:]
p=run/'actual_loss_input_trace.jsonl'
if p.exists():d['loss_trace_rows']=len(p.read_text().splitlines())
v=d.get('dispatch.json',{});d['pid_alive']=(Path('/proc')/str(v.get('pid',0))).exists()
d['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
print(json.dumps(d))
'''
if __name__=='__main__':
 cmd=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','USER@GPU_HOST','/path/to/ece285_math_rl_project/.venv/bin/python -B -']
 r=subprocess.run(cmd,input=CODE,text=True,capture_output=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr[-2000:])
 d=json.loads(r.stdout);(BASE/'latest_snapshot_v2.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
