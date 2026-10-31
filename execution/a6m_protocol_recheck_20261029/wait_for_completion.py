"""Finite read-only wait for this existing controller; not an automation or dispatcher."""
from inspect_remote import SSH
import subprocess
REMOTE=r'''
from pathlib import Path
import json,time,os
p=Path('/srv/research/ece285_math_rl_project/runs/A6M_protocol_recheck_20261029');last=None
for _ in range(720):
 def read(n):
  f=p/(n+'.json');return json.loads(f.read_text()) if f.exists() else {}
 c=read('controller');g=read('progress');key=(c.get('stage'),g.get('done'))
 if key!=last:
  print(json.dumps({'stage':c.get('stage'),'done':g.get('done',0),'total':1819,'dataset':g.get('dataset'),'elapsed_seconds':g.get('elapsed_seconds',0)}),flush=True);last=key
 if c.get('stage')=='complete':print('COMPLETE',flush=True);break
 if read('controller_failure') or read('generation_failure'):print('FAILED',flush=True);break
 if c.get('pid'):
  try:os.kill(c['pid'],0)
  except ProcessLookupError:print('CONTROLLER_NOT_ALIVE: inspect receipts',flush=True);break
 time.sleep(20)
'''
subprocess.run(SSH+['/srv/research/ece285_math_rl_project/.venv/bin/python -B -'],input=REMOTE,text=True,check=True)
