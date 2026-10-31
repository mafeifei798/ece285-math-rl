"""Read-only finite wait for the running three-arm resume acceptance gate."""
from inspect_remote import SSH
import subprocess
remote=r'''
from pathlib import Path
import json,time,os
p=Path('/srv/research/ece285_math_rl_project/runs/ABC_seed17_resume6M_V1_20261030');last=None
def read(f):return json.loads(f.read_text()) if f.exists() else {}
for _ in range(720):
 c=read(p/'controller.json');gate=read(p/'acceptance_gate.json');steps={a:read(p/a/'progress.json').get('global_step',48) for a in ['A','C_mix','C_bounded']}
 key=(c.get('stage'),*steps.values(),bool(gate))
 if key!=last:
  print(json.dumps({'stage':c.get('stage'),'steps':steps,'acceptance_gate':bool(gate.get('passed')),'unix':time.time()}),flush=True);last=key
 if read(p/'controller_failure.json'):
  print('FAILED: '+json.dumps(read(p/'controller_failure.json')),flush=True);break
 if gate.get('passed') and max(steps.values())>50:print('ACCEPTED_AND_LONG_CONTINUATION_STARTED',flush=True);break
 if c.get('stage')=='complete':print('COMPLETE',flush=True);break
 time.sleep(20)
'''
subprocess.run(SSH+['/srv/research/ece285_math_rl_project/.venv/bin/python -B -'],input=remote,text=True,check=True)
