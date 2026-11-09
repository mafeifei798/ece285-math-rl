"""Final read-only process/GPU evidence after all required results exist."""
from pathlib import Path
import json,runpy,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
CODE=r'''
from pathlib import Path
import json,time,psutil,subprocess
r=Path('/path/to/ece285_math_rl_project');p=r/'runs/Budget_extension_seed17_12M_20261108'
c=json.loads((p/'controller.json').read_text());assert c['stage']=='complete'
alive=[]
for pid in [c['pid']]+[x['pid'] for x in c.get('children',[])]:
 try:
  q=psutil.Process(pid)
  if q.status()!=psutil.STATUS_ZOMBIE and any('budget_extension_seed17_12m_20261108' in s for s in q.cmdline()):alive.append(pid)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
assert not alive
print(json.dumps({'passed':True,'unix':time.time(),'our_live_processes':alive,
 'GPU_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
 'controller_stage':'complete','new_training_dispatched':False}))
'''
def main():
    assert json.loads((HERE/'completion_verification.json').read_text())['passed']
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=CODE,text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr[-5000:])
    x=json.loads(p.stdout);(HERE/'runtime_closeout.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
if __name__=='__main__':main()
