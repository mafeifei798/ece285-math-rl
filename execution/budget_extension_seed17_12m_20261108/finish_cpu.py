from pathlib import Path
import json,runpy,subprocess,base64,zlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
def main():
    content=base64.b64encode(zlib.compress((HERE/'independent_acceptance.py').read_bytes())).decode()
    code=r'''
from pathlib import Path
import json,subprocess,base64,zlib,os,fcntl,time
r=Path('/path/to/ece285_math_rl_project');run=r/'runs/Budget_extension_seed17_12M_20261108';audit=r/'audits/Budget_extension_seed17_12M_20261108'
assert json.loads((run/'controller.json').read_text())['stage']=='complete'
lock=(audit/'CPU_closeout.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (audit/'independent_acceptance.json').exists(),'Already completed: collect it, do not rerun'
p=audit/'independent_acceptance.py';data=zlib.decompress(base64.b64decode(PAYLOAD))
if p.exists():assert p.read_bytes()==data
else:p.write_bytes(data)
(audit/'CPU_running.json').write_text(json.dumps({'pid':os.getpid(),'unix':time.time()})+'\n')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
start=time.time();proc=subprocess.run([str(r/'.venv/bin/python'),'-B',str(p)],cwd=r,env=env,capture_output=True,text=True)
(audit/'independent_acceptance.log').write_text(proc.stdout+proc.stderr)
assert proc.returncode==0,proc.stderr[-5000:]
assert json.loads((audit/'independent_acceptance.json').read_text())['passed']
print(json.dumps({'passed':True,'seconds':time.time()-start,'GPU_started':False}))
'''.replace('PAYLOAD',repr(content))
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=code,text=True,capture_output=True,timeout=900)
    if p.returncode:raise RuntimeError(p.stderr[-6000:])
    x=json.loads(p.stdout);(HERE/'CPU_receipt.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
if __name__=='__main__':main()
