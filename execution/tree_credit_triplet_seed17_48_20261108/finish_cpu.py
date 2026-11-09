"""Run the prepared independent CPU closeout only after controller completion."""
from pathlib import Path
import json,base64,zlib,hashlib,runpy,subprocess
HERE=Path(__file__).resolve().parent
SSH=runpy.run_path(str(HERE.parents[1]/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']


def main():
    files={n:(HERE/n).read_text() for n in ('independent_cpu_verify.py','mechanism_diagnostics.py','verifier_replay.py')}
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    code=r'''
from pathlib import Path
import os,json,subprocess,hashlib,base64,zlib,time,fcntl
root=Path('/path/to/ece285_math_rl_project')
run=root/'runs/Tree_credit_triplet_seed17_48_20261108';audit=root/'audits/Tree_credit_triplet_seed17_48_20261108'
assert json.loads((run/'controller.json').read_text())['stage']=='complete'
assert json.loads((run/'comparison.json').read_text())['passed']
lock=(audit/'cpu_closeout.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(audit/'cpu_closeout_running.json').write_text(json.dumps({'pid':os.getpid(),'unix':time.time(),'scope':'CPU only; no training dispatch'})+'\n')
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
receipts=[]
for name,text in files.items():
 path=audit/name;raw=text.encode();h=hashlib.sha256(raw).hexdigest()
 if path.exists():assert path.read_bytes()==raw,'Audit helper changed; preserve earlier version'
 else:path.write_bytes(raw)
 expected={'independent_cpu_verify.py':'independent_tensor_acceptance.json','mechanism_diagnostics.py':'mechanism_diagnostics.json','verifier_replay.py':'verifier_replay.json'}[name]
 if (audit/expected).exists():
  assert json.loads((audit/expected).read_text())['passed']
  receipts.append({'name':name,'sha256':h,'reused_completed_result':True});continue
 start=time.time();p=subprocess.run([str(root/'.venv/bin/python'),'-B',str(path)],env=env,cwd=root,capture_output=True,text=True)
 (audit/(name+'.log')).write_text(p.stdout+p.stderr)
 receipt={'name':name,'sha256':h,'started_unix':start,'seconds':time.time()-start,'exit_code':p.returncode};receipts.append(receipt)
 if p.returncode:
  (audit/'cpu_closeout_failure.json').write_text(json.dumps(receipts,indent=2)+'\n')
  raise RuntimeError(name+' failed: '+p.stderr[-4000:])
 assert json.loads((audit/expected).read_text())['passed']
out={'passed':True,'GPU_started':False,'receipts':receipts}
(audit/'cpu_closeout_receipt.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
'''.replace('PAYLOAD',repr(payload))
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=code,text=True,capture_output=True,timeout=600)
    if p.returncode:raise RuntimeError(p.stderr[-5000:])
    value=json.loads(p.stdout);(HERE/'cpu_closeout_receipt.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))


if __name__=='__main__':main()
