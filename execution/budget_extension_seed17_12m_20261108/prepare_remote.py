from pathlib import Path
import base64,hashlib,json,runpy,subprocess,zlib,ast
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
def main():
    source=ROOT/'experiments/budget_extension_seed17_12m_20261108'
    for p in source.glob('*.py'):ast.parse(p.read_text())
    old=json.loads((source/'release.json').read_text())
    old['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file() and p.name!='release.json'}
    (source/'release.json').write_text(json.dumps(old,indent=2)+'\n')
    files={p.name:p.read_text() for p in source.iterdir() if p.is_file()}
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    code=r'''
from pathlib import Path
import json,hashlib,base64,zlib,os,psutil
r=Path('/path/to/ece285_math_rl_project');s=r/'experiments/budget_extension_seed17_12m_20261108';run=r/'runs/Budget_extension_seed17_12M_20261108';audit=r/'audits/Budget_extension_seed17_12M_20261108'
assert not s.exists() and not run.exists(),'Existing release or job: inspect, never overwrite'
diag=json.loads((audit/'placeholder_diagnostic/result.json').read_text());assert diag['passed'] and diag['optimizer_updates']==0
for proc in psutil.process_iter(['pid','cmdline','status']):
 try:
  c=proc.info['cmdline'] or []
  assert not (any(str(r) in x for x in c) and any('controller.py' in x or 'train_arm.py' in x for x in c) and proc.info['status']!=psutil.STATUS_ZOMBIE),('Active project process',proc.pid)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)));release=json.loads(files['release.json'])
for n,h in release['dependencies'].items():assert sha(r/n)==h,('Dependency changed',n)
source={}
for arm in ('A','C_bounded'):
 cp=r/'runs/ABC_seed17_resumed_V1_20261104'/arm/'checkpoint-288'
 names=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin','completion_accounting.json','pilot_checkpoint.json']
 source[arm]={n:sha(cp/n) for n in names}
 for n,h in json.loads((cp/'pilot_checkpoint.json').read_text())['files_sha256'].items():assert sha(cp/n)==h
protected=json.loads((r/'audits/ABC_seed17_resumed_V1_20261104/protected_before.json').read_text())
for base in [r/'experiments',r/'runs/ABC_seed17_resumed_V1_20261104',r/'runs/AC_seed42_V1_20261106',r/'runs/Tree_credit_triplet_seed17_48_20261108']:
 for p in base.rglob('*'):
  if not p.is_file() or p.suffix not in ['.py','.json','.jsonl','.txt','.log']:continue
  if any(x.startswith('checkpoint-') or x=='sampling_adapters' for x in p.relative_to(base).parts):continue
  name=str(p.relative_to(r));h=sha(p)
  if name in protected:assert protected[name]==h,('Historical hash mismatch',name)
  protected[name]=h
for name,h in protected.items():assert sha(r/name)==h,name
for p in [s,run]:p.mkdir(parents=True)
for arm in ['A','C_bounded']:(run/arm).mkdir()
for n,text in files.items():(s/n).write_text(text)
for n,h in release['files'].items():assert sha(s/n)==h
(audit/'source_checkpoints.json').write_text(json.dumps(source,indent=2)+'\n')
(audit/'protected_before.json').write_text(json.dumps(protected,indent=2)+'\n')
x={'prepared':True,'GPU_started':False,'release_sha256':sha(s/'release.json'),'source_files':len(files),'dependencies':len(release['dependencies']),
 'protected_files':len(protected),'source_checkpoints':source,'source':str(s),'run':str(run)}
(audit/'prepared.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
'''.replace('PAYLOAD',repr(payload))
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=code,text=True,capture_output=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr[-6000:])
    value=json.loads(p.stdout);(HERE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k!='source_checkpoints'},indent=2))
if __name__=='__main__':main()
