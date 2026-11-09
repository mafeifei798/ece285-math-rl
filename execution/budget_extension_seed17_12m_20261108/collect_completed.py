from pathlib import Path
import json,hashlib,zlib,base64,runpy,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
CODE=r'''
from pathlib import Path
import json,hashlib,base64,zlib,sys,subprocess
r=Path('/path/to/ece285_math_rl_project');run=r/'runs/Budget_extension_seed17_12M_20261108';audit=r/'audits/Budget_extension_seed17_12M_20261108'
assert json.loads((run/'controller.json').read_text())['stage']=='complete'
assert json.loads((audit/'independent_acceptance.json').read_text())['passed']
files={}
for base,prefix in [(run,''),(audit,'admission/')]:
 for p in base.rglob('*'):
  if not p.is_file() or p.suffix not in ['.json','.jsonl','.log','.txt']:continue
  rel=p.relative_to(base)
  if 'sampling_adapters' in rel.parts or any(x.startswith('checkpoint-') for x in rel.parts):continue
  if 'evaluation' in rel.parts and 'batches' in rel.parts:continue
  b=p.read_bytes();files[prefix+str(rel)]={'sha256':hashlib.sha256(b).hexdigest(),'content':base64.b64encode(b).decode()}
for arm in ['A','C_bounded']:
 p=r/'runs/ABC_seed17_resumed_V1_20261104'/arm/'checkpoint-288/completion_accounting.json';b=p.read_bytes()
 files[f'admission/source_accounting_{arm}.json']={'sha256':hashlib.sha256(b).hexdigest(),'content':base64.b64encode(b).decode()}
 for step in [432,576]:
  p=run/arm/f'milestones/checkpoint-{step}/pilot_checkpoint.json';b=p.read_bytes()
  files[str(p.relative_to(run))]={'sha256':hashlib.sha256(b).hexdigest(),'content':base64.b64encode(b).decode()}
for dataset in ['train','validation','math500','gsm8k']:
 p=r/f'data/prepared/{dataset}.jsonl';b=p.read_bytes()
 files[f'datasets/{dataset}.jsonl']={'sha256':hashlib.sha256(b).hexdigest(),'content':base64.b64encode(b).decode()}
out={'files':files,'GPU_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)}
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode(),6))
'''
def main():
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=CODE.encode(),capture_output=True,timeout=300)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-6000:])
    x=json.loads(zlib.decompress(p.stdout));out=HERE/'completed_evidence';out.mkdir(exist_ok=True);manifest={}
    for name,value in x['files'].items():
        rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
        b=base64.b64decode(value['content']);assert hashlib.sha256(b).hexdigest()==value['sha256']
        path=out/rel;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==b
        else:path.write_bytes(b)
        manifest[name]={'bytes':len(b),'sha256':value['sha256']}
    (HERE/'completed_manifest.json').write_text(json.dumps({'files':manifest,'GPU_apps_after_completion':x['GPU_apps']},indent=2)+'\n')
    print(json.dumps({'files':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'GPU_apps':x['GPU_apps']}))
if __name__=='__main__':main()
