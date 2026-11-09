"""Read-only collection of completed receipts/predictions; never dispatches."""
from pathlib import Path
import json,base64,hashlib,subprocess,runpy
HERE=Path(__file__).resolve().parent
SSH=runpy.run_path(str(HERE.parents[1]/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
CODE="""from pathlib import Path
import json,hashlib,base64,subprocess
r=Path('/path/to/ece285_math_rl_project');run=r/'runs/Length_norm_pair_seed17_48_20261107'
c=json.loads((run/'controller.json').read_text());assert c['stage']=='complete'
comparison=json.loads((run/'comparison.json').read_text());assert comparison['passed']
result={}
for p in run.rglob('*'):
 if not p.is_file() or p.suffix not in ['.json','.jsonl','.log']:continue
 rel=p.relative_to(run)
 if any(s.startswith('checkpoint-') for s in rel.parts) or 'sampling_adapters' in rel.parts:continue
 if rel.parts[0]=='evaluation' and 'batches' in rel.parts:continue
 if p.stat().st_size>100000000:raise RuntimeError(('Unexpectedly large evidence file',str(rel),p.stat().st_size))
 b=p.read_bytes();result[str(rel)]={'sha256':hashlib.sha256(b).hexdigest(),'content':base64.b64encode(b).decode()}
for p in (r/'audits/Length_norm_pair_seed17_48_20261107').rglob('*'):
 if not p.is_file() or p.suffix not in ['.json','.jsonl','.log']:continue
 if 'sampling_adapters' in p.parts:continue
 b=p.read_bytes();name='admission/'+str(p.relative_to(r/'audits/Length_norm_pair_seed17_48_20261107'))
 result[name]={'sha256':hashlib.sha256(b).hexdigest(),'content':base64.b64encode(b).decode()}
for arm in ['grpo','fixed_length']:
 cp=run/arm/'checkpoint-336';p=cp/'short_pilot_checkpoint.json';value=json.loads(p.read_text())
 for name,h in value['files_sha256'].items():assert hashlib.sha256((cp/name).read_bytes()).hexdigest()==h,('Final checkpoint changed',arm,name)
 b=p.read_bytes();result[arm+'/checkpoint-336/short_pilot_checkpoint.json']={'sha256':hashlib.sha256(b).hexdigest(),'content':base64.b64encode(b).decode()}
p=r/'data/prepared/validation.jsonl';b=p.read_bytes()
assert hashlib.sha256(b).hexdigest()==json.loads((run/'evaluation/contract.json').read_text())['dataset_sha256']
result['admission/validation.jsonl']={'sha256':hashlib.sha256(b).hexdigest(),'content':base64.b64encode(b).decode()}
print(json.dumps({'files':result,'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)}))
"""
def main():
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python','-B','-'],input=CODE,text=True,capture_output=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr)
    x=json.loads(p.stdout);out=HERE/'completed_evidence';out.mkdir(exist_ok=True);manifest={}
    for name,v in x['files'].items():
        relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
        b=base64.b64decode(v['content']);assert hashlib.sha256(b).hexdigest()==v['sha256']
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert dest.read_bytes()==b,('Existing evidence changed',name)
        else:dest.write_bytes(b)
        manifest[name]={'bytes':len(b),'sha256':v['sha256']}
    (HERE/'completed_manifest.json').write_text(json.dumps({'files':manifest,'gpu_after_completion':x['gpu']},indent=2)+'\n')
    print(json.dumps({'files':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'gpu':x['gpu']}))
if __name__=='__main__':main()
