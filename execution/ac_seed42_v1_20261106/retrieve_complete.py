"""Read completed receipts/predictions and compact signals; never starts GPU work."""
from pathlib import Path
import hashlib, io, json, subprocess, tarfile
from inspect_remote import SSH
HERE=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import io,json,hashlib,tarfile,sys
root=Path('/path/to/ece285_math_rl_project')
run=root/'runs/AC_seed42_V1_20261106'
assert json.loads((run/'controller.json').read_text())['stage']=='complete'
assert json.loads((run/'comparison.json').read_text())['passed']
items={}
names=['comparison.json','controller.json','dispatch.json','started.json','common144_gate.json','stages.jsonl','REPORT.txt']
for step in [144,288]:
 names += [f'evaluation/{step}/{n}.json' for n in ['contract','summary','acceptance']]
 for arm in ['A','C_bounded']:
  names += [f'{arm}/{n}_{step}.json' for n in ['resume_acceptance','optimizer_acceptance','resume_build']]
  names += [f'{arm}/stage_{step}_result.json',f'{arm}/milestones/checkpoint-{step}/pilot_checkpoint.json']
  names += [f'evaluation/{step}/{arm}/{ds}.jsonl' for ds in (['validation'] if step==144 else ['math500','gsm8k'])]
for arm in ['A','C_bounded']:
 names += [f'{arm}/{n}' for n in ['progress.json','last_checkpoint.json','generation_checks.jsonl','update_audits.jsonl','resource_samples.jsonl','final_result.json']]
 rr=[json.loads(s) for s in (run/arm/'rollouts.jsonl').read_text().splitlines() if s.strip()]
 keys=['problem_id','optimizer_step_before_update','completion_tokens','outcome_reward','advantage','outcome_component','process_contribution','min_score','status','valid_for_process_reward','tag','token_mask_sha256']
 items[f'{arm}/compact_signals.jsonl']=''.join(json.dumps({k:r[k] for k in keys if k in r})+'\n' for r in rr).encode()
for n in names:items[n]=(run/n).read_bytes()
for n in ['preflight.json','protected_before.json']:
 items['audit/'+n]=(root/'audits/AC_seed42_V1_20261106'/n).read_bytes()
items['manifest.json']=json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in items.items()},indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tf:
 for n,b in items.items():
  ti=tarfile.TarInfo(n);ti.size=len(b);tf.addfile(ti,io.BytesIO(b))
'''
dest=HERE/'completed_evidence'
assert not dest.exists(),'Preserve downloaded evidence; do not overwrite.'
r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=REMOTE.encode(),capture_output=True)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-4000:])
dest.mkdir()
with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as tf:
 for item in tf:
  p=dest/item.name
  assert item.isfile() and p.resolve().is_relative_to(dest.resolve())
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tf.extractfile(item).read())
manifest=json.loads((dest/'manifest.json').read_text())
for n,h in manifest.items():assert hashlib.sha256((dest/n).read_bytes()).hexdigest()==h,n
(HERE/'retrieval_receipt.json').write_text(json.dumps({'passed':True,'files_verified':len(manifest),'stream_sha256':hashlib.sha256(r.stdout).hexdigest(),'stream_bytes':len(r.stdout),'GPU_work':False,'remote_files_changed':False},indent=2)+'\n')
print(json.dumps({'passed':True,'files':len(manifest),'transfer_MB':len(r.stdout)/1e6,'destination':str(dest)}))
