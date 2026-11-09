"""Copy only completed evaluation artifacts and source; verify every prediction hash."""
from pathlib import Path
import subprocess,json,tarfile,shutil,hashlib
HERE=Path(__file__).resolve().parent
SSH=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','USER@GPU_HOST']
REMOTE=r'''
from pathlib import Path
import json
r=Path('/path/to/ece285_math_rl_project/runs/Final_four_eval_seed17_1M_20261029')
assert json.loads((r/'controller.json').read_text())['stage']=='complete'
assert json.loads((r/'acceptance.json').read_text())['passed']
print('READY')
'''
r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=REMOTE,text=True,capture_output=True,check=True)
assert r.stdout.strip()=='READY'
archive=HERE/'completed_evidence.tar.gz';assert not archive.exists(),'Preserve prior downloaded evidence'
paths=['experiments/final_four_eval_20261029','audits/Final_four_eval_seed17_1M_20261029',
       'runs/Final_four_eval_seed17_1M_20261029',
       'data/prepared/manifest.json','data/prepared/math500.jsonl','data/prepared/gsm8k.jsonl',
       'data/prepared/train.jsonl','data/prepared/validation.jsonl','receipts/assets.json',
       'training/baseline_config.json','audits/reward_repair_v3_20261021/unit_metadata.json',
       'experiments/reward_repair_v3_20261021/repairs.py','experiments/reward_repair_v2_20261021/v2core.py',
       'experiments/reward_repair_v2_20261021/legacy_snapshot']
for arm in ['A','C_mix','C_bounded']:
 for filename in ['adapter_model.safetensors','adapter_config.json','final_result.json','run_config.json','metrics.jsonl','update_audits.jsonl','generation_checks.jsonl','rollouts.jsonl']:
  suffix='final_adapter/'+filename if filename.startswith('adapter_') else filename
  paths.append('runs/C_bounded_triplet_seed17_1M_20261029_v2/'+arm+'/'+suffix)
with archive.open('wb') as f:
 r=subprocess.run(SSH+['tar -C /path/to/ece285_math_rl_project -czf - '+' '.join(paths)],stdout=f,stderr=subprocess.PIPE)
assert r.returncode==0,r.stderr.decode(errors='replace')
dest=HERE/'evidence';dest.mkdir(exist_ok=False)
with tarfile.open(archive,'r:gz') as tf:
 for item in tf:
  target=dest/item.name
  assert target.resolve().is_relative_to(dest.resolve()) and not item.issym() and not item.islnk()
  if item.isdir():target.mkdir(parents=True,exist_ok=True)
  else:
   assert item.isfile();target.parent.mkdir(parents=True,exist_ok=True)
   with tf.extractfile(item) as inp,target.open('wb') as out:shutil.copyfileobj(inp,out)
run=dest/'runs/Final_four_eval_seed17_1M_20261029'
comparison=json.loads((run/'comparison.json').read_text());assert comparison['acceptance']['regraded']==7276
verified={};total=0
for arm in ['initial','A','C_mix','C_bounded']:
 for dataset in ['math500','gsm8k']:
  p=run/arm/f'{dataset}.jsonl';digest=hashlib.sha256(p.read_bytes()).hexdigest()
  expected=comparison['summary'][arm][dataset];assert digest==expected['predictions_sha256']
  rr=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  assert len(rr)==len({x['problem_id'] for x in rr})==expected['n']
  assert sum(x['reward'] for x in rr)==expected['correct']
  assert sum(x['generated_tokens'] for x in rr)==expected['generated_tokens']
  assert all(x['generated_tokens']==len(x['completion_token_ids']) for x in rr)
  total+=len(rr);verified[str(p.relative_to(HERE))]=digest
receipt={'passed':True,'local_prediction_records_recounted':total,'prediction_sha256':verified,
 'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'remote_verifier_regrade':comparison['acceptance']['regraded'],
 'scope':'Local recount/hash/token verification; mathematical regrading happened independently in remote CPU audit.',
 'new_training':False,'new_generation':False}
if (HERE/'math_case_candidates.json').exists():
 candidates=json.loads((HERE/'math_case_candidates.json').read_text())
 final_math={a:{r['problem_id']:r for r in [json.loads(s) for s in (run/a/'math500.jsonl').read_text().splitlines()]} for a in ['initial','A','C_mix','C_bounded']}
 for case in candidates['cases']:
  for arm,old in case['outputs'].items():
   final=final_math[arm][case['problem_id']]
   for k in ['completion','reward','status','answer','question_sha256','adapter_sha256']:assert final[k]==old[k]
 receipt['mathematical_review_case_texts_match_final_predictions']=True
if (HERE/'gsm_case_candidates.json').exists():
 candidates=json.loads((HERE/'gsm_case_candidates.json').read_text())
 final_gsm={a:{r['problem_id']:r for r in [json.loads(s) for s in (run/a/'gsm8k.jsonl').read_text().splitlines()]} for a in ['initial','A','C_mix','C_bounded']}
 for case in candidates['cases']:
  for arm,old in case['outputs'].items():
   final=final_gsm[arm][case['problem_id']]
   for k in ['completion','reward','status','answer','question_sha256','adapter_sha256']:assert final[k]==old[k]
 receipt['GSM_review_case_texts_match_final_predictions']=True
contract=json.loads((run/'contract.json').read_text());receipt['local_adapter_sha256']={}
manifest=json.loads((dest/'data/prepared/manifest.json').read_text())
for name,item in manifest['files'].items():
 assert hashlib.sha256((dest/'data/prepared'/name).read_bytes()).hexdigest()==item['sha256']
receipt['prepared_data_hashes_verified']=True
for arm in ['A','C_mix','C_bounded']:
 adapter=dest/'runs/C_bounded_triplet_seed17_1M_20261029_v2'/arm/'final_adapter'
 for filename,key in [('adapter_model.safetensors','adapter_sha256'),('adapter_config.json','adapter_config_sha256')]:
  digest=hashlib.sha256((adapter/filename).read_bytes()).hexdigest()
  assert digest==contract['states'][arm][key]
  receipt['local_adapter_sha256'][str((adapter/filename).relative_to(HERE))]=digest
(HERE/'retrieval_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
(HERE/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':True,'rows':total,'archive_bytes':archive.stat().st_size,'evidence':str(dest)},ensure_ascii=False))
