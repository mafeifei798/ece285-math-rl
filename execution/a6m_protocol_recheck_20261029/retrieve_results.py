"""Fetch completed evidence once and verify new/reused prediction hashes locally."""
from pathlib import Path
import json,subprocess,tarfile,hashlib,shutil
from inspect_remote import SSH
HERE=Path(__file__).resolve().parent
remote=r'''
from pathlib import Path
import json
p=Path('/path/to/ece285_math_rl_project/runs/A6M_protocol_recheck_20261029')
assert json.loads((p/'controller.json').read_text())['stage']=='complete'
assert json.loads((p/'acceptance.json').read_text())['passed']
print('READY')
'''
s=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=remote,text=True,capture_output=True,check=True)
assert s.stdout.strip()=='READY'
archive=HERE/'completed_evidence.tar.gz';assert not archive.exists(),'Do not overwrite downloaded evidence'
paths=['experiments/a6m_protocol_recheck_20261029','runs/A6M_protocol_recheck_20261029','audits/A6M_protocol_recheck_20261029',
    'data/prepared/math500.jsonl','data/prepared/gsm8k.jsonl','data/prepared/manifest.json','receipts/assets.json','training/baseline_config.json',
    'experiments/reward_repair_v3_20261021/repairs.py','experiments/reward_repair_v2_20261021/v2core.py','experiments/reward_repair_v2_20261021/legacy_snapshot',
    'audits/reward_repair_v3_20261021/unit_metadata.json','runs/Final_four_eval_seed17_1M_20261029/contract.json',
    'runs/Final_four_eval_seed17_1M_20261029/comparison.json','runs/A_corrected_seed17_6M_20261022/evaluation',
    'runs/A_corrected_seed17_6M_20261022/run_config.json']
for arm in ['initial','A']:
 for dataset in ['math500','gsm8k']:paths.append(f'runs/Final_four_eval_seed17_1M_20261029/{arm}/{dataset}.jsonl')
for base in ['runs/A_corrected_seed17_6M_20261022/checkpoint-284','runs/C_bounded_triplet_seed17_1M_20261029_v2/A/final_adapter']:
 for n in ['adapter_model.safetensors','adapter_config.json']:paths.append(base+'/'+n)
paths+=['runs/A_corrected_seed17_6M_20261022/checkpoint-284/completion_accounting.json',
    'runs/A_corrected_seed17_6M_20261022/checkpoint-284/trainer_state.json',
    'runs/C_bounded_triplet_seed17_1M_20261029_v2/A/run_config.json']
with archive.open('wb') as f:
 s=subprocess.run(SSH+['tar -C /path/to/ece285_math_rl_project -czf - '+' '.join(paths)],stdout=f,stderr=subprocess.PIPE)
assert s.returncode==0,s.stderr.decode(errors='replace')
dest=HERE/'evidence';dest.mkdir(exist_ok=False)
with tarfile.open(archive,'r:gz') as tf:
 for item in tf:
  target=dest/item.name
  assert target.resolve().is_relative_to(dest.resolve()) and not item.issym() and not item.islnk()
  if item.isdir():target.mkdir(parents=True,exist_ok=True)
  else:
   assert item.isfile();target.parent.mkdir(parents=True,exist_ok=True)
   with tf.extractfile(item) as inp,target.open('wb') as out:shutil.copyfileobj(inp,out)
run=dest/'runs/A6M_protocol_recheck_20261029';c=json.loads((run/'comparison.json').read_text());contract=json.loads((run/'contract.json').read_text())
count=0;verified={}
for a,base in {'initial':dest/'runs/Final_four_eval_seed17_1M_20261029/initial','A_1M':dest/'runs/Final_four_eval_seed17_1M_20261029/A',
    'A_6M_historical':dest/'runs/A_corrected_seed17_6M_20261022/evaluation','A_6M_recheck':run/'A_6M'}.items():
 for dataset in ['math500','gsm8k']:
  p=base/f'{dataset}.jsonl';digest=hashlib.sha256(p.read_bytes()).hexdigest();expected=c['summary'][a][dataset]
  assert digest==expected['predictions_sha256'];rr=[json.loads(s) for s in p.read_text().splitlines()]
  assert len(rr)==len({r['problem_id'] for r in rr})==expected['n']
  assert sum(r['reward'] for r in rr)==expected['correct']
  assert sum(r['generated_tokens'] for r in rr)==expected['generated_tokens']
  if a=='A_6M_recheck':
   assert all(len(r['completion_token_ids'])==r['generated_tokens'] for r in rr);count+=len(rr)
  verified[str(p.relative_to(HERE))]=digest
adapter=dest/'runs/A_corrected_seed17_6M_20261022/checkpoint-284'
assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==contract['states']['A_6M']['adapter_sha256']
assert hashlib.sha256((adapter/'adapter_config.json').read_bytes()).hexdigest()==contract['states']['A_6M']['adapter_config_sha256']
assert count==1819 and c['acceptance']['frozen_verifier_regraded_records']==7276
receipt={'passed':True,'new_raw_token_records_recounted':count,'total_prediction_records_recounted':7276,'prediction_sha256':verified,
    'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'historical_adapter_hash_checked':True,'new_training':False,'new_generation':False}
if (HERE/'math_case_candidates.json').exists():
 candidates=json.loads((HERE/'math_case_candidates.json').read_text())
 mappings={a:{r['problem_id']:r for r in [json.loads(s) for s in path.read_text().splitlines()]} for a,path in {
     'A_1M':dest/'runs/Final_four_eval_seed17_1M_20261029/A/math500.jsonl',
     'A_6M':run/'A_6M/math500.jsonl'}.items()}
 for case in candidates['cases']:
  for a in ['A_1M','A_6M']:
   for k,v in case[a].items():assert mappings[a][case['problem_id']][k]==v
 receipt['early_read_case_texts_match_completed_predictions']=True
(HERE/'comparison.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
(HERE/'retrieval_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'passed':True,'new_rows':count,'reference_rows':7276-count,'archive_bytes':archive.stat().st_size,'evidence':str(dest)}))
