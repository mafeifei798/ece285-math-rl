"""Make a fixed model-identity-blinded diagnostic sample; do not inspect the key before scoring."""
from pathlib import Path
import json, random, hashlib
HERE=Path(__file__).resolve().parent
OUT=HERE/'blind_audit_20261105';OUT.mkdir(exist_ok=False)
E=HERE/'completed_evidence/evaluation/288'
Q=HERE.parent/'final_four_eval_20261029/evidence/data/prepared'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
rng=random.Random(20261105);selected=[];source_hashes={}
for ds in ['math500','gsm8k']:
 qs={r['problem_id']:r for r in rows(Q/f'{ds}.jsonl')}
 aa=rows(E/f'A/{ds}.jsonl');bb=rows(E/f'C_bounded/{ds}.jsonl')
 for direction in ['win','loss']:
  pool=[(a,b) for a,b in zip(aa,bb) if (b['reward']>a['reward'] if direction=='win' else b['reward']<a['reward'])]
  for a,b in rng.sample(pool,5):selected.append((ds,direction,qs[a['problem_id']],a,b))
 for arm in ['A','C_bounded']:
  p=E/arm/f'{ds}.jsonl';source_hashes[f'{arm}/{ds}']=hashlib.sha256(p.read_bytes()).hexdigest()
rng.shuffle(selected);key=[];blind=[]
for i,(ds,direction,q,a,b) in enumerate(selected,1):
 pair=[('A',a),('C_bounded',b)];rng.shuffle(pair);case=f'R{i:02d}'
 item={'case_id':case,'problem':q['problem'],'gold':q['answer'],'outputs':{label:r['completion'] for label,(_,r) in zip(['X','Y'],pair)}}
 blind.append(item)
 key.append({'case_id':case,'dataset':ds,'direction':direction,'problem_id':q['problem_id'],'outputs':{label:{'arm':arm,'reward':r['reward'],'status':r['status'],'finish_reason':r['finish_reason']} for label,(arm,r) in zip(['X','Y'],pair)}})
(OUT/'blind.json').write_text(json.dumps(blind,ensure_ascii=False,indent=2)+'\n')
(OUT/'sealed_key.json').write_text(json.dumps(key,ensure_ascii=False,indent=2)+'\n')
plan={'fixed_sample_seed':20261105,'pairs':20,'responses':40,'sampling':'5 C_bounded wins and 5 losses per dataset; cases and X/Y assignments shuffled.',
 'not_population_estimate':True,'reviewer':'assistant; model identity and stored verifier labels hidden, not independent human reviewers',
 'assess':['independently solve problem','check mathematical validity of both outputs','check final answer against body','flag suspected verifier/format issues; uncertain stays uncertain'],
 'gate':'Do not launch seed42 if material unexplained grading/data/code problems are found. Freeze annotations before unblinding. Known output format failures are reported under unchanged verifier. Sampling cannot prove absence of all problems.',
 'source_sha256':source_hashes,'key_sha256':hashlib.sha256((OUT/'sealed_key.json').read_bytes()).hexdigest()}
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
for start in range(0,20,5):
 (OUT/f'packet_{start//5+1}.txt').write_text('\n\n'.join(f"CASE {r['case_id']}\nPROBLEM: {r['problem']}\nGOLD: {r['gold']}\nX:\n{r['outputs']['X']}\nY:\n{r['outputs']['Y']}" for r in blind[start:start+5]))
print(json.dumps({'pairs':20,'responses':40,'path':str(OUT),'key_not_printed':True}))
