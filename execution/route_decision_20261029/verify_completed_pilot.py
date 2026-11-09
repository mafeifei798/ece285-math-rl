"""Read existing completed receipts/predictions over SSH; no models or mutations remotely."""
from pathlib import Path
import json
import subprocess

REMOTE = r'''
from pathlib import Path
import json, hashlib, collections, datetime
import numpy as np

root=Path('/path/to/ece285_math_rl_project/runs/C_bounded_triplet_seed17_1M_20261029_v2')
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'read_only_remote':True,'model_calls':0,'optimizer_steps':0,'arms':{},'paired':{}}
pred={}; contracts={}
for arm in ['initial','A','C_mix','C_bounded']:
    p=root/'validation'/arm
    f=p/'predictions.jsonl'
    rr=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    s=json.loads((p/'summary.json').read_text())
    contract=json.loads((p/'contract.json').read_text())
    ids=[x['problem_id'] for x in rr]
    assert len(ids)==len(set(ids))==500
    digest=hashlib.sha256(f.read_bytes()).hexdigest()
    assert digest==s['predictions_sha256']
    correct=sum(x['reward'] for x in rr)
    assert correct==s['correct']
    assert all(x['reward'] in [0,1] for x in rr)
    pred[arm]={x['problem_id']:x for x in rr}
    contracts[arm]={k:v for k,v in contract.items() if k not in ['arm','adapter_sha256']}
    out['arms'][arm]={'n':500,'correct':correct,'accuracy':correct/500,
       'predictions_sha256':digest,'summary_matches_saved_labels':True,
       'validation_contract':contracts[arm]}
    if arm!='initial':
        f=root/arm/'rollouts.jsonl'
        train=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        counts=collections.Counter(x['problem_id'] for x in train)
        result=json.loads((root/arm/'final_result.json').read_text())
        out['arms'][arm].update(rollout_count=len(train),unique_training_questions=len(counts),
           responses_per_question_histogram=dict(collections.Counter(counts.values())),
           training=result)
assert len({json.dumps(c,sort_keys=True) for c in contracts.values()})==1
keys=sorted(pred['initial'])
for arm in pred:
    assert set(pred[arm])==set(keys)
    assert all(pred[arm][k]['question_sha256']==pred['initial'][k]['question_sha256'] for k in keys)
rng=np.random.default_rng(20261029)
indices=rng.integers(0,500,size=(10000,500),dtype=np.int32)
for lhs,rhs in [('C_bounded','A'),('C_bounded','C_mix'),('C_bounded','initial'),('C_mix','A'),('A','initial')]:
    delta=np.array([pred[lhs][k]['reward']-pred[rhs][k]['reward'] for k in keys])
    b=delta[indices].mean(axis=1)*100
    out['paired'][lhs+' minus '+rhs]={'delta_pp':float(delta.mean()*100),
       'wins':int((delta>0).sum()),'losses':int((delta<0).sum()),
       'paired_question_bootstrap95_pp':np.quantile(b,[.025,.975]).tolist(),
       'bootstrap_replicates':10000,'bootstrap_seed':20261029}
out['all_four_generation_contracts_equal_except_arm_and_adapter']=True
out['verifier_rerun']=False
out['limitation']='Recounts saved binary labels and checks hashes/contracts. Does not rejudge mathematical correctness or include training-seed uncertainty.'
out['passed']=True
print(json.dumps(out,ensure_ascii=False))
'''

if __name__ == '__main__':
    command=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519',
       '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
       'USER@GPU_HOST','/path/to/ece285_math_rl_project/.venv/bin/python -B -']
    result=subprocess.run(command,input=REMOTE,text=True,capture_output=True,check=True)
    data=json.loads(result.stdout)
    path=Path(__file__).with_name('completed_pilot_independent_check.json')
    assert not path.exists(), 'Preserve prior receipt; do not overwrite.'
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':data['passed'],'paired':data['paired'],
       'unique_training_questions':{a:d.get('unique_training_questions') for a,d in data['arms'].items()},
       'evidence':str(path)},ensure_ascii=False,indent=2))
