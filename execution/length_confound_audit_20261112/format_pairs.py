"""Post-hoc paired format strata; CPU only, no regrading or model imports."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
common = module('common_cpu',ROOT/'training/common.py')
order = module('order_cpu',ROOT/'execution/answer_order_audit_20261108/audit.py')
inputs = {}
def rows(path):
    raw=path.read_bytes()
    inputs[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.splitlines() if line]
def eligible(r):
    answer=common.last_boxed(r['completion'])
    return r['finish_reason']=='stop' and answer is not None and len(answer)<=2048
out={'GPU_used':False,'regraded':False,'strata':{},'missing_details':{}}
for seed,folder in [(17,'abc_seed17_resumed_v1_20261104'),(42,'ac_seed42_v1_20261106')]:
    base=ROOT/'execution'/folder/'completed_evidence/evaluation/288'
    for dataset in ['math500','gsm8k']:
        ar=rows(base/'A'/f'{dataset}.jsonl');cr=rows(base/'C_bounded'/f'{dataset}.jsonl')
        assert len(ar)==len(cr)
        stats={}
        for a,c in zip(ar,cr):
            assert a['problem_id']==c['problem_id'] and a['gold']==c['gold']
            ka,kc=eligible(a),eligible(c)
            key=f'A_eligible={ka}/C_eligible={kc}'
            t=stats.setdefault(key,Counter())
            t['n']+=1;t['A_correct']+=int(a['reward']);t['C_correct']+=int(c['reward'])
            t['C_wins']+=int(c['reward']>a['reward']);t['C_losses']+=int(c['reward']<a['reward'])
        for t in stats.values():
            t['net_C_minus_A']=t['C_correct']-t['A_correct']
            t['delta_pp_within_stratum']=100*t['net_C_minus_A']/t['n']
            t['contribution_pp_full_dataset']=100*t['net_C_minus_A']/len(ar)
        out['strata'][f'{seed}/{dataset}']=stats
        for arm,rr in [('A',ar),('C_bounded',cr)]:
            missing=[r for r in rr if r['status']=='missing_or_invalid_box']
            descriptions=[order.describe(r) for r in missing]
            out['missing_details'][f'{seed}/{arm}/{dataset}']={
                'n':len(missing),
                'first_placeholder':sum(d['first_box_placeholder'] for d in descriptions),
                'finish_reasons':dict(Counter(r['finish_reason'] for r in missing)),
                'last_token_ids':dict(Counter(r['completion_token_ids'][-1] for r in missing if r['completion_token_ids'])),
                'deterministic_first5_endings':[{'problem_id':r['problem_id'],'ending':r['completion'][-280:]} for r in missing[:5]]}
out['eligibility']='stop plus final complete nonempty box <=2048 chars, exactly the verifier pre-equivalence gate; does not require correct or symbolically parseable'
out['limits']=['Post-treatment selection: strata do not establish a causal effect of fixing output format.',
               'Counts use original correctness labels, not relaxed extraction.',
               'First five endings are illustrative in file order, not population-level manual labels.']
out['inputs_sha256']=inputs
(HERE/'format_pairs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'strata':out['strata'],'missing':{k:{a:b for a,b in v.items() if a!='deterministic_first5_endings'} for k,v in out['missing_details'].items()}},indent=2))
