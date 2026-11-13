"""CPU-only descriptive audit of frozen predictions and actual training signals."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
inputs = {}

def read_rows(path):
    raw = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return [json.loads(s) for s in raw.splitlines() if s]

def corr(x, y):
    if len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    x = [a - mx for a in x]
    y = [a - my for a in y]
    den = math.sqrt(sum(a*a for a in x)*sum(b*b for b in y))
    return sum(a*b for a,b in zip(x,y))/den if den else None

def ranks(x):
    out = [0.0]*len(x)
    ids = sorted(range(len(x)), key=x.__getitem__)
    i = 0
    while i < len(ids):
        j = i+1
        while j < len(ids) and x[ids[j]] == x[ids[i]]:
            j += 1
        for k in range(i,j):
            out[ids[k]] = (i+j-1)/2
        i = j
    return out

result = {'GPU_used': False, 'training_started': False, 'evaluations': {}, 'signals': {}}
for seed,folder in [(17,'abc_seed17_resumed_v1_20261104'),(42,'ac_seed42_v1_20261106')]:
    base = ROOT/'execution'/folder/'completed_evidence'
    for arm in ['A','C_bounded']:
        for dataset in ['math500','gsm8k']:
            rows = read_rows(base/'evaluation'/'288'/arm/(dataset+'.jsonl'))
            missing = [r for r in rows if r['status']=='missing_or_invalid_box']
            result['evaluations'][f'{seed}/{arm}/{dataset}'] = {
                'n':len(rows), 'truncated_by_finish_reason':sum(r['finish_reason']!='stop' for r in rows),
                'missing_box_status':len(missing),
                'missing_box_nonstop':sum(r['finish_reason']!='stop' for r in missing),
                'missing_box_at_length_cap':sum(r['generated_tokens']>=2048 for r in missing),
                'missing_box_max_tokens':max((r['generated_tokens'] for r in missing), default=None)}
    rows = read_rows(base/'C_bounded'/'compact_signals.jsonl')
    groups = [rows[i:i+8] for i in range(0,len(rows),8)]
    assert all(len(g)==8 and len({(r['problem_id'],r['optimizer_step_before_update']) for r in g})==1 for g in groups)
    for kind in ['mixed','all_correct']:
        selected = [g for g in groups if (0<sum(r['outcome_reward'] for r in g)<8 if kind=='mixed' else sum(r['outcome_reward'] for r in g)==8)]
        correct = [[r for r in g if r['outcome_reward']==1] for g in selected]
        flat = [r for g in correct for r in g]
        x,y,rx,ry = [],[],[],[]
        support=Counter()
        for g in correct:
            support[len(g)] += 1
            if len(g)<2:
                continue
            lx=[r['completion_tokens'] for r in g]; my=[r['min_score'] for r in g]
            x.extend(v-mean(lx) for v in lx); y.extend(v-mean(my) for v in my)
            a,b=ranks(lx),ranks(my)
            rx.extend(v-mean(a) for v in a); ry.extend(v-mean(b) for v in b)
        pairing={}
        for ratio in [1.1,1.25]:
            any_groups=eligible_rows=0
            for g in correct:
                eligible=[any(j!=i and max(r['completion_tokens'],s['completion_tokens'])/max(1,min(r['completion_tokens'],s['completion_tokens']))<=ratio and r['min_score']!=s['min_score'] for j,s in enumerate(g)) for i,r in enumerate(g)]
                any_groups += any(eligible); eligible_rows += sum(eligible)
            pairing[str(ratio)]={'groups_with_possible_nontrivial_pair':any_groups,'correct_rows_with_possible_partner':eligible_rows}
        result['signals'][f'{seed}/{kind}']={
            'groups':len(selected),'correct_rows':len(flat),'correct_count_histogram':dict(support),
            'global_spearman':corr(ranks([r['completion_tokens'] for r in flat]),ranks([r['min_score'] for r in flat])),
            'within_group_centered_pearson':corr(x,y),
            'within_group_rank_correlation':corr(rx,ry),
            'absolute_process_contribution':sum(abs(r['process_contribution']) for r in flat),
            'length_match_feasibility_only':pairing}
result['input_sha256']=inputs
result['limits']=[
    'Associations are descriptive and conditional on answer correctness; not causal process validity.',
    'No significance test treating responses or pairs as independent; repeated problems and policy updates are dependent.',
    'Length ratios are illustrative feasibility checks, not selected training thresholds or a disjoint permutation algorithm.',
    'Marginal and within-group correlations cannot exclude nonlinear or format/step-count confounding.',
    'Frozen verifier statuses are read unchanged, not newly graded.']
(HERE/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2,ensure_ascii=False))
