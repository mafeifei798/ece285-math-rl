"""Frozen-C-PRM paired placeholder deletion; no policy generation or updates."""
from pathlib import Path
import json,sys,time,hashlib,random,fcntl,os
from collections import defaultdict
import numpy as np
import torch
ROOT=Path('/path/to/ece285_math_rl_project')
OUT=ROOT/'audits/Budget_extension_seed17_12M_20261108/placeholder_diagnostic'
sys.path.insert(0,str(OUT))
from answer_order_audit import boxes,describe
sys.path.insert(0,str(ROOT/'experiments/c_bounded_integration_20261029_v2'))
from c_prm import FrozenCScorer
from c_rewards import CanonicalPRMAdapter,repaired_outcome
sys.path.insert(0,str(ROOT/'experiments/abc_seed17_resumed_v1_20261104'))
from advantages import compute_advantages

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def edited(text):
    bb=boxes(text)
    if not bb or not bb[0]['placeholder']:return text
    b=bb[0]
    if len(''.join(text[:b['start']].split()))>20:return text
    # Delete exactly the placeholder span. Never remove a line containing reasoning.
    return text[:b['start']]+text[b['end']:]
def aligned(a,b,sa,sb):
    def spans(text,scores):
        d=defaultdict(list)
        for p,v in zip(scores['step_raw_spans'],scores['step_scores']):
            d[text[p['raw_start']:p['raw_end']].strip()].append(v)
        return d
    x,y=spans(a,sa),spans(b,sb);delta=[]
    for text,values in x.items():
        for l,r in zip(values,y.get(text,[])):delta.append(l-r)
    return delta

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'result.json').exists() and not (OUT/'started.json').exists()
    save(OUT/'started.json',{'pid':os.getpid(),'unix':time.time(),'no_training':True})
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
    test=read(OUT/'sample200.json')
    train={r['problem_id']:r for r in rows(ROOT/'data/prepared/train.jsonl')}
    path=ROOT/'runs/ABC_seed17_resumed_V1_20261104/C_bounded/rollouts.jsonl'
    rr=rows(path);assert len(rr)==9216
    candidates=[]
    for offset in range(0,len(rr),8):
        group=rr[offset:offset+8];assert len({r['problem_id'] for r in group})==1
        good=[r for r in group if r['outcome_reward']==1]
        flags=[edited(r['completion'])!=r['completion'] for r in good]
        if len(good)>=2 and any(flags) and not all(flags):candidates.append(offset)
    assert len(candidates)>=20
    selected=sorted(random.Random(2851108).sample(candidates,20))
    groups=[rr[k:k+8] for k in selected]
    save(OUT/'group_selection.json',{'source':str(path),'sha256':sha(path),'eligible_groups':len(candidates),
        'selected_offsets':selected,'selection':'targeted20 groups with>=2 correct, both placeholder/nonplaceholder among correct; not a population estimate'})
    requests=[];meta=[]
    for sample in test:
        a=sample['completion'];b=edited(a);assert a!=b
        requests.extend([(sample['question'],a),(sample['question'],b)])
        meta.append((sample,a,b))
    group_indices=[]
    for group in groups:
        indices=[]
        for row in group:
            if not row['outcome_reward']:indices.append(None);continue
            q=train[row['problem_id']]['problem'];a=row['completion'];b=edited(a)
            i=len(requests);requests.append((q,a))
            j=i
            if b!=a:j=len(requests);requests.append((q,b))
            indices.append((i,j))
        group_indices.append(indices)
    for sample,a,b in meta:
        for text in (a,b):
            v=repaired_outcome(text,sample['gold'],True,question=sample['question'],problem_id=sample['problem_id'],unit_metadata=units)
            assert v['reward']==1,'Counterfactual changed fixed-verifier correctness'
    started=time.time();scorer=FrozenCScorer(receipt_dir=OUT);wrapper=CanonicalPRMAdapter(scorer)
    torch.cuda.reset_peak_memory_stats();scored=[];costs=[]
    for start in range(0,len(requests),16):
        chunk=requests[start:start+16]
        values,cost=wrapper.score([q for q,t in chunk],[t for q,t in chunk])
        scored.extend(values);costs.append(cost)
        save(OUT/'progress.json',{'done':len(scored),'total':len(requests),'seconds':time.time()-started,'pid':os.getpid()})
    comparisons=[]
    for k,(sample,a,b) in enumerate(meta):
        sa,sb=scored[k*2:k*2+2];unchanged=aligned(a,b,sa,sb)
        comparisons.append({'problem_id':sample['problem_id'],'original':sa,'deleted':sb,
            'min_original_minus_deleted':sa['min_score']-sb['min_score'],
            'unchanged_step_pairs':len(unchanged),'mean_unchanged_step_score_delta':float(np.mean(unchanged)) if unchanged else None})
    differences=np.array([r['min_original_minus_deleted'] for r in comparisons])
    rng=np.random.default_rng(2851108)
    boot=differences[rng.integers(0,len(differences),size=(10000,len(differences)))].mean(1)
    replay=[]
    for group,indices in zip(groups,group_indices):
        y=[r['outcome_reward'] for r in group];before=[scored[k[0]]['min_score'] if k else 0 for k in indices]
        after=[scored[k[1]]['min_score'] if k else 0 for k in indices]
        left=compute_advantages(y,before);right=compute_advantages(y,after)
        assert not ((torch.tensor(y)==0)&(left['advantages']>0)).any()
        assert not ((torch.tensor(y)==0)&(right['advantages']>0)).any()
        replay.append({'problem_id':group[0]['problem_id'],'outcomes':y,'placeholder_removed':[bool(k and k[0]!=k[1]) for k in indices],
            'original_minima':before,'deleted_minima':after,
            'original_advantages':left['advantages'].tolist(),'deleted_advantages':right['advantages'].tolist(),
            'advantage_sign_changes':int(((left['advantages']>0)!=(right['advantages']>0)).sum()),
            'correct_process_rank_changed':sorted([i for i in range(8) if y[i]],key=lambda i:before[i])!=sorted([i for i in range(8) if y[i]],key=lambda i:after[i]),
            'max_abs_advantage_change':float((left['advantages']-right['advantages']).abs().max())})
    scorer.assert_frozen()
    save(OUT/'scores.json',{'comparisons':comparisons,'group_replay':replay})
    result={'passed':True,'policy_generation_calls':0,'optimizer_updates':0,'PRM_frozen':True,
        'PRM_adapter_sha256':'4917be38966c8438f58e283eb06c7fa818fed2bb13b726152078c19af013c356',
        'paired_test_responses':200,'mean_min_original_minus_deleted':float(differences.mean()),
        'paired_mean_bootstrap95CI':np.quantile(boot,[.025,.975]).tolist(),
        'positive_gt1e6':int((differences>1e-6).sum()),'negative_lt_minus1e6':int((differences< -1e-6).sum()),
        'near_ties':int((abs(differences)<=1e-6).sum()),
        'training_groups_replayed':20,'groups_with_correct_process_rank_change':sum(x['correct_process_rank_changed'] for x in replay),
        'advantage_sign_changes':sum(x['advantage_sign_changes'] for x in replay),
        'maximum_abs_advantage_change':max(x['max_abs_advantage_change'] for x in replay),
        'scored_versions':len(scored),'seconds':time.time()-started,
        'PRM_input_tokens':sum(x['prm_input_tokens'] for x in costs),'PRM_forward_seconds':sum(x['prm_forward_seconds'] for x in costs),
        'peak_torch_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'limits':['Conditional200 correct-placeholder MATH500 outputs and targeted20 training groups, not overall prevalence.',
        'Paired score sensitivity isolates this text edit, not causal evidence that PRM training induced the policy format.',
        'Deleting a placeholder can change segmentation; unchanged-step matches are reported separately.',
        'No diagnostic result changes continuation reward/loss or selects checkpoints.']}
    save(OUT/'result.json',result);print(json.dumps(result))

if __name__=='__main__':main()
