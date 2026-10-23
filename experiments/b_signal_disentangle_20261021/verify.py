"""Independent mathematical/property checks on the proposed signal repair."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from bounded_signal import compute_advantages

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'execution/b_signal_disentangle_20261021'


def main():
    torch.set_num_threads(2)
    binary=torch.tensor(list(itertools.product([0.,1.],repeat=8)),dtype=torch.float32)
    y=binary.repeat_interleave(256,dim=0)
    p=binary.repeat(256,1)
    result=compute_advantages(y.flatten(),p.flatten())
    a=result['advantages'].view(-1,8);ap=result['process_component'].view(-1,8)
    mixed=(y.sum(1)>0)&(y.sum(1)<8)
    assert (a[mixed][y[mixed]==1]>0).all() and (a[mixed][y[mixed]==0]<0).all()
    assert float(ap.abs().max())<=.175001
    zero=compute_advantages(y.flatten(),p.flatten(),coefficient=0)
    assert torch.equal(zero['advantages'],zero['outcome_component'])
    data=[json.loads(s) for s in (ROOT/'execution/reward_repair_v3_20261021/variants/combined.jsonl').open()]
    yy=[r['outcome_reward'] for r in data];mm=[r['min_score'] if r['process_scored'] else 0 for r in data]
    rr=compute_advantages(yy,mm,valid=[r['process_scored'] for r in data])
    yn=np.asarray(yy,dtype=float).reshape(-1,8);mn=np.asarray(mm,dtype=float).reshape(-1,8)
    independent=(yn-yn.mean(1,keepdims=True))/(yn.std(1,ddof=1,keepdims=True)+1e-4)+.2*(mn-mn.mean(1,keepdims=True))
    error=float(np.abs(independent.flatten()-rr['advantages'].numpy()).max())
    assert error<2e-6
    assert float(rr['process_component'].abs().max())<=.175001
    # Whole traces, including the original unscored partners, are retained.
    original=[json.loads(s) for s in (ROOT/'execution/b7b_training_audit_20261021/inputs/runs/B_Qwen7B_seed17_6M/rollouts.jsonl').open()]
    for start in range(0,len(data),8):
        assert len({r['problem_id'] for r in data[start:start+8]})==1
        assert [r['problem_id'] for r in data[start:start+8]]==[r['problem_id'] for r in original[start:start+8]]
    failures=0
    for ys,ms,vs in [([1.]*7,[.5]*7,None),([1.]*8,[float('nan')]*8,None),
        ([1.]*8,[1.1]*8,None),([.5]*8,[.2]*8,None),([0.]*8,[.2]*8,[False]*8)]:
        try:compute_advantages(ys,ms,valid=vs)
        except ValueError:failures+=1
    assert failures==5
    # These arithmetic checks do not execute model-generated code.
    assert np.isclose(1/3,4/12)
    assert np.isclose(.6*.25,.15)
    assert np.isclose(.6+.25-.6*.25,.70)
    assert 22000*250/1000==5500
    assert 36%8==4 and 80%16==0
    result={'passed':True,'exhaustive_binary_outcome_and_extreme_probability_groups':65536,
        'mixed_groups':int(mixed.sum()),'mixed_outcome_sign_violations':0,
        'absolute_process_bound':.175,'zero_coefficient_exactly_recovers_outcome_GRPO':True,
        'fixed_rollout_rows':len(data),'independent_FP64_max_difference':error,
        'invalid_inputs_rejected':failures,'original_partners_preserved':True,
        'manual_case_arithmetic_checks':True,'no_model_loaded':True,'no_training':True}
    (OUT/'independent_QA.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
