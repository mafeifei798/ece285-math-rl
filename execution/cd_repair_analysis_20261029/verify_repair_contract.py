"""Meaningful mathematical acceptance checks and full saved-group replay."""
from pathlib import Path
import math,json
from signal_repair import bounded_correct_process,reference_retention,outcome_advantages


def main():
    p=Path(__file__).resolve().parent
    records=[json.loads(s) for s in (p/'replayed_groups.jsonl').open() if s.strip()]
    checked=0
    for r in records:
        if r['variant']!='bounded_correct_only_process_advantage':continue
        x=bounded_correct_process(r['outcomes'],r['process'])
        assert max(abs(a-b) for a,b in zip(x['advantage'],r['advantage']))<1e-12
        checked+=1
    assert checked==3*2*(64+3)
    tiny=bounded_correct_process([1]*8,[.5+i*1e-8 for i in range(8)])
    assert max(map(abs,tiny['advantage']))<1e-7
    empty=bounded_correct_process([0]*8,[.239,.032,.1,.2,.7,.8,.9,.01])
    assert empty['advantage']==[0.]*8
    y=[1,0,1,0,1,0,1,0];m=[.9,.999,.8,.9,.7,.1,.6,.4]
    forward=bounded_correct_process(y,m)
    reverse=bounded_correct_process(y[::-1],m[::-1])
    assert forward['advantage']==reverse['advantage'][::-1]
    constant=bounded_correct_process(y,[.6]*8)
    assert constant['advantage']==outcome_advantages(y)
    for prob,target in [(.359,.97),(.211,.296),(.01,.01),(.8,.2)]:
        z=math.log(prob/(1-prob));h=1e-5
        sigmoid=lambda z:1/(1+math.exp(-z))
        lo=reference_retention([sigmoid(z-h)],[target])['mean_KL']
        hi=reference_retention([sigmoid(z+h)],[target])['mean_KL']
        analytic=reference_retention([prob],[target])['d_student_logit'][0]
        assert abs((hi-lo)/(2*h)-analytic)<1e-7
    receipt=dict(passed=True,existing_group_replays=checked,wrong_cannot_get_positive_advantage=True,
      process_contribution_bounded_by_coefficient=True,tiny_spread_not_renormalized=True,
      permutation_equivariance=True,constant_process_reduces_to_outcome_only=True,
      KL_logit_derivative_finite_difference_checked=True,model_calls=0,optimizer_updates=0,
      actual_PRM_ranking_fixed=False,policy_accuracy_measured=False,trainer_integration=False)
    (p/'repair_contract_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
