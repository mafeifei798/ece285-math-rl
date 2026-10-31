"""CPU-only paired validation comparisons; no automatic budget expansion."""
import random,math,sys
from pilot_common import *

def percentile(v,p):return sorted(v)[int((len(v)-1)*p)]
def main():
    verify();pred={};summary={}
    for a in ['initial',*ARMS]:
        p=PAIR/'validation'/a
        summary[a]=read(p/'summary.json');assert summary[a]['passed']
        assert sha(p/'predictions.jsonl')==summary[a]['predictions_sha256']
        pred[a]={r['problem_id']:r for r in rows(p/'predictions.jsonl')};assert len(pred[a])==500
    comparisons={}
    for lhs,rhs in [('C_bounded','C_mix'),('C_bounded','A'),('C_mix','A'),('A','initial')]:
        assert pred[lhs].keys()==pred[rhs].keys()
        d=[pred[lhs][i]['reward']-pred[rhs][i]['reward'] for i in sorted(pred[lhs])]
        rng=random.Random(285);boot=[sum(rng.choices(d,k=500))/500 for _ in range(5000)]
        comparisons[lhs+' minus '+rhs]=dict(delta=sum(d)/500,paired_bootstrap95=[percentile(boot,.025),percentile(boot,.975)],
            wins=d.count(1),losses=d.count(-1),unchanged=d.count(0),n=500)
    training={a:read(PAIR/a/'final_result.json') for a in ARMS}
    for a in ARMS:
        t=training[a];assert t['passed'] and 1000000<=t['accounting']['generated_completion_tokens']<1065536
        assert read(PAIR/a/'resume_acceptance.json')['passed']
        if a!='C_mix':assert t['accounting']['loss_wrong_positive']==0
    import sys
    sys.path.insert(0,str(ORIGIN));from audit_common import protect
    assert protect()==read(AUDIT/'protected_originals.json')
    result=dict(
        passed=True,validation=summary,paired=comparisons,training=training,original_ABCD_preserved=True,
        source_release_sha256=verify(),test_sets_used_for_selection=False,new_seed_or_6M_started=False,D_adaptation=False,
        interpretation='Exploratory seed17 pilot. The zero wrong-positive rate of bounded/A is guaranteed by construction; accuracy and nonzero process contributions must be considered separately. Small paired intervals are not seed robustness or equivalence evidence.')
    atomic(PAIR/'comparison.json',result)
    lines=['Three-arm C bounded pilot: COMPLETE','Validation only; no 6M or D adaptation started.','']
    for a,s in summary.items():lines.append(f"{a}: {s['correct']}/500 = {s['accuracy']:.1%}; empty {s['empty']}, short wrong {s['short_wrong']}, truncated {s['truncated']}")
    for k,v in comparisons.items():lines.append(f"{k}: {v['delta']:+.1%}; paired 95% {v['paired_bootstrap95']}; wins/losses {v['wins']}/{v['losses']}")
    (PAIR/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
