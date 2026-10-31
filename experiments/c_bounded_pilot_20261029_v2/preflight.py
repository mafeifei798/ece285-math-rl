"""CPU admission, independent formula comparison, and validation separation."""
import math,random,sys
from pilot_common import *
import torch
from advantages import compute_advantages

def main():
    guard();release=verify();AUDIT.mkdir(parents=True,exist_ok=True)
    from runtime import protect,trainer_class,components_for_arm
    trainer_class();protected=protect();atomic(AUDIT/'protected_originals.json',protected)
    # Regression: correct outcome + no PRM request is valid for arm A.
    probe_y=[1,0,1,0,0,1,0,0]
    a=components_for_arm('A',probe_y,[0.]*8,[False]*8)
    torch.testing.assert_close(a['advantages'],a['outcome_component'],atol=0,rtol=0)
    assert a['process_component'].count_nonzero()==0
    try:components_for_arm('C_bounded',probe_y,[0.]*8,[False]*8)
    except ValueError:pass
    else:raise AssertionError('C invalid-output guard was weakened')
    assert read(ROOT/'runs/C_bounded_no_update_20261029_v2/result.json')['passed']
    manifest=read(ROOT/'data/prepared/manifest.json')
    for n,item in manifest['files'].items():assert sha(ROOT/'data/prepared'/n)==item['sha256']
    train=rows(ROOT/'data/prepared/train.jsonl');val=rows(ROOT/'data/prepared/validation.jsonl')
    assert len(val)==500
    vs={r['question_sha256'] for r in val};ts={r['question_sha256'] for r in train}
    assert not vs&ts
    cd=ROOT/'runs/CD_PRM7B_pair_seed17_20261028'
    anchors=rows(cd/'anchors.jsonl');buffer=rows(cd/'buffer_questions.jsonl')
    assert not vs&{r['question_sha256'] for r in anchors+buffer}
    assert sha(cd/'C/adapter/adapter_model.safetensors')==read(HERE/'protocol.json')['C_adapter_sha256']
    cached=[r for r in rows(cd/'group_advantages.jsonl') if r['arm']=='C']
    checked=0;maxerr=0.;kinds=set()
    for g in cached:
        y=g.get('outcomes');m=g.get('min_scores')
        if y is None:
            # Historical files store gold outcome arrays under this fixed key.
            y=g.get('outcome_rewards')
        if y is None:raise ValueError(('Unknown group outcomes schema',list(g)))
        assert len(y)==len(m)==8
        result=compute_advantages(y,m);got=result['advantages'].tolist();mu=sum(y)/8
        sd=math.sqrt(sum((v-mu)**2 for v in y)/7);nc=sum(y)
        mc=sum(v*p for v,p in zip(y,m))/nc if nc else 0.
        reference=[(v-mu)/(sd+1e-4)+.2*v*(p-mc) for v,p in zip(y,m)]
        maxerr=max(maxerr,max(abs(a-b) for a,b in zip(got,reference)))
        assert maxerr<3e-7 and all(a<=0 for a,v in zip(got,y) if not v)
        checked+=8;kinds.add(g['kind'])
    assert kinds=={'all_wrong','all_correct','mixed'}
    generator=torch.Generator().manual_seed(17);order=torch.randperm(len(train),generator=generator).tolist()
    groups=[[train[i]['problem_id'] for i in order[start:start+4]] for start in range(0,len(order)-3,4)]
    old=rows(ROOT/'runs/A_corrected_seed17_6M_20261022/rollouts.jsonl')
    assert [r['problem_id'] for r in old[:32:8]]==groups[0]
    receipt=dict(passed=True,source_release_sha256=release,CPU_replay_rows=checked,max_reference_error=maxerr,group_types=sorted(kinds),
        validation_count=500,train_count=len(train),validation_train_overlap=0,validation_C_adaptation_overlap=0,
        validation_sha256=sha(ROOT/'data/prepared/validation.jsonl'),train_sha256=sha(ROOT/'data/prepared/train.jsonl'),
        first_epoch_question_groups=groups,initial_adapter_digest=read(ROOT/'runs/A_corrected_seed17_6M_20261022/run_config.json')['actor_adapter_initial_sha256'],
        A_no_PRM_is_not_invalid_regression=True,C_invalid_output_guard_retained=True,
        no_test_based_selection=True,protocol=read(HERE/'protocol.json'),originals=protected)
    atomic(AUDIT/'preflight.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['first_epoch_question_groups','originals','protocol']},indent=2),flush=True)

if __name__=='__main__':main()
