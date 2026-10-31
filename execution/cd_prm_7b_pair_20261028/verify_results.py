"""Independent CPU verification of saved scoring, ranking and advantage evidence."""
from pathlib import Path
import json
import hashlib
import struct
import statistics as st
import math
import collections
import random
import re
import unicodedata

BASE=Path(__file__).resolve().parent
P=BASE/'artifacts'
def read(n):return json.loads((P/n).read_text())
def rows(n):return [json.loads(s) for s in (P/n).open() if s.strip()]
def sha(n):return hashlib.sha256((P/n).read_bytes()).hexdigest()
def avg(v):return st.fmean(v) if v else None
def auroc(pos,neg):return sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg)) if pos and neg else None
def adv(rewards):
    q=[struct.unpack('f',struct.pack('f',float(v)))[0] for v in rewards]
    mu=st.fmean(q);std=st.stdev(q)
    return [(v-mu)/(std+1e-4) for v in q],std


def main():
    post=read('wiring_recovery/postprocess.json');assert post['status']=='complete'
    assert post['policy_optimizer_updates']==post['PRM_optimizer_updates']==0
    assert post['preservation']['unchanged']
    assert read('wiring_recovery/fp32_reference_v2/gradient_replay.json')['strict_identity_check_passed']
    assert read('paired_wiring.json')['passed'] is False
    summary=read('summary.json');assert sha('summary.json')==post['summary_sha256']
    sources={a:rows(f'scores_{a}.jsonl') for a in ['B','C','D']}
    ix={a:{r['example_id']:r for r in data} for a,data in sources.items()}
    assert all(len(ix[a])==len(sources[a])==728 for a in ix)
    assert set(ix['B'])==set(ix['C'])==set(ix['D'])
    for a in ix:
        receipt=read(f'scoring_{a}.json');assert sha(f'scores_{a}.jsonl')==receipt['sha256']
        assert receipt['actor_updates']==0 and receipt['native_scoring'] and receipt['normalize_crlf']
        if a!='B':assert receipt['adapter_sha256']==read(f'{a}/completion.json')['adapter_sha256']
        for r in sources[a]:
            assert (r['step_scores'] and min(r['step_scores'])==r['min_score']) if r['eligible'] else r['min_score']==0 and not r['step_scores']
    for i in ix['B']:
        assert len({(ix[a][i]['dataset'],ix[a][i]['eligible'],ix[a][i]['outcome']) for a in ix})==1
    for k in ['initial_adapter_hash','prm_revision','batch_manifest_sha256','steps','config_sha256','trainable_parameters']:
        assert read('C/initialization.json')[k]==read('D/initialization.json')[k]
    assert read('C/completion.json')['steps_completed']==read('D/completion.json')['steps_completed']==512
    grouped=collections.defaultdict(list)
    for r in rows('group_advantages.jsonl'):grouped[(r['arm'],r['dataset'])].append(r)
    metrics={};cases=[]
    for a in ix:
        arms=collections.defaultdict(list)
        for g in grouped[(a,'probe')]:
            items=[ix[a][i] for i in g['example_ids']];assert len(items)==8
            assert [r['outcome'] for r in items]==g['outcomes']
            m=[r['min_score'] if r['eligible'] else 0. for r in items];y=g['outcomes']
            rewards=[v+.2*s for v,s in zip(y,m)]
            actual,std=adv(rewards);a_only,_=adv(y)
            assert max(abs(x-y) for x,y in zip(actual,g['advantages']))<1e-10
            assert max(abs(x-y) for x,y in zip(rewards,g['rewards']))<1e-12
            ranks=[1+sum(s>v for s in m)+.5*(sum(s==v for s in m)-1) for v in m]
            assert ranks==g['ranks']
            kind='all_wrong' if not any(y) else 'all_correct' if all(y) else 'mixed';assert kind==g['kind']
            d=dict(mean_min=avg(m),min_score_std=st.stdev(m),reward_std=std,
                   advantage_mean_abs=avg([abs(v) for v in actual]),
                   deviation_from_outcome_only=avg([abs(x-y) for x,y in zip(actual,a_only)]),
                   wrong_count=sum(not v for v in y),wrong_positive=sum(not v and x>0 for v,x in zip(y,actual)),
                   low_signal=std<1e-4,auc=auroc([s for v,s in zip(y,m) if v],[s for v,s in zip(y,m) if not v]))
            arms[kind].append(d)
        metrics[a]={k:dict(groups=len(v),mean_min_score=avg([g['mean_min'] for g in v]),
            mean_min_score_std=avg([g['min_score_std'] for g in v]),mean_abs_advantage=avg([g['advantage_mean_abs'] for g in v]),
            mean_abs_advantage_deviation_from_A=avg([g['deviation_from_outcome_only'] for g in v]),
            wrong_positive=sum(g['wrong_positive'] for g in v),wrong_count=sum(g['wrong_count'] for g in v),
            groups_with_reward_std_below_epsilon=sum(g['low_signal'] for g in v),
            mean_group_correctness_AUROC=avg([g['auc'] for g in v if g['auc'] is not None])) for k,v in arms.items()}
    pb_original={f'PB-{r["id"]}':r for r in rows('processbench.jsonl')}
    pb={}
    for a in ix:
        rs=[r for r in sources[a] if r['dataset']=='processbench' and r['eligible']]
        for r in rs:assert len(r['step_scores'])==len(pb_original[r['example_id']]['steps'])
        valid=[r for r in rs if r['label']==-1];invalid=[r for r in rs if r['label']!=-1]
        accept=sum(min(r['step_scores'])>=.5 for r in valid)/len(valid)
        first=sum(next((i for i,v in enumerate(r['step_scores']) if v<.5),-1)==r['label'] for r in invalid)/len(invalid)
        auc=auroc([r['min_score'] for r in valid if r['outcome']],[r['min_score'] for r in invalid if r['outcome']])
        pb[a]=dict(n=len(rs),valid_acceptance=accept,first_error_accuracy=first,
                   harmonic_mean=2*accept*first/(accept+first) if accept+first else 0.,answer_correct_AUROC=auc)
        for k,v in [('valid_acceptance',accept),('first_error_accuracy',first),('answer_correct_valid_vs_invalid_AUROC',auc)]:
            assert math.isclose(summary['processbench'][a][k],v,abs_tol=1e-12)
    illustrations={r['example_id']:r for r in rows('illustrations.jsonl')}
    all_groups={(r['arm'],str(r['group_id'])):r for r in rows('group_advantages.jsonl') if r['dataset']=='illustrations'}
    for i,r in illustrations.items():
        if not r['known_error_case']:continue
        case=dict(example_id=i,source_row=r['row_index'],outcome=ix['B'][i]['outcome'],scores={})
        for a in ix:
            g=all_groups[(a,str(r['group_id']))];j=g['example_ids'].index(i)
            case['scores'][a]=dict(min_score=ix[a][i]['min_score'],rank=g['ranks'][j],advantage=g['advantages'][j])
        cases.append(case)
    clusters=collections.defaultdict(list)
    for i, r in pb_original.items():
        if not ix['B'][i]['eligible']:continue
        key=re.sub(r'\s+','',unicodedata.normalize('NFKC',r['problem']))
        clusters[key].append(i)
    rng=random.Random(285);keys=list(clusters);cb=[];dc=[]
    for _ in range(2000):
        draw=[i for k in rng.choices(keys,k=len(keys)) for i in clusters[k]]
        aucs={a:auroc([ix[a][i]['min_score'] for i in draw if ix[a][i]['label']==-1 and ix[a][i]['outcome']],
                     [ix[a][i]['min_score'] for i in draw if ix[a][i]['label']!=-1 and ix[a][i]['outcome']]) for a in ix}
        if all(v is not None for v in aucs.values()):cb.append(aucs['C']-aucs['B']);dc.append(aucs['D']-aucs['C'])
    def interval(values):
        values=sorted(values);return [values[int(q*(len(values)-1))] for q in [.025,.975]]
    dc_ci=interval(dc);cb_ci=interval(cb)
    assert all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(dc_ci,summary['D_minus_C_answer_correct_AUROC_CI95']))
    result=dict(passed=True,score_rows_per_arm=728,full_groups_verified=64*3,reference_FP32_passed=True,
                historical_BF16_check_preserved_failed=True,originals_unchanged=post['preservation'],
                groups=metrics,processbench=pb,known_cases=cases,
                paired_question_cluster_bootstrap=dict(clusters=len(keys),replicates=2000,
                    D_minus_C_AUROC_CI95=dc_ci,C_minus_B_AUROC_CI95_exploratory=cb_ci),
                interpretation_rules=['Verifier wrong is not a proof that all derivation steps are invalid.',
                    'All-wrong groups generally retain positive relative advantages; fewer positives alone is not proof of better ranking.',
                    'Very small reward spread can suppress advantages through epsilon/FP32 quantization without improving PRM ranking.',
                    'No policy RL or downstream mathematics accuracy was measured.'])
    (BASE/'independent_result_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
