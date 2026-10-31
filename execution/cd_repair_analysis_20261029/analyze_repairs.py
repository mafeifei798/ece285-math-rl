"""Frozen-score CPU diagnosis and counterfactual replay; no model/optimizer."""
from pathlib import Path
import json,math,statistics as st,collections,hashlib,struct,os

HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('CD_FROZEN_ARTIFACTS',str(HERE.parent/'cd_prm_7b_pair_20261028/artifacts')))
LAM=.2;EPS=1e-4
def read(n):return json.loads((SOURCE/n).read_text())
def rows(n):return [json.loads(s) for s in (SOURCE/n).open() if s.strip()]
def mean(v):return st.fmean(v) if v else None
def std(v):return st.stdev(v) if len(v)>1 else 0.
def f32(v):return struct.unpack('f',struct.pack('f',float(v)))[0]
def advantages(rs):
    rs=list(map(f32,rs));mu=mean(rs);sd=std(rs)
    return [(r-mu)/(sd+EPS) for r in rs],mu,sd
def logit(p):
    p=max(1e-7,min(1-1e-7,p));return math.log(p)-math.log1p(-p)
def sigmoid(z):return 1/(1+math.exp(-z))
def shifted(ps,d):return [sigmoid(logit(p)+d) for p in ps]
def sws(ps):
    es=[1-p for p in ps];z=max(es);w=[math.exp((e-z)/.2) for e in es]
    return sum(p*x for p,x in zip(ps,w))/sum(w)
def auc(pos,neg):return sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg)) if pos and neg else None
def rank(vs,v):return 1+sum(x>v for x in vs)+.5*(sum(x==v for x in vs)-1)
def feature(r):
    text=r['completion'].strip();bare=False
    if text.startswith('\\boxed{'):
        depth=0;end=None
        for i,c in enumerate(text[text.index('{'):],text.index('{')):
            depth+=(c=='{')-(c=='}')
            if depth==0:end=i+1;break
        bare=end==len(text)
    return dict(bare_box=bare,characters=len(text),paragraph_steps=len([s for s in text.replace('\r\n','\n').split('\n\n') if s.strip()]))


def main():
    used=['probe.jsonl','illustrations.jsonl','processbench.jsonl','group_advantages.jsonl',*[f'scores_{a}.jsonl' for a in ['B','C','D']]]
    hashes={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in used}
    source={r['example_id']:r for n in ['probe.jsonl','illustrations.jsonl'] for r in rows(n)}
    pb_src={r['example_id']:r for r in rows('processbench.jsonl')}
    scores={a:{r['example_id']:r for r in rows(f'scores_{a}.jsonl')} for a in ['B','C','D']}
    groups=[r for r in rows('group_advantages.jsonl') if r['arm']=='B']
    result=dict(optimizer_updates=0,new_model_calls=0,new_training=False,analysis_only=True,
                selection='All frozen groups included. No threshold selected using these diagnostics.',cases=[],length_pairs={},PB={},reward_replay={},logit_shift_stress={},weak_loss_shift_stress={})
    aggregates={'min':min,'sws_training_pool':sws,'mean_diagnostic_only':mean}
    for a in scores:
        result['PB'][a]={}
        for name,agg in aggregates.items():
            rr=[r for r in scores[a].values() if r['dataset']=='processbench' and r['eligible']]
            pos=[agg(r['step_scores']) for r in rr if r['label']==-1 and r['outcome']]
            neg=[agg(r['step_scores']) for r in rr if r['label']!=-1 and r['outcome']]
            result['PB'][a][name]=dict(answer_correct_validity_AUROC=auc(pos,neg),valid_acceptance=sum(agg(r['step_scores'])>=.5 for r in rr if r['label']==-1)/64,
                invalid_answer_correct_acceptance=sum(p>=.5 for p in neg)/len(neg),
                mean_valid_sequence_score=mean(pos),mean_invalid_answer_correct_sequence_score=mean(neg))
        valid=[r for r in scores[a].values() if r['dataset']=='processbench' and r['label']==-1 and r['eligible']]
        bad=[r for r in scores[a].values() if r['dataset']=='processbench' and r['label']!=-1 and r['eligible']]
        result['PB'][a]['step_distribution']=dict(valid_mean_step_score=mean([p for r in valid for p in r['step_scores']]),
            valid_mean_sequence_min=mean([r['min_score'] for r in valid]),valid_fraction_steps_below_half=mean([p<.5 for r in valid for p in r['step_scores']]),
            valid_step_count=sum(len(r['step_scores']) for r in valid),valid_steps_below_half=sum(p<.5 for r in valid for p in r['step_scores']),
            first_error_mean_score=mean([r['step_scores'][r['label']] for r in bad]))
        if a!='B':
            result['PB'][a]['logit_drift_vs_B']=dict(valid_steps_mean=mean([logit(p)-logit(q) for r in valid for p,q in zip(r['step_scores'],scores['B'][r['example_id']]['step_scores'])]),
                first_error_mean=mean([logit(r['step_scores'][r['label']])-logit(scores['B'][r['example_id']]['step_scores'][r['label']]) for r in bad]))
        # Only derivatives with respect to cached logits, not parameter gradients.
        # This assesses the direction a prospective PRM retention term supplies.
        correct_invalid=[r for r in bad if r['outcome']]
        def sws_success_gradient_at_error(r):
            ps=r['step_scores'];es=[1-p for p in ps];top=max(es)
            weights=[math.exp((e-top)/.2) for e in es];total=sum(weights);weights=[w/total for w in weights]
            t=r['label'];q=sum(w*p for w,p in zip(weights,ps))
            return -weights[t]*ps[t]*(1-ps[t])/max(q,1e-7)
        result['PB'][a]['retention_term_logit_directions']=dict(
            known_valid_steps_teacher_KL_gradient=mean([p-q for r in valid for p,q in zip(r['step_scores'],scores['B'][r['example_id']]['step_scores'])]),
            answer_correct_first_error_actual_SWS_success_gradient=mean([sws_success_gradient_at_error(r) for r in correct_invalid]),
            answer_correct_first_error_teacher_KL_gradient=mean([r['step_scores'][r['label']]-scores['B'][r['example_id']]['step_scores'][r['label']] for r in correct_invalid]),
            answer_correct_first_error_teacher_KL_trajectory_averaged_gradient=mean([(r['step_scores'][r['label']]-scores['B'][r['example_id']]['step_scores'][r['label']])/len(r['step_scores']) for r in correct_invalid]),
            reference='SWS success derivative=-w_t*p_t*(1-p_t)/sum(w*p), detached w. KL derivative=(p_current-p_B)/T. Raw KL derivative also shown before step averaging.',
            parameter_gradient_or_training_result=False)
        pairs=collections.defaultdict(list);styles=collections.defaultdict(list)
        for g in groups:
            if g['dataset']!='probe':continue
            rs=[scores[a][i] for i in g['example_ids'] if scores[a][i]['eligible']]
            for r in rs:
                f=feature(source[r['example_id']]);styles[(g['kind'],'bare' if f['bare_box'] else 'nonbare')].append(r['min_score'])
            for i,x in enumerate(rs):
                for y in rs[i+1:]:
                    if x['outcome']!=y['outcome']:continue
                    if len(x['step_scores'])==len(y['step_scores']):continue
                    short,long=sorted([x,y],key=lambda r:len(r['step_scores']))
                    pairs[g['kind']].append((short['min_score']>long['min_score'])+.5*(short['min_score']==long['min_score']))
        result['length_pairs'][a]={k:dict(pairs=len(v),shorter_wins_fraction=mean(v)) for k,v in pairs.items()}
        result['logit_shift_stress'][a]={}
        for delta in [0.,math.log(10),math.log(100)]:
            rs=[r for r in scores[a].values() if r['dataset']=='processbench' and r['eligible']]
            pos=[min(shifted(r['step_scores'],delta)) for r in rs if r['label']==-1 and r['outcome']]
            neg=[min(shifted(r['step_scores'],delta)) for r in rs if r['label']!=-1 and r['outcome']]
            ac=auc(pos,neg)
            assert math.isclose(ac,result['PB'][a]['min']['answer_correct_validity_AUROC'],abs_tol=1e-12)
            result['logit_shift_stress'][a][str(delta)]=dict(AUROC_unchanged=ac,valid_acceptance=sum(p>=.5 for p in pos)/len(pos),invalid_answer_correct_acceptance=sum(p>=.5 for p in neg)/len(neg))
        result['weak_loss_shift_stress'][a]={}
        for delta in [-math.log(10),0.,math.log(10)]:
            losses={0:[],1:[]}
            for r in scores[a].values():
                if r['dataset']!='probe' or not r['eligible']:continue
                q=max(1e-7,min(1-1e-7,sws(shifted(r['step_scores'],delta))))
                losses[r['outcome']].append(-math.log(q) if r['outcome'] else -math.log1p(-q))
            result['weak_loss_shift_stress'][a][str(delta)]=dict(failure_only_mean=mean(losses[0]),success_mean=mean(losses[1]),
                success_plus_failure_mean=mean(losses[0])+mean(losses[1]),anchors_not_included=True)
    for g in groups:
        if g['dataset']!='illustrations':continue
        for i in g['example_ids']:
            if not source[i]['known_error_case']:continue
            case=dict(example_id=i,question=source[i]['question'],completion=source[i]['completion'],style=feature(source[i]),arms={})
            for a in scores:
                rs=[scores[a][j] for j in g['example_ids']];mm=[r['min_score'] for r in rs];rr=[r['outcome']+LAM*m for r,m in zip(rs,mm)]
                aa,baseline,sd=advantages(rr);idx=g['example_ids'].index(i)
                case['arms'][a]=dict(step_scores=scores[a][i]['step_scores'],process_min=mm[idx],reward=rr[idx],group_reward_mean=baseline,group_reward_sample_std=sd,advantage=aa[idx],eligible_group_members=sum(r['eligible'] for r in rs),
                    all_wrong=not any(r['outcome'] for r in rs),aggregator_ranks={name:rank([agg(r['step_scores']) if r['eligible'] else 0 for r in rs],agg(scores[a][i]['step_scores'])) for name,agg in aggregates.items()})
            result['cases'].append(case)
    replay_rows=[]
    for a in scores:
        for agg_name,agg in [('min',min),('sws_training_pool',sws)]:
            variants=collections.defaultdict(list)
            for g in groups:
                rs=[scores[a][i] for i in g['example_ids']];y=[r['outcome'] for r in rs];m=[agg(r['step_scores']) if r['eligible'] else 0. for r in rs]
                original,_,_=advantages([v+LAM*p for v,p in zip(y,m)])
                gated,_,_=advantages([v*(1+LAM*p) for v,p in zip(y,m)])
                ao,_,_=advantages(y);correct=[p for v,p in zip(y,m) if v]
                centered=[LAM*(p-mean(correct)) if v else 0. for v,p in zip(y,m)] if correct else [0.]*len(y)
                bounded=[v+d for v,d in zip(ao,centered)]
                assert max(map(abs,centered))<=LAM+1e-12
                assert not any(v>0 and not good for v,good in zip(gated,y))
                assert not any(v>0 and not good for v,good in zip(bounded,y))
                assert abs(sum(centered))<1e-12
                for name,vs in [('original',original),('outcome_gate_same_GRPO_normalization',gated),('bounded_correct_only_process_advantage',bounded)]:
                    d=dict(arm=a,aggregator=agg_name,dataset=g['dataset'],group=g['group_id'],kind=g['kind'],variant=name,example_ids=g['example_ids'],outcomes=y,process=m,advantage=vs,
                        wrong_positive=sum(not good and v>0 for good,v in zip(y,vs)),wrong_count=sum(not v for v in y),mean_abs_advantage=mean([abs(v) for v in vs]),
                        max_abs_change_vs_outcome_only=max(abs(x-z) for x,z in zip(vs,ao)))
                    replay_rows.append(d);variants[(g['dataset'],g['kind'],name)].append(d)
            result['reward_replay'][a+'/'+agg_name]={ds+'/'+kind+'/'+name:dict(groups=len(v),wrong_positive=sum(x['wrong_positive'] for x in v),wrong_count=sum(x['wrong_count'] for x in v),mean_abs_advantage=mean([x['mean_abs_advantage'] for x in v]),max_abs_change_vs_outcome_only=max(x['max_abs_change_vs_outcome_only'] for x in v)) for (ds,kind,name),v in variants.items()}
    assert hashes=={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in used}
    result['frozen_input_hashes']=hashes
    result['all_original_artifacts_unchanged']=True
    (HERE/'diagnosis_and_counterfactuals.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'replayed_groups.jsonl').open('w') as f:
        for r in replay_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ['PB','length_pairs','weak_loss_shift_stress']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
