"""CPU acceptance and paired validation comparison; not test-set selection."""
from common import *
from collections import Counter
import numpy as np
from scipy.stats import binomtest

def paired(left,right):
    assert [x['problem_id'] for x in left]==[x['problem_id'] for x in right]
    x=np.array([r['reward'] for r in left]);y=np.array([r['reward'] for r in right]);d=x-y
    rng=np.random.default_rng(20261107)
    bootstrap=np.concatenate([d[rng.integers(0,len(d),size=(1000,len(d)))].mean(1) for _ in range(20)])
    wins=int(((x==1)&(y==0)).sum());losses=int(((x==0)&(y==1)).sum())
    return {'difference_percentage_points':float(d.mean()*100),'paired_bootstrap_95CI_pp':(np.quantile(bootstrap,[.025,.975])*100).tolist(),
        'wins':wins,'losses':losses,'both_correct':int(((x==1)&(y==1)).sum()),'both_wrong':int(((x==0)&(y==0)).sum()),
        'mcnemar_exact_two_sided_p':float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,'n':len(d)}


def main():
    verify();source_unchanged();evaluation=read(RUN/'evaluation/summary.json');assert evaluation['passed']
    data={a:rows(RUN/'evaluation'/a/'predictions.jsonl') for a in ('source',)+ARMS}
    costs={};settings=None
    for arm in ARMS:
        p=RUN/arm;result=read(p/'result.json');batches=rows(p/'generation_checks.jsonl');updates=rows(p/'update_audits.jsonl')
        assert result['passed'] and len(batches)==len(updates)==48 and result['loss_rows']==1536
        assert [b['question_ids'] for b in batches]==read(AUDIT/'preflight.json')['question_groups']
        assert [u['global_step'] for u in updates]==list(range(START+1,TARGET+1))
        cfg={k:v for k,v in read(p/'build.json')['settings'].items() if k not in ('output_dir','logging_dir','run_name')}
        assert cfg['loss_type']=='grpo'
        if settings is None:settings=cfg
        else:assert cfg==settings,'Actor settings changed beyond approved tree credit/PRM presence'
        reductions=rows(p/'loss_reductions.jsonl');assert len(reductions)==1536
        assert all(r['post_scalar_normalizations']==0 and r['gradient_accumulation_steps']==32 for r in reductions)
        assert all(b['wrong_positive_prm_tokens']==0 and b['post_scalar_normalizations']==0 for b in batches)
        assert all(b['signal_checks']['wrong_positive_exclusive_suffix_tokens']==0 for b in batches)
        if arm=='chain_cb':assert all(b['wrong_positive_tokens']==0 for b in batches)
        resource=rows(p/'resources.jsonl');statuses=Counter();groups=Counter()
        for b in batches:statuses.update(b['answer_statuses']);groups.update(b['group_types'])
        costs[arm]={'new_updates':48,'leaf_responses':1536,
           'unique_generation_tokens':sum(b['unique_generation_tokens'] for b in batches),
           'expanded_leaf_training_tokens':sum(b['tokens'] for b in batches),
           'PRM_input_tokens':sum(b['PRM_cost'].get('prm_input_tokens',0) for b in batches),
           'PRM_seconds':sum(b['PRM_cost'].get('prm_forward_seconds',0) for b in batches),
           'seconds':result['seconds'],'peak_sampled_GPU_GiB':max(r.get('own_total_MiB',0) for r in resource)/1024,
           'empty':sum(b['empty'] for b in batches),'short_wrong':sum(b['short_wrong'] for b in batches),
           'truncated':sum(b['truncated'] for b in batches),'statuses':dict(statuses),'group_types':dict(groups),
           'wrong_positive_total_tokens':sum(b['wrong_positive_tokens'] for b in batches),
           'wrong_positive_prm_tokens':0,
           'fork_fallbacks':sum(b['signal_checks']['fallback_roots'] for b in batches)}
        if arm=='tree_vr':assert costs[arm]['PRM_input_tokens']==0 and not read(p/'build.json')['prm_loaded']
    pairs={f'{a}_minus_{b}':paired(data[a],data[b]) for a,b in
          [('tree_cb','chain_cb'),('tree_vr','chain_cb'),('tree_cb','tree_vr')]+[(a,'source') for a in ARMS]}
    for arm,rr in data.items():
        assert sha(RUN/'evaluation'/arm/'predictions.jsonl')==evaluation['summary'][arm]['predictions_sha256']
    disagreements=[]
    for a,b in [('tree_cb','chain_cb'),('tree_cb','tree_vr'),('tree_vr','chain_cb')]:
        for l,r in zip(data[a],data[b]):
            if l['reward']!=r['reward']:disagreements.append({'pair':f'{a}_minus_{b}','problem_id':l['problem_id'],
                 'left_reward':l['reward'],'right_reward':r['reward'],'gold':l['gold'],
                 'left_completion':l['completion'],'right_completion':r['completion'],
                 'left_status':l['status'],'right_status':r['status']})
    atomic(RUN/'validation_disagreements.json',disagreements)
    protected=util.unchanged()
    for name,h in read(AUDIT/'protected_before.json').items():assert sha(ROOT/name)==h,('Historical input changed',name)
    atomic(RUN/'comparison.json',{'passed':True,'validation_summary':evaluation['summary'],'paired_comparisons':pairs,
         'training_costs':costs,'same_source_checkpoint_and_optimizer':True,'same_question_order':True,
         'same_update_count':True,'same_actor_settings':True,'historical_files_verified':protected,
         'additional_historical_files_verified':len(read(AUDIT/'protected_before.json')),
         'no_update_gate':read(AUDIT/'no_update/result.json'),'seed':17,'test_sets_used':False,
         'limits':['48-step continuation from C-bounded, not from-initial A/TreeRL reproduction.',
                   'Leaf count and updates matched; unique generation tokens and wall time are measured, not forced equal.',
                   'Paper-formula TreeRL-style credit with all8 leaves; no upstream RLOO preprocessing/selected-leaf resampling.',
                   'Wrong shared prefixes may have positive outcome credit; wrong PRM and exclusive-suffix positives are forbidden.',
                   'Question bootstrap excludes training-seed and numerical evaluation-repeatability variance.',
                   'Whole-trajectory PRM minimum projected onto an exclusive suffix remains a heuristic.']})
    print(json.dumps(read(RUN/'comparison.json'),ensure_ascii=False))


if __name__=='__main__':main()

