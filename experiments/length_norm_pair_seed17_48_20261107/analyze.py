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
    for a,rr in data.items():
        assert sha(RUN/'evaluation'/a/'predictions.jsonl')==evaluation['summary'][a]['predictions_sha256']
        assert sum(x['reward'] for x in rr)==evaluation['summary'][a]['correct']
    costs={};reference_order=None;settings=None
    for arm in ARMS:
        run=RUN/arm;result=read(run/'result.json');updates=rows(run/'update_audits.jsonl');batches=rows(run/'generation_checks.jsonl')
        assert result['passed'] and len(updates)==len(batches)==48
        assert [r['global_step'] for r in updates]==list(range(START+1,TARGET+1))
        order=[r['question_ids'] for r in batches]
        assert order==read(AUDIT/'preflight.json')['question_groups']
        if reference_order is None:reference_order=order
        else:assert order==reference_order
        raw_cfg=read(run/'build.json')['settings']
        assert raw_cfg['loss_type']==('grpo' if arm=='grpo' else 'dr_grpo')
        cfg={k:v for k,v in raw_cfg.items() if k not in ['output_dir','logging_dir','run_name','loss_type']}
        if settings is None:settings=cfg
        else:assert cfg==settings,'Settings differ beyond approved loss reduction'
        reductions=rows(run/'loss_reductions.jsonl');assert len(reductions)==48*32
        assert all(r['constant_length']==2048 and r['gradient_accumulation_steps']==32 and r['post_scalar_normalizations']==0 and r['loss_type']==raw_cfg['loss_type'] for r in reductions)
        for b in batches:assert b['wrong_positive_tokens']==b['post_scalar_normalizations']==0
        resource=rows(run/'resources.jsonl');peak=max(x.get('own_total_MiB',0) for x in resource)/1024
        statuses=Counter();groups=Counter()
        for b in batches:statuses.update(b['answer_statuses']);groups.update(b['group_types'])
        costs[arm]={'new_updates':48,'new_training_responses':1536,'new_generation_tokens':sum(b['tokens'] for b in batches),
            'PRM_input_tokens':sum(b['PRM_cost'].get('prm_input_tokens',0) for b in batches),
            'PRM_seconds':sum(b['PRM_cost'].get('prm_forward_seconds',0) for b in batches),'total_seconds':result['seconds'],
            'peak_sampled_GPU_GiB':peak,'protected_tokens_if_prefix':sum(b['protected_tokens'] for b in batches),
            'signal_statuses':dict(statuses),'group_types':dict(groups),'empty':sum(b['empty'] for b in batches),
            'short_wrong':sum(b['short_wrong'] for b in batches),'truncated':sum(b['truncated'] for b in batches),
            'truncation_rate':sum(b['truncated'] for b in batches)/1536,
            'missing_or_invalid_box':statuses['missing_or_invalid_box'],
            'missing_or_invalid_box_rate':statuses['missing_or_invalid_box']/1536,
            'loss_type':raw_cfg['loss_type']}
    pairs={f'{a}_minus_{b}':paired(data[a],data[b]) for a,b in [('fixed_length','grpo')]+[(a,'source') for a in ARMS]}
    main=pairs['fixed_length_minus_grpo']
    conclusion='inconclusive'
    if main['paired_bootstrap_95CI_pp'][0]>0:conclusion='fixed_length_higher_on_this_single_seed_validation'
    elif main['paired_bootstrap_95CI_pp'][1]<0:conclusion='fixed_length_lower_on_this_single_seed_validation'
    disagreements=[{'problem_id':l['problem_id'],'fixed_length_reward':l['reward'],'grpo_reward':r['reward'],
        'fixed_length_completion':l['completion'],'grpo_completion':r['completion'],'gold':l['gold'],
        'fixed_length_status':l['status'],'grpo_status':r['status']} for l,r in zip(data['fixed_length'],data['grpo']) if l['reward']!=r['reward']]
    atomic(RUN/'validation_disagreements.json',disagreements)
    protected=util.unchanged();extra=0
    for p,h in read(AUDIT/'protected_before.json').items():assert sha(ROOT/p)==h,('Historical evidence changed',p);extra+=1
    out={'passed':True,'validation_summary':evaluation['summary'],'training_costs':costs,'paired_comparisons':pairs,'main_conclusion':conclusion,
        'same_question_order':True,'same_update_count':True,'same_settings_except_loss_type':True,'same_bounded_advantage_formula':True,'source_checkpoint_unchanged':True,
        'old_protected_files_verified':protected,'additional_historical_files_verified':extra,'single_seed':17,'validation_previously_used':True,'test_sets_used':False,
        'no_update_gate':read(AUDIT/'no_update/result.json'),
        'loss_change':'Per-response effective length denominator -> constant2048 only. Group reward std, PRM, KL and LR unchanged.',
        'limits':'Question bootstrap does not measure training-seed variance. Small short-pilot differences can be inconclusive. No automatic budget expansion.'}
    atomic(RUN/'comparison.json',out);print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
