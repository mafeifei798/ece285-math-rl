"""Paired reward/rank/advantage replay and fixed-label error audit; no fitting."""
from collections import Counter
import csv
import html
import json
import math
from pathlib import Path
import random
import statistics
import sys
from core import (atomic_json, auc, cpu_advantages, midranks, read_jsonl,
                  verified_inputs, write_jsonl)


def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def sign(x):
    return int(x>1e-6)-int(x< -1e-6)


def relation(x,y):
    return sign(x-y)


def pearson(x,y):
    xm=statistics.mean(x);ym=statistics.mean(y)
    a=math.fsum((v-xm)**2 for v in x);b=math.fsum((v-ym)**2 for v in y)
    if a==0 or b==0:return None
    return math.fsum((v-xm)*(w-ym) for v,w in zip(x,y))/math.sqrt(a*b)


def paired_bootstrap_auc(records, repetitions=2000):
    if not any(r['valid'] for r in records) or not any(not r['valid'] for r in records):return None
    def stat(sample,key):
        return auc([r[key] for r in sample if r['valid']], [r[key] for r in sample if not r['valid']])
    old=stat(records,'h4');new=stat(records,'qwen')
    rng=random.Random(20261021);diff=[]
    for _ in range(repetitions):
        sample=[records[rng.randrange(len(records))] for i in range(len(records))]
        a,b=stat(sample,'h4'),stat(sample,'qwen')
        if a is not None and b is not None:diff.append(b-a)
    diff.sort()
    return {'h4':old,'qwen7b':new,'difference':new-old,
            'difference_95_percentile_interval':[diff[int(.025*(len(diff)-1))],diff[int(.975*(len(diff)-1))]],
            'clean_solutions':sum(r['valid'] for r in records),'erroneous_solutions':sum(not r['valid'] for r in records),
            'paired_resampling_unit':'previously reviewed question/solution','bootstrap_repetitions':len(diff)}


def main():
    root,out=map(lambda p:Path(p).resolve(),sys.argv[1:3])
    manifest=verified_inputs(root,out)
    import torch,trl
    from trl_kernel import actual_trl_advantages,save_provenance
    kernel=Path(trl.__file__).parent/'trainer/grpo_trainer.py'
    save_provenance(kernel,out/'actual_trl_tensor_kernel.json')
    rows=read_jsonl(root/'runs/B_seed17_6M/rollouts.jsonl')
    repaired=read_jsonl(root/'audits/reward_repair_v2_20261021/outcomes.jsonl')
    scores={m:{(r['source_row'],r['view']):r for r in read_jsonl(out/f'scores_{m}.jsonl')} for m in ['h4','qwen7b']}
    enc={m:json.loads((out/f'encoding_{m}.json').read_text()) for m in scores}
    reviews=json.loads((out/'reviewed_records.json').read_text())
    quarantine=json.loads((root/'audits/reward_repair_v2_20261021/quarantine.json').read_text())
    # The quarantined ID is fixed before this audit; retain original data and whole groups.
    quarantine_ids={'math/train/number_theory/551'}

    def score(m,i,view):
        value=scores[m].get((i,view))
        if value is None and view=='normalized' and str(i) in enc[m]['aliases']:
            value=scores[m].get((i,'raw'))
        return value

    variants={}
    max_arithmetic_difference=0.;checked_groups=0
    for name in ['logged_B','h4_raw','qwen7b_raw','h4_repaired','qwen7b_repaired','outcome_only_raw','outcome_only_repaired']:
        result=[]
        for i,row in enumerate(rows):
            use_repaired=name.endswith('repaired')
            y=float(repaired[i]['v2_outcome']['reward']) if use_repaired else row['outcome_reward']
            p=None
            if row['valid_for_process_reward'] and not name.startswith('outcome_only'):
                if name=='logged_B':p={'step_scores':row['step_scores'],'min_score':row['min_score'],'step_count':row['step_count']}
                else:p=score('h4' if name.startswith('h4') else 'qwen7b',i,'normalized' if use_repaired else 'raw')
            unavailable=(row['valid_for_process_reward'] and not name.startswith('outcome_only') and p is None)
            reward=None if unavailable else (y+ .2*p['min_score'] if p is not None else y)
            if not row['valid_for_process_reward']:assert y==0 and reward==0
            result.append({'source_row':i,'problem_id':row['problem_id'],'outcome_reward':y,
                           'process_scored':p is not None,'min_score':None if p is None else p['min_score'],
                           'step_count':0 if p is None else p['step_count'],'reward':reward,
                           'quarantined':row['problem_id'] in quarantine_ids})
        for offset in range(0,len(rows),8):
            group=result[offset:offset+8]
            y=[r['outcome_reward'] for r in group]
            kind='all_correct' if all(v==1 for v in y) else ('all_wrong' if all(v==0 for v in y) else 'mixed')
            if any(r['reward'] is None for r in group):
                for r in group:r.update(reward_fp32=None,advantage=None,reward_rank=None,group_type=kind,group_usable=False)
                continue
            values=[r['reward'] for r in group]
            reward_tensor,adv=actual_trl_advantages(values,kernel,device='cuda:0')
            rewards=reward_tensor.cpu().tolist();advantages=adv.cpu().tolist()
            independent=cpu_advantages(rewards)
            error=max(abs(x-y) for x,y in zip(independent,advantages))
            max_arithmetic_difference=max(max_arithmetic_difference,error);checked_groups+=1
            assert error<.001,'Independent sample-std reference disagrees'
            ranks=midranks(rewards)
            for r,rv,av,rank in zip(group,rewards,advantages,ranks):
                r.update(reward_fp32=rv,advantage=av,reward_rank=rank,group_type=kind,group_usable=True)
        variants[name]=result
        write_jsonl(out/'variants'/f'{name}.jsonl',result)

    def compare(a,b,label):
        left,right=variants[a],variants[b]
        usable=[];details=[]
        for offset in range(0,len(rows),8):
            aa,bb=left[offset:offset+8],right[offset:offset+8]
            if any(r['quarantined'] or not r['group_usable'] for r in aa+bb):continue
            usable.extend(range(offset,offset+8))
            ar=[r['reward_fp32'] for r in aa];br=[r['reward_fp32'] for r in bb]
            process_indices=[j for j in range(8) if aa[j]['process_scored'] and bb[j]['process_scored']]
            pa=[aa[j]['min_score'] for j in process_indices];pb=[bb[j]['min_score'] for j in process_indices]
            pair_changes=sum(relation(ar[i],ar[j])!=relation(br[i],br[j]) for i in range(8) for j in range(i+1,8))
            process_changes=sum(relation(pa[i],pa[j])!=relation(pb[i],pb[j]) for i in range(len(pa)) for j in range(i+1,len(pa)))
            top_a={i for i,v in enumerate(ar) if abs(v-max(ar))<=1e-6};top_b={i for i,v in enumerate(br) if abs(v-max(br))<=1e-6}
            details.append({'comparison':label,'group':offset//8,'problem_id':aa[0]['problem_id'],'group_type':aa[0]['group_type'],
                            'reward_pair_order_changes':pair_changes,'process_pair_order_changes':process_changes,
                            'reward_top_set_changed':top_a!=top_b,
                            'reward_rank_spearman':pearson(midranks(ar),midranks(br)),
                            'process_rank_spearman':pearson(midranks(pa),midranks(pb)) if len(pa)>=2 else None,
                            'advantage_sign_flips':sum(sign(x['advantage'])!=sign(y['advantage']) for x,y in zip(aa,bb)),
                            'mean_abs_advantage_change':statistics.mean(abs(x['advantage']-y['advantage']) for x,y in zip(aa,bb)),
                            'max_abs_advantage_change':max(abs(x['advantage']-y['advantage']) for x,y in zip(aa,bb))})
        def aggregate(selected):
            return {'groups':len(selected),
                'reward_order_changed_groups':sum(r['reward_pair_order_changes']>0 for r in selected),
                'process_order_changed_groups':sum(r['process_pair_order_changes']>0 for r in selected),
                'top_reward_set_changed_groups':sum(r['reward_top_set_changed'] for r in selected),
                'advantage_sign_flips':sum(r['advantage_sign_flips'] for r in selected),
                'mean_rank_spearman':statistics.mean([r['reward_rank_spearman'] for r in selected if r['reward_rank_spearman'] is not None]) if any(r['reward_rank_spearman'] is not None for r in selected) else None}
        summary=aggregate(details)
        summary.update({'rows':len(usable),'reward_changed_rows_gt_1e_minus_6':sum(abs(left[i]['reward_fp32']-right[i]['reward_fp32'])>1e-6 for i in usable),
                        'advantage_changed_rows_gt_0p01':sum(abs(left[i]['advantage']-right[i]['advantage'])>.01 for i in usable),
                        'mean_abs_advantage_change':statistics.mean(abs(left[i]['advantage']-right[i]['advantage']) for i in usable),
                        'max_abs_advantage_change':max(abs(left[i]['advantage']-right[i]['advantage']) for i in usable),
                        'by_group_type':{k:aggregate([r for r in details if r['group_type']==k]) for k in ['mixed','all_correct','all_wrong']}})
        return summary,details

    comparisons={};group_details=[]
    for a,b,label in [('h4_raw','qwen7b_raw','model_only_raw'),('h4_repaired','qwen7b_repaired','model_only_repaired'),
                      ('logged_B','h4_raw','historical_H4_reinference_drift')]:
        comparisons[label],detail=compare(a,b,label);group_details.extend(detail)
    write_csv(out/'group_changes.csv',group_details)

    review_details=[];step_details=[]
    for review in reviews:
        i=review['source_row'];labels=review['review']['step_labels']
        h4=score('h4',i,'raw');qwen=score('qwen7b',i,'raw')
        assert h4 is not None and qwen is not None
        assert len(labels)==h4['step_count']==qwen['step_count']
        verdict=review['review']['chain_verdict']
        issue=review['review'].get('input_issue') is not None or rows[i]['problem_id'] in quarantine_ids
        first=next((j for j,l in enumerate(labels) if l=='E'),None)
        determinate_first=(first is not None and all(l=='C' for l in labels[:first]))
        record={'review_id':review['review_id'],'source_row':i,'category':review['sampling_category'],
                'verdict':verdict,'excluded_question_issue':issue,'first_error':None if first is None else first+1,
                'determinate_first_error':determinate_first,'outcome_original':rows[i]['outcome_reward'],
                'outcome_repaired':repaired[i]['v2_outcome']['reward'],'group_type':variants['h4_raw'][i]['group_type']}
        prefix_valid=True
        for j,label in enumerate(labels):
            if label!='C':prefix_valid=False
            step_details.append({'review_id':review['review_id'],'source_row':i,'category':review['sampling_category'],
                                 'step':j+1,'label':label,'certain_valid_prefix':prefix_valid,'excluded_question_issue':issue,
                                 'h4_score':h4['step_scores'][j],'qwen7b_score':qwen['step_scores'][j],
                                 'text':review['steps'][j]})
        for name,p in [('h4',h4),('qwen7b',qwen)]:
            first_reject=next((j+1 for j,v in enumerate(p['step_scores']) if v<.5),None)
            record.update({f'{name}_min':p['min_score'],f'{name}_first_reject':first_reject,
                           f'{name}_first_error_score':None if first is None else p['step_scores'][first],
                           f'{name}_reward':variants[name+'_raw'][i]['reward_fp32'],
                           f'{name}_rank':variants[name+'_raw'][i]['reward_rank'],
                           f'{name}_advantage':variants[name+'_raw'][i]['advantage']})
        review_details.append(record)
    write_csv(out/'review_cases.csv',review_details)
    write_csv(out/'review_steps.csv',step_details)
    atomic_json(out/'review_cases.json',review_details)

    review_summary={}
    for category in ['uniform_random','targeted_combined','all_reviewed']:
        selected=[r for r in review_details if not r['excluded_question_issue'] and
                  (category=='all_reviewed' or (r['category']=='uniform_random')==(category=='uniform_random'))]
        selected_ids={r['review_id'] for r in selected}
        valid=[r for r in selected if r['verdict']=='clean_valid']
        wrong=[r for r in selected if r['verdict']=='contains_errors']
        certain=[r for r in wrong if r['determinate_first_error']]
        valid_steps=[r for r in step_details if r['review_id'] in selected_ids and r['certain_valid_prefix']]
        summary={'reviewed_usable':len(selected),'clean_solutions':len(valid),'erroneous_solutions':len(wrong),
                 'uncertain_solutions':len(selected)-len(valid)-len(wrong),'determinate_first_error_solutions':len(certain),
                 'certain_valid_prefix_steps':len(valid_steps)}
        for name in ['h4','qwen7b']:
            key='h4_score' if name=='h4' else 'qwen7b_score'
            summary[name]={'clean_solutions_falsely_rejected_at_0p5':sum(r[f'{name}_min']<.5 for r in valid),
                           'erroneous_solutions_fully_passed_at_0p5':sum(r[f'{name}_min']>=.5 for r in wrong),
                           'first_error_steps_missed_at_0p5':sum(r[f'{name}_first_error_score']>=.5 for r in certain),
                           'first_error_location_exact_at_0p5':sum(r[f'{name}_first_reject']==r['first_error'] for r in certain),
                           'valid_prefix_steps_falsely_rejected_at_0p5':sum(r[key]<.5 for r in valid_steps),
                           'clean_solutions_negative_advantage':sum(r[f'{name}_advantage'] < -1e-6 for r in valid),
                           'erroneous_solutions_positive_advantage':sum(r[f'{name}_advantage'] > 1e-6 for r in wrong)}
        auc_rows=[{'valid':r['verdict']=='clean_valid','h4':r['h4_min'],'qwen':r['qwen7b_min']} for r in valid+wrong]
        summary['chain_AUROC']=paired_bootstrap_auc(auc_rows)
        correct_rows=[{'valid':r['verdict']=='clean_valid','h4':r['h4_min'],'qwen':r['qwen7b_min']}
                      for r in valid+wrong if r['outcome_repaired']==1]
        summary['answer_correct_internal_AUROC']=paired_bootstrap_auc(correct_rows)
        review_summary[category]=summary
    atomic_json(out/'review_summary.json',review_summary)

    # Former known errors are descriptive regressions, not extra labelled random examples.
    known=[]
    for i in manifest['extra_known_cases']:
        item={'source_row':i,'problem_id':rows[i]['problem_id']}
        for m in ['h4','qwen7b']:
            p=score(m,i,'raw');r=variants[m+'_raw'][i]
            item.update({m+'_min':p['min_score'],m+'_rank':r['reward_rank'],m+'_advantage':r['advantage']})
        known.append(item)
    atomic_json(out/'known_error_cases.json',known)
    receipts={m:json.loads((out/f'receipt_{m}.json').read_text()) for m in ['h4','qwen7b']}
    summary={'complete':True,'source_rows':len(rows),'source_groups':len(rows)//8,'scored_raw_rows':len(manifest['eligible_rows']),
             'comparisons':comparisons,'review':review_summary,'known_error_cases':known,
             'resources':receipts,'independent_advantage_check':{'groups':checked_groups,'max_FP64_vs_GPU_FP32_difference':max_arithmetic_difference},
             'new_rollouts':0,'model_weight_updates':0,'new_training':False,
             'limitations':['50 prior single-reviewer Codex labels are not independent human gold.',
                            'Targeted samples were selected by old H4 scores; never treat them as population error rates.',
                            'Scores are evaluated on trajectories from old B, not on a policy trained with Qwen PRM.',
                            'Native serialization differs by model; this compares complete native scorer pipelines.',
                            'Positive advantage for an answer-wrong solution is not by itself proof of wrong process credit.',
                            'Group ranking changes have no group-wide process correctness gold; direction is checked only on reviewed examples.',
                            'Single-scorer GPU memory and speed exclude actor/RL costs.',
                            'Process reward may remain full-strength after normalization in same-outcome groups.']}
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'analysis_QA.json',{'passed':True,'fixed_review_steps_checked':len(step_details),
        'original_eight_partners_preserved':True,'scores_match_review_step_counts':True,
        'actual_installed_TRL_tensor_kernel_used':True,'independent_reference_groups':checked_groups,
        'max_independent_advantage_difference':max_arithmetic_difference,'new_training':False})
    print(json.dumps({'complete':True,'comparisons':comparisons,'review':review_summary},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
