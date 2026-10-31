"""Auditable signal/format metrics; these metrics never enter the reward."""
from collections import Counter
import math,re
import torch

REFUSAL=re.compile(r"(?:i (?:cannot|can't|am unable to|won't)|sorry[, ]|unable to (?:solve|answer|provide))",re.I)

def _mean(t):return float(t.float().mean()) if t.numel() else None

def summarize_signal(result,records):
    if len(records)!=result['outcomes'].numel():raise ValueError('Metric records lost alignment')
    y=result['outcomes'].view(-1,8);a=result['advantages'].view(-1,8)
    proc=result['process_component'].view(-1,8);out=result['outcome_component'].view(-1,8)
    delta=result['deviation_from_outcome'].view(-1,8);kinds=[]
    for g in y:
        n=int(g.sum());kinds.append('all_wrong' if n==0 else 'all_correct' if n==8 else 'mixed')
    by_kind={}
    for kind in ['mixed','all_correct','all_wrong']:
        select=torch.tensor([k==kind for k in kinds],dtype=torch.bool,device=y.device)
        yy,aa,pp,oo,dd=y[select],a[select],proc[select],out[select],delta[select]
        outcome_l1=float(oo.abs().sum());process_l1=float(pp.abs().sum())
        by_kind[kind]=dict(groups=int(select.sum()),rows=yy.numel(),wrong_rows=int((yy==0).sum()),
            wrong_positive=int(((yy==0)&(aa>0)).sum()),answer_correct_negative=int(((yy==1)&(aa<0)).sum()),
            process_abs_mean=_mean(pp.abs()) if result['process_component_is_separable'] else None,outcome_abs_mean=_mean(oo.abs()),
            deviation_from_outcome_abs_mean=_mean(dd.abs()),
            process_to_outcome_L1=process_l1/outcome_l1 if outcome_l1>0 and result['process_component_is_separable'] else None,
            process_active_groups=(int((pp.abs().sum(1)>1e-8).sum()) if pp.numel() else 0) if result['process_component_is_separable'] else None)
    correct=y.flatten().bool();flat_a=a.flatten()
    short_wrong=[i for i,r in enumerate(records) if r['completion_tokens']<=32 and not r['outcome_reward']]
    refusal=[i for i,r in enumerate(records) if REFUSAL.search(r['completion'])]
    no_answer=[i for i,r in enumerate(records) if r.get('answer') is None]
    return dict(mode=result['config']['mode'],rows=len(records),group_types=dict(Counter(kinds)),by_group_type=by_kind,
        process_component_is_separable=result['process_component_is_separable'],
        wrong_positive=int(((~correct)&(flat_a>0)).sum()),
        correct_subset_std_median=float(result['correct_subset_std'].median()),
        process_abs_max=float(proc.abs().max()) if result['process_component_is_separable'] else None,
        process_active_rows=int((proc.abs()>1e-8).sum()) if result['process_component_is_separable'] else None,
        process_scored_rows=int(result['process_scored'].sum()),post_combination_normalizations=0,
        empty_outputs=sum(not r['completion'].strip() for r in records),
        short_wrong_outputs=len(short_wrong),short_wrong_positive=sum(float(flat_a[i])>0 for i in short_wrong),
        refusal_keyword_candidates=len(refusal),refusal_candidate_positive=sum(float(flat_a[i])>0 for i in refusal),
        no_extractable_answer=len(no_answer),no_answer_positive=sum(float(flat_a[i])>0 for i in no_answer),
        one_segment_outputs=sum(r.get('step_count')==1 for r in records),
        raw_no_double_newline=sum('\n\n' not in r['completion'].strip() for r in records),
        truncated_outputs=sum(not r['terminated'] for r in records),
        mean_completion_tokens=sum(r['completion_tokens'] for r in records)/len(records),
        caveats=['Refusal keyword candidates are screening flags, not judged refusals.',
            'Short wrong outputs can be genuine attempts; these metrics do not by themselves establish reward hacking.',
            'Correct final answers are not step-validity labels.',
            'C-mix deviation from outcome is a change in the whole normalized signal, not an isolated process advantage.'])
