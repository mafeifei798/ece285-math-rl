"""Separate mathematical ranking, reward centering, and variance amplification.

Uses only completed v3 fixed-response replay and existing PRM probabilities.
No model loading, answer generation, optimizer, backward, or training entry.
"""
from collections import Counter
import csv
import hashlib
import html
import itertools
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/"execution/reward_repair_v3_20261021"
AUDIT=ROOT/"execution/b7b_training_audit_20261021"
OUT=ROOT/"execution/b_signal_disentangle_20261021"
sys.path.insert(0,str(ROOT/"experiments/reward_repair_v3_20261021"))
from repairs import proposed_bounded_advantages

read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(s) for s in Path(p).open() if s.strip()]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def kind(values):
    return 'all_correct' if sum(values)==8 else 'all_wrong' if sum(values)==0 else 'mixed'


def summary(values):
    v=np.asarray(values,dtype=float)
    return {'n':len(v),'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),
            'p99':float(np.quantile(v,.99)),'min':float(v.min()),'max':float(v.max())} if len(v) else None


def make_reviews():
    # These are single-assistant mathematical judgments, not independent human
    # labels. Initial PRM scores were known for some previously reviewed cases.
    # E = an explicit false mathematical claim somewhere in the whole trace;
    # recovery is separate. G = proof gap, T = terminology ambiguity; neither
    # enters the primary clean-vs-explicit-error pairwise comparison.
    groups={
      896:{
        896:('V','valid','1-1/3-5/12=1/4; arithmetic and code are valid.'),
        897:('V','valid','All fraction equalities and the complement rule are correct.'),
        898:('V','valid','Common-denominator subtraction is correct.'),
        899:('E','isolated_symbol_error_recovered','Explicitly states 1/3 != 4/12, which is false; later uses the equality correctly. This may be a symbol typo, not evidence of broad reasoning inability.'),
        900:('V','valid','Subtraction, simplification, and displayed code output are correct.'),
        901:('V','valid','Opening typography is messy, but mathematical statements and final result are correct.'),
        902:('V','valid','Derivation and sum-to-one check are correct.'),
        903:('V','valid','Derivation and Python output0.25 are correct.')},
      4984:{i:('V','valid','250g=0.25kg and22000*0.25=5500; full response has no mathematical error. Formatting variants are not math errors.') for i in range(4984,4992)},
      6344:{
        6344:('E','explicit_error','Infers both m+n and m-n are multiples of4 merely because they are even; m=3,n=1 is a counterexample.'),
        6345:('G','proof_gap','Examples and a divisor4 argument do not prove universal divisibility by8. No clean complete proof. Excluded from primary inversion counts.'),
        6346:('E','explicit_error','Claims even factors alone imply an extra factor2; shown Python output is impossible for that loop, which produces an empty list.'),
        6347:('G','proof_gap','Finite examples plus divisibility by4 do not prove all odd-square differences divisible by8. Excluded from primary inversion counts.'),
        6348:('E','explicit_error','Claims odd+odd can be odd and prints false differences; e.g.7^2-5^2=24, not32.'),
        6349:('E','error_recovered','Claims a+b+1 and a-b can both be odd, although their difference2b+1 is odd. Incorrectly concludes universal divisor4 before ending with8.'),
        6350:('E','explicit_error','36 mod8 is4;80 is divisible by16. Final8 does not validate these false proof steps.'),
        6351:('E','explicit_error','Claims all odd integers are of the form3c+/-1 and all differences divisible by3. m=3,n=1 disproves the latter.')},
      8744:{
        8744:('V','valid','Independent multiplication0.6*0.25=0.15=15% is correct throughout.'),
        8745:('V','valid','Independent multiplication and conversion to percent are correct throughout.'),
        8746:('E','false_start_recovered','Opening50% is false; subsequent reasoning correctly obtains15%.'),
        8747:('T','terminology_slip','Says dependent events but then explicitly uses independence and the correct formula. Treat as ambiguous wording, not a hard mathematical-error label.'),
        8748:('V','valid','A malformed early box has no mathematical content; the actual derivation and final15 are correct.'),
        8749:('E','error_recovered','Claims a+b-ab=15%, although0.6+0.25-0.15=0.70; later corrects to the intersection formula.'),
        8750:('E','false_start_recovered','Opening boxed2% is false; later derives15% correctly.'),
        8751:('V','valid','Independence, arithmetic, and percent conversion are correct throughout.')}
    }
    return groups


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'complete.json').exists(),'Completed analysis exists; inspect, do not repeat'
    torch.set_num_threads(2)
    sources=[OLD/'variants/combined.jsonl',OLD/'paired_scores.jsonl',OLD/'final_receipt.json',
        AUDIT/'inputs/runs/B_Qwen7B_seed17_6M/rollouts.jsonl',
        AUDIT/'inputs/data/prepared/train.jsonl',ROOT/'execution/qwen7b_offline_20261021/review_summary.json']
    protected={str(p.relative_to(ROOT)):sha(p) for p in sources}
    fixed=rows(sources[0]);original=rows(sources[3]);paired={x['source_row']:x for x in rows(sources[1])}
    questions={x['problem_id']:x for x in rows(sources[4])}
    n=len(fixed);assert n==9056 and len(original)==n
    for i,r in enumerate(fixed):assert r['source_row']==i and r['problem_id']==original[i]['problem_id']
    y=torch.tensor([r['outcome_reward'] for r in fixed],dtype=torch.float32).view(-1,8)
    m=torch.tensor([r['min_score'] if r['process_scored'] else 0 for r in fixed],dtype=torch.float32).view(-1,8)
    r=torch.tensor([r['reward_fp32'] for r in fixed],dtype=torch.float32).view(-1,8)
    std=r.std(1,keepdim=True);ym=y-y.mean(1,keepdim=True)
    pm=m-m.mean(1,keepdim=True)
    original_a=torch.tensor([r['advantage'] for r in fixed],dtype=torch.float32).view(-1,8)
    reference=(r-r.mean(1,keepdim=True))/(std+1e-4)
    fp64=r.numpy().astype(np.float64)
    fp64=(fp64-fp64.mean(1,keepdims=True))/(fp64.std(1,ddof=1,keepdims=True)+1e-4)
    numeric={'stored_vs_local_FP32_max':float((original_a-reference).abs().max()),
        'stored_vs_independent_FP64_max':float(np.abs(original_a.numpy()-fp64).max()),
        'local_FP32_vs_independent_FP64_max':float(np.abs(reference.numpy()-fp64).max()),
        'sign_changes_on_stored_abs_A_above_1e_minus_3':int(((original_a.sign()!=reference.sign())&(original_a.abs()>1e-3)).sum()),
        'policy':'Preserve saved actual advantages. Cross-platform FP32 differences are recorded against independent FP64, not required to be bitwise identical. No tensor in any trainer is changed.'}
    assert max(numeric[k] for k in ['stored_vs_independent_FP64_max','local_FP32_vs_independent_FP64_max'])<1e-3
    assert numeric['sign_changes_on_stored_abs_A_above_1e_minus_3']==0
    ay=ym/(y.std(1,keepdim=True)+1e-4)
    bounded,ap=proposed_bounded_advantages(y.flatten(),m.flatten())
    bounded,ap=bounded.view(-1,8),ap.view(-1,8)
    homogeneous=(y.sum(1)==0)|(y.sum(1)==8)
    only_homogeneous=original_a.clone();only_homogeneous[homogeneous]=ap[homogeneous]
    homogeneous_off=original_a.clone();homogeneous_off[homogeneous]=0
    hold_outcome=ym/(std+1e-4)+ap
    scale_small=y+.02*m
    scale_small=(scale_small-scale_small.mean(1,keepdim=True))/(scale_small.std(1,keepdim=True)+1e-4)
    variants={'outcome_only_same_rollouts':ay,'repaired_original_B':original_a,
        'homogeneous_only_bounded':only_homogeneous,'bounded_process_hold_old_outcome_denominator':hold_outcome,
        'bounded_residual_recommended':bounded,'homogeneous_off_diagnostic_only':homogeneous_off,
        'lambda_0p02_diagnostic_only':scale_small}
    counts=Counter();group_details=[];signal_stats={}
    for name,a in variants.items():
        stats={}
        for k in ['mixed','all_correct','all_wrong']:
            ids=[g for g in range(n//8) if kind(y[g].tolist())==k]
            yy=y[ids];aa=a[ids]
            stats[k]={'groups':len(ids),'mean_abs_advantage':float(aa.abs().mean()),
                'max_abs_advantage':float(aa.abs().max()),
                'correct_negative_rows':int(((yy==1)&(aa < -1e-6)).sum()),
                'wrong_positive_rows':int(((yy==0)&(aa > 1e-6)).sum())}
        signal_stats[name]=stats
    for g in range(n//8):
        k=kind(y[g].tolist());counts[k]+=1
        rawcenter=.2*pm[g];ratio=1/(float(std[g])+1e-4)
        quantum=r[g]-y[g];qcenter=quantum-quantum.mean()
        process_old=qcenter/(std[g]+1e-4)
        record={'group':g,'source_start':g*8,'problem_id':fixed[g*8]['problem_id'],'kind':k,
            'all_process_scored':all(x['process_scored'] for x in fixed[g*8:g*8+8]),
            'score_std':float(m[g].std()),'reward_std':float(std[g]),
            'process_residual_gain_from_normalization':ratio,'nonzero_original_advantage':bool((original_a[g].abs()>1e-6).any()),
            'mean_abs_original_advantage':float(original_a[g].abs().mean()),
            'mean_abs_bounded_advantage':float(bounded[g].abs().mean()),
            'tenfold_coefficient_mean_abs_A_ratio':float(scale_small[g].abs().mean()/original_a[g].abs().mean()) if original_a[g].abs().mean()>1e-8 else None,
            'quantized_decomposition_max_error':float((original_a[g]-ym[g]/(std[g]+1e-4)-process_old).abs().max())}
        group_details.append(record)
        # Same-group centering and positive division cannot reverse reward order.
        for i,j in itertools.combinations(range(8),2):
            counts['reward_pairs_checked']+=1
            if r[g,i]!=r[g,j]:assert bool(r[g,i]>r[g,j])==bool(original_a[g,i]>original_a[g,j])
    selected=sorted(random.Random(286).sample([g*8 for g in range(n//8) if y[g].sum()==8 and g*8 not in [6344,8744]],2))
    assert selected==[896,4984]
    labels=make_reviews();reviewed=[];rank_stats={}
    for start,review in labels.items():
        candidates=[]
        for i,(label,subtype,reason) in review.items():
            p=paired[i]['normalized']['step_scores'] if i in paired else original[i]['step_scores']
            assert min(p)==fixed[i]['min_score']
            steps=[s.strip() for s in original[i]['completion'].replace('\r\n','\n').split('\n\n') if s.strip()]
            assert len(steps)==len(p)
            record={'source_row':i,'group_start':start,'label':label,'subtype':subtype,'reason':reason,
                'outcome':fixed[i]['outcome_reward'],'min_score':fixed[i]['min_score'],
                'original_advantage':float(original_a.flatten()[i]),'bounded_advantage':float(bounded.flatten()[i]),
                'question':questions[original[i]['problem_id']]['problem'],'completion':original[i]['completion'],
                'steps':[{'text':t,'score':s} for t,s in zip(steps,p)]}
            candidates.append(record);reviewed.append(record)
        valid=[v for v in candidates if v['label']=='V'];errors=[v for v in candidates if v['label']=='E']
        pairs=[{'valid_row':v['source_row'],'error_row':e['source_row'],'valid_score':v['min_score'],
                'error_score':e['min_score'],'inverted':v['min_score']<e['min_score'],
                'tie':v['min_score']==e['min_score']} for v in valid for e in errors]
        rank_stats[str(start)]={'valid':len(valid),'explicit_error':len(errors),
            'proof_gap_or_terminology_excluded':len(candidates)-len(valid)-len(errors),
            'pairs':len(pairs),'inversions':sum(p['inverted'] for p in pairs),'details':pairs,
            'interpretation':'Within-question descriptive pairs, not independent observations. No ranking-accuracy estimate if no clean/error pairs.'}
    gains={}
    for k in ['mixed','all_correct','all_wrong']:
        v=[g for g in group_details if g['kind']==k and g['nonzero_original_advantage']]
        gains[k]={'normalization_gain':summary([g['process_residual_gain_from_normalization'] for g in v]),
            'gain_above_10_groups':sum(g['process_residual_gain_from_normalization']>10 for g in v),
            'gain_above_100_groups':sum(g['process_residual_gain_from_normalization']>100 for g in v),
            'tenfold_coefficient_actual_amplitude_ratio':summary([g['tenfold_coefficient_mean_abs_A_ratio'] for g in v])}
    first_error_comparison=read(sources[-1])['uniform_random']
    summary_all={'complete':True,'scope':'Fixed v3 repaired7B-B answers and scores; no trained-policy performance claim',
        'rows':n,'groups':n//8,'signal_stats':signal_stats,'normalization':gains,'within_question_rank_reviews':rank_stats,
        'cross_platform_numeric_check':numeric,
        'reward_order_reversed_by_advantage':0,'reward_pairs_checked':counts['reward_pairs_checked'],
        'proposed_process_term_bound':.175,'observed_max_process_term':float(ap.abs().max()),
        'prior_H4_vs_Qwen_review':first_error_comparison,
        'review_protocol':{'new_groups':selected,'new_group_selection':'random.Random(286), uniform2 of31 other repaired all-correct groups',
            'targeted_previously_known_groups':[6344,8744],'reviewer':'Codex single reviewer; not independent human gold',
            'score_blinding':'Not a blinded independent study; some earlier case scores were known. Mathematical reasons provided for every label.',
            'ranking_definition':'Whole trace contains an explicit mathematical false statement vs fully valid trace; recovered false starts remain explicit errors, with subtype. Proof gaps and wording ambiguity excluded.',
            'not_population_estimates':True},
        'limitations':['Offline results identify score errors and signal amplification, not causal shares of B-minus-A accuracy.',
            'Advantage amplitudes are not parameter gradient contributions; no backward or optimizer was run.',
            'All-error groups do not establish a valid-vs-invalid ranking inversion.',
            'One literal false inequality may be a symbol typo; it does not prove broad inability to judge mathematics.',
            'Across-question AUROC in prior sparse reviews is not within-question ranking accuracy.',
            'A negative advantage on one valid trace is not alone proof of a harmful policy update.',
            'The bounded rule preserves existing PRM within-outcome ordering; it cannot repair a wrong ordering.'],
        'new_training':False,'actor_generation':0,'PRM_forward_calls':0,'PRM_updates':0}
    assert signal_stats['bounded_residual_recommended']['mixed']['correct_negative_rows']==0
    assert signal_stats['bounded_residual_recommended']['mixed']['wrong_positive_rows']==0
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    save('summary.json',summary_all);save('group_details.json',group_details);save('reviewed_groups.json',reviewed)
    save('input_manifest.json',{'sha256':protected,'script_sha256':sha(__file__),'originals_preserved':True})
    with (OUT/'per_response.csv').open('w') as stream:
        names=['source_row','problem_id','outcome','process_minimum',*variants]
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
        flat={k:v.flatten().tolist() for k,v in variants.items()}
        for i,rr in enumerate(fixed):writer.writerow({'source_row':i,'problem_id':rr['problem_id'],
            'outcome':rr['outcome_reward'],'process_minimum':rr['min_score'],**{k:v[i] for k,v in flat.items()}})
    save('complete.json',{'complete':True,'new_training':False,'new_PRM_inference':False,
        'originals_preserved':True,'rows':n,'manually_reviewed_rows':len(reviewed),
        'max_local_FP32_advantage_replay_difference':float((original_a-reference).abs().max()),
        'reward_order_reversals':0,'summary_sha256':sha(OUT/'summary.json')})
    print(json.dumps({'signal_stats':signal_stats,'normalization':gains,'within_question_rank_reviews':rank_stats},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
