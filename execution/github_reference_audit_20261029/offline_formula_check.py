"""CPU stdlib formula audit on fixed historical C scores; NOT policy evaluation.

This transcribes PAPO's inspected core function. It does not import/execute the
official repository or claim that a transplanted formula reproduces its paper.
PAPO native epsilon1e-6; our fixed audited outcome epsilon1e-4. Shared lambda.2
is additionally compared with PAPO's native default lambda1. No model calls.
"""
from pathlib import Path
import json,math,hashlib,statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'execution/cd_prm_7b_pair_20261028/artifacts/group_advantages.jsonl'

def std(v):
    mu=sum(v)/len(v)
    return math.sqrt(sum((x-mu)**2 for x in v)/(len(v)-1)) if len(v)>1 else 0.

def components(y,m,method,coefficient):
    mean=sum(y)/len(y);sd=std(y)
    if method=='ours':out=[(x-mean)/(sd+1e-4) for x in y]
    else:out=[((x-mean)/(sd+1e-6) if method=='papo_std' else x-mean) if sd>1e-6 else 0. for x in y]
    ix=[i for i,v in enumerate(y) if v==1];proc=[0.]*len(y)
    if len(ix)>=2:
        vals=[m[i] for i in ix];mu=sum(vals)/len(vals);s=std(vals)
        if method=='ours' or s>1e-6:
            for i in ix:proc[i]=coefficient*(m[i]-mu)/(s+1e-6 if method=='papo_std' else 1.)
    return out,proc,[a+b for a,b in zip(out,proc)]

def main():
    groups=[json.loads(s) for s in SOURCE.open() if s.strip()]
    groups=[g for g in groups if g['arm']=='C'];assert len(groups)==67
    variants={'ours_lambda0.2':('ours',.2),'PAPO_std_lambda0.2':('papo_std',.2),
        'PAPO_std_lambda1_native_default':('papo_std',1.),'PAPO_no_std_both_lambda0.2':('papo_no_std',.2)}
    result={};pergroup=[]
    for name,(method,coefficient) in variants.items():
        wrong_positive=0;correct_negative_mixed=0;process_abs=[];total_abs=[];active=0
        for g in groups:
            y,m=g['outcomes'],g['min_scores'];assert len(y)==len(m)==8
            out,proc,total=components(y,m,method,coefficient)
            wrong_positive+=sum(v==0 and a>0 for v,a in zip(y,total))
            if g['kind']=='mixed':correct_negative_mixed+=sum(v==1 and a<0 for v,a in zip(y,total))
            process_abs.extend(abs(a) for a in proc);total_abs.extend(abs(a) for a in total)
            active+=any(abs(a)>1e-12 for a in proc)
            if method=='ours':assert max(map(abs,proc))<=.2+1e-12
            pergroup.append(dict(variant=name,dataset=g['dataset'],group_id=g['group_id'],kind=g['kind'],
                outcomes=y,process_scores=m,outcome_component=out,process_component=proc,advantages=total))
        assert wrong_positive==0
        result[name]=dict(groups=len(groups),rows=len(process_abs),wrong_positive=wrong_positive,
            correct_negative_in_mixed_groups=correct_negative_mixed,process_active_groups=active,
            process_abs_mean=statistics.mean(process_abs),process_abs_max=max(process_abs),
            total_abs_mean=statistics.mean(total_abs))
    receipt=dict(passed=True,scope='Formula-only FP64 transcription on fixed scores; not official code execution, training, or accuracy comparison',
        source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),group_count=67,response_count=536,
        native_PAPO_epsilon=1e-6,our_epsilon=1e-4,shared_lambda_comparison=.2,variants=result,
        limitations=['PAPO uses a rubric-based generative grader in its published recipe; these are our frozen discriminative C scores.',
            'Same lambda means an explicit numerical control, not equal gradient norm or equal paper configurations.',
            'Larger or smaller process magnitude does not prove better mathematical reasoning.',
            'PAPO no-std flag changes BOTH outcome and process components; it is not identical to our hybrid scaling.'])
    (HERE/'offline_formula_check.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'offline_formula_groups.jsonl').open('w') as f:
        for x in pergroup:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
