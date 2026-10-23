"""CPU re-verification and paired 2x2 analysis; official grading is immutable."""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/AB_prompt_validation_20261022'
sys.path.insert(0,str(ROOT/'training'))
from common import atomic_json,file_hash,read_jsonl
sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'))
from repairs import repaired_outcome

def box_flags(text,terminated=True):
    matches=list(re.finditer(r'\\(?:boxed|fbox)\s*\{',text));spans=[];malformed=False
    for m in matches:
        depth=1;end=None
        for i in range(m.end(),len(text)):
            if text[i]=='{':depth+=1
            elif text[i]=='}':depth-=1
            if depth==0:end=i+1;break
        if end is None or not text[m.end():end-1].strip():
            if m==matches[-1]:malformed=True
        else:spans.append((m.start(),end))
    valid_last=bool(matches and spans and not malformed and spans[-1][0]==matches[-1].start())
    end=spans[-1][1] if valid_last else None
    suffix=text[end:] if valid_last else None
    no_text_after=valid_last and bool(re.fullmatch(r'[\s.$\\\[\](){}!,;:`]*',suffix))
    return {'box_count':len(matches),'valid_last_box':valid_last,
        'single_terminal_box':bool(terminated and len(matches)==1 and no_text_after),
        'early_last_box':bool(valid_last and end<=0.25*len(text) and len(text)-end>=100),
        'starts_with_box':bool(matches and matches[0].start()<80),
        'multiple_boxes':len(matches)>1,'nontrivial_text_after_last_box':bool(valid_last and not no_text_after),
        'last_box_end_character_fraction':end/max(len(text),1) if valid_last else None}

def self_test():
    assert box_flags(r'We compute. \boxed{1}')['single_terminal_box']
    assert box_flags(r'\[\boxed{1+{2}}\]')['single_terminal_box']
    assert not box_flags(r'\boxed{1} now redo it')['single_terminal_box']
    assert not box_flags(r'\boxed{1} then \boxed{2')['valid_last_box']
    assert box_flags(r'\boxed{1}'+' text'*60)['early_last_box']
    assert not box_flags('Nothing here')['valid_last_box']
    assert not box_flags(r'\boxed{1}',False)['single_terminal_box']
    assert not box_flags(r'\boxed{1} \boxed{2}')['single_terminal_box']
    print('8 structural-metric checks passed')

def main():
    args=argparse.ArgumentParser();args.add_argument('--self-test',action='store_true');cli=args.parse_args()
    if cli.self_test:self_test();return
    import numpy as np
    before={p.name:file_hash(p) for p in [OUT/'predictions.jsonl',OUT/'questions.jsonl',OUT/'unit_metadata.json',OUT/'protocol.json',OUT/'release.json']}
    release=json.loads((OUT/'release.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    for p,h in release['input_sha256'].items():assert file_hash(ROOT/p)==h
    summary=json.loads((OUT/'evaluation_summary.json').read_text());assert summary['status']=='completed'
    assert summary['predictions_sha256']==before['predictions.jsonl']
    rows=read_jsonl(OUT/'predictions.jsonl');queries=read_jsonl(OUT/'questions.jsonl');byq={r['problem_id']:r for r in queries}
    units=json.loads((OUT/'unit_metadata.json').read_text());by={};check_count=0
    for row in rows:
        key=(row['cell'],row['problem_id']);assert key not in by
        q=byq[row['problem_id']]
        assert row['question_sha256']==q['question_sha256'] and row['gold']==q['answer']
        fresh=repaired_outcome(row['completion'],q['answer'],row['finish_reason']=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
        for key2,value in fresh.items():assert row[key2]==value,(key,key2)
        assert row['status']!='verifier_error'
        by[key]=dict(row,box_flags=box_flags(row['completion'],row['finish_reason']=='stop'));check_count+=1
    assert len(by)==protocol['max_completions']==3992
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'independently_reverified':check_count,
        'predictions_sha256':before['predictions.jsonl'],'official_reward_changed':False,'training':False,'datasets':{},
        'limitations':protocol['limitations']+['Confidence intervals resample questions, not training seeds. Multiple contrasts are exploratory, unadjusted intervals.']}
    cells=protocol['cells']
    for dataset,n in protocol['datasets'].items():
        ids=[q['problem_id'] for q in queries if q['dataset']==dataset];assert len(ids)==n
        matrix=np.array([[by[(cell,pid)]['reward'] for cell in cells] for pid in ids])
        # Column order: Aoriginal,Afinal,Boriginal,Bfinal.
        vectors={'B_minus_A_original':matrix[:,2]-matrix[:,0],'B_minus_A_final_only':matrix[:,3]-matrix[:,1],
            'prompt_effect_A':matrix[:,1]-matrix[:,0],'prompt_effect_B':matrix[:,3]-matrix[:,2],
            'interaction':(matrix[:,3]-matrix[:,2])-(matrix[:,1]-matrix[:,0])}
        rng=np.random.default_rng(285);draws={k:[] for k in vectors}
        for _ in range(20):
            indices=rng.integers(0,n,size=(500,n))
            for k,v in vectors.items():draws[k].extend(v[indices].mean(1).tolist())
        comparisons={k:{'point_pp':float(v.mean()*100),'ci95_pp':[float(x*100) for x in np.quantile(draws[k],[.025,.975])],
            'positive_question_differences':int((v>0).sum()),'negative_question_differences':int((v<0).sum())} for k,v in vectors.items()}
        metrics={}
        for cell in cells:
            part=[by[(cell,pid)] for pid in ids]
            m={'n':n,'correct':int(sum(r['reward'] for r in part)),'accuracy':sum(r['reward'] for r in part)/n,
                'truncated':sum(r['finish_reason']!='stop' for r in part),'status_counts':dict(Counter(r['status'] for r in part)),
                'mean_generated_tokens':sum(r['generated_tokens'] for r in part)/n,
                'generated_tokens':sum(r['generated_tokens'] for r in part)}
            for name in ['single_terminal_box','early_last_box','starts_with_box','multiple_boxes','nontrivial_text_after_last_box','valid_last_box']:
                m[name]=sum(r['box_flags'][name] for r in part)
            assert m['correct']==summary['cells'][cell][dataset]['correct']
            metrics[cell]=m
        result['datasets'][dataset]={'n':n,'cells':metrics,'paired_contrasts':comparisons}
    protected=json.loads((OUT/'protection_manifest.json').read_text())
    assert all(file_hash(ROOT/k)==v['sha256'] for k,v in protected.items())
    result['protected_old_files_unchanged']=len(protected)
    assert all(file_hash(OUT/k)==v for k,v in before.items())
    atomic_json(OUT/'analysis.json',result)
    # Create a fixed sample with no gold, rewards, arm or prompt displayed.
    selected=json.loads((OUT/'blind_review_question_ids.json').read_text());blind=[];key=[]
    rng=random.Random(2851702)
    for qi,pid in enumerate(selected,1):
        order=cells[:];rng.shuffle(order);q=byq[pid]
        for ci,cell in enumerate(order,1):
            code=f'Q{qi:02d}-R{ci}';r=by[(cell,pid)]
            blind.append({'review_id':code,'problem':q['problem'],'completion':r['completion'],'finish_reason':r['finish_reason']})
            key.append({'review_id':code,'problem_id':pid,'cell':cell,'dataset':q['dataset'],'gold':q['answer'],'reward':r['reward']})
    atomic_json(OUT/'blind_review.json',blind);atomic_json(OUT/'blind_review_key.json',key)
    text='\n\n'.join('=== '+r['review_id']+' ===\nQUESTION\n'+r['problem']+'\nRESPONSE\n'+r['completion']+'\nFINISH '+r['finish_reason'] for r in blind)
    (OUT/'blind_review.txt').write_text(text)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
