"""CPU acceptance at each milestone and paired final statistics."""
from pilot_common import *
import sys,collections,math,html

def main():
    guard();verify();assert not __import__('torch').cuda.is_initialized()
    step=int(sys.argv[1]);assert step in [144,288];out=PAIR/'evaluation'/str(step)
    assert not (out/'acceptance.json').exists(),'Preserve existing CPU audit'
    e=read(out/'summary.json');assert e['passed'];contract=read(out/'contract.json')
    sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'));from repairs import repaired_outcome
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json');regraded=0;all_preds={};accounting={}
    for arm in ARMS:
        result=read(PAIR/arm/f'stage_{step}_result.json');assert result['passed'] and result['global_step']==step
        rr=rows(PAIR/arm/'rollouts.jsonl');assert len(rr)==step*32
        schedule=read(AUDIT/'preflight.json')['first_epoch_question_groups']
        for k in range(step):
            group=rr[k*32:(k+1)*32];assert [r['problem_id'] for r in group[::8]]==schedule[k]
            assert all(r['optimizer_step_before_update']==k for r in group)
        assert sum(r['completion_tokens'] for r in rr)==result['accounting']['generated_completion_tokens']
        if arm!='C_mix':assert not any(r['outcome_reward']==0 and r['advantage']>0 for r in rr)
        accounting[arm]=result['accounting']
        for dataset,s in e['summary'][arm].items():
            p=out/arm/f'{dataset}.jsonl';assert sha(p)==s['predictions_sha256'];preds=rows(p);qs=rows(ROOT/'data/prepared'/f'{dataset}.jsonl')
            assert len(preds)==len(qs) and [r['problem_id'] for r in preds]==[q['problem_id'] for q in qs]
            for r,q in zip(preds,qs):
                assert len(r['completion_token_ids'])==r['generated_tokens']<=2048
                assert r['adapter_sha256']==contract['adapters'][arm]['sha256'] and r['question_sha256']==q['question_sha256']
                judged=repaired_outcome(r['completion'],q['answer'],r['finish_reason']=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
                for k,v in judged.items():assert r[k]==v,(arm,dataset,r['problem_id'],k)
                regraded+=1
            all_preds.setdefault(dataset,{})[arm]=preds
    protected=unchanged()
    acceptance={'passed':True,'step':step,'regraded':regraded,'same_prompt_order_and_update_counts':True,'raw_tokens_and_adapter_hashes_checked':True,'protected_files_unchanged':protected,'source_release_sha256':verify()}
    atomic(out/'acceptance.json',acceptance)
    if step==144:return
    import numpy as np
    paired={}
    for dataset,aa in all_preds.items():
        n=len(aa['A']);indices=np.random.default_rng(285).integers(0,n,size=(10000,n),dtype=np.int32);paired[dataset]={}
        for lhs,rhs in [('C_mix','A'),('C_bounded','A'),('C_bounded','C_mix')]:
            d=np.array([a['reward']-b['reward'] for a,b in zip(aa[lhs],aa[rhs])]);win=int((d>0).sum());lose=int((d<0).sum());nd=win+lose
            p=min(1.,2*sum(math.comb(nd,i) for i in range(min(win,lose)+1))/2**nd) if nd else 1.
            paired[dataset][lhs+' minus '+rhs]={'delta_pp':float(d.mean()*100),'wins':win,'losses':lose,'paired_question_bootstrap95_pp':np.quantile(d[indices].mean(1)*100,[.025,.975]).tolist(),'bootstrap_replicates':10000,'seed':285,'exact_mcnemar_two_sided_p':p}
        del indices
    times={a:sum(read(PAIR/a/f'stage_{s}_result.json')['seconds'] for s in [144,288]) for a in ARMS}
    report={'passed':True,'seed':17,'global_step':288,'summary':e['summary'],'paired':paired,'validation_144':read(PAIR/'evaluation/144/summary.json'),
        'accounting':accounting,'resumed_training_seconds_since_20261104':times,'prior_cost_evidence':read(AUDIT/'preflight.json')['prior_cost_evidence'],'acceptance':acceptance,'source_release_sha256':verify(),
        'limitations':['One training seed; intervals cover questions only, not training-seed variation.','3M/6M labels are approximate token equivalents; all arms have identical prompt schedule and update count.','V1 sampling RNG is recreated on process restart. User pause adds a restart at A step115 while C arms start at50; common prompt/update controls hold, but restart history is asymmetric.','Short wrong output is a diagnostic, not proof of hacking; correct final answers do not certify valid derivations.','No test-based checkpoint, prompt, reward or coefficient selection.','Shared-GPU timing does not measure isolated algorithm efficiency.'],'finished_unix':time.time()}
    atomic(PAIR/'comparison.json',report)
    lines=['Seed17 A/C-mix/C-bounded checkpoint continuation complete.','All arms: step288, V1, unchanged reward/loss, common prompt order.']
    for a,ds in report['summary'].items():
        lines.append(a+' training tokens='+str(accounting[a]['generated_completion_tokens'])+' '+json.dumps({n:v['accuracy'] for n,v in ds.items()}))
    lines.append(json.dumps(paired));lines+=report['limitations']
    (PAIR/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    (PAIR/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Seed17 continuation results</title><pre>'+html.escape('\n'.join(lines))+'</pre>')
if __name__=='__main__':main()
