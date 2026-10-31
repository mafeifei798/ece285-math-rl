"""Same saved responses and backend for frozen B / adapted C / adapted D."""
from common import *


def main(arm):
    assert_root();verify_release();torch=setup_torch()
    target=OUT/f'scores_{arm}.jsonl';receipt=OUT/f'scoring_{arm}.json'
    if receipt.exists():assert sha(target)==read(receipt)['sha256'];return
    tok,model,info=load_prm(None if arm=='B' else OUT/arm/'adapter',training=False)
    versions=[(p,p._version) for p in model.parameters()]
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
    data=[]
    for r in rows(OUT/'probe.jsonl'):
        data.append({**r,'dataset':'probe','eligible':r['valid_for_process_reward'],'outcome':int(r['outcome_reward'])})
    for r in rows(OUT/'illustrations.jsonl'):
        j=repaired_outcome(r['completion'],r['gold'],r['terminated'],question=r['question'],problem_id=r['problem_id'],unit_metadata=units)
        data.append({**r,'dataset':'illustrations','eligible':valid_for_prm(j),'outcome':int(j['reward']),'status':j['status']})
    for r in rows(OUT/'processbench.jsonl'):
        data.append({**r,'dataset':'processbench','question':r['problem'],'eligible':True,'outcome':int(r['final_answer_correct'])})
    saved={r['example_id']:r for r in rows(target)} if target.exists() else {}
    started=time.time();elapsed_tokens=0
    with target.open('a') as f,torch.inference_mode():
        for r in data:
            if r['example_id'] in saved:continue
            output={k:r.get(k) for k in ['example_id','dataset','group_id','source_row','row_index','known_error_case','stratum','label','outcome','status']}
            output.update(arm=arm,eligible=r['eligible'],step_scores=[],min_score=0.,input_tokens=0)
            if r['eligible']:
                if r['dataset']=='processbench':e=encode_steps(tok,r['question'],r['steps'])
                else:e=encode_qwen(tok,r['question'],r['completion'],normalize=True)
                if len(e['input_ids'])>CFG['prm_max_tokens']:
                    if r['dataset']!='processbench':
                        raise RuntimeError('Probe context overflow; no silent reward substitution: '+r['example_id'])
                    output.update(eligible=False,excluded='native_context_overflow')
                else:
                    ids=torch.tensor([e['input_ids']],device='cuda:0')
                    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,e['positions']].float()
                    assert logits.shape==(len(e['positions']),2) and torch.isfinite(logits).all()
                    scores=logits.softmax(-1)[:,1].cpu().tolist();assert scores and all(0<=x<=1 for x in scores)
                    output.update(step_scores=scores,min_score=min(scores),input_tokens=len(e['input_ids']))
                    elapsed_tokens+=len(e['input_ids'])
                    del ids,logits
            f.write(json.dumps(output,ensure_ascii=False,allow_nan=False)+'\n');f.flush();saved[r['example_id']]=output
            if len(saved)%32==0:event('scoring',arm=arm,done=len(saved),total=len(data),elapsed_seconds=time.time()-started)
    assert len(saved)==len(data)
    assert all(not p.requires_grad and p.grad is None and p._version==v for p,v in versions)
    atomic(receipt,dict(arm=arm,sha256=sha(target),rows=len(data),input_tokens_this_invocation=elapsed_tokens,elapsed_seconds_this_invocation=time.time()-started,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,native_scoring=True,normalize_crlf=True,actor_updates=0,
        adapter_sha256=None if arm=='B' else sha(OUT/arm/'adapter/adapter_model.safetensors')))
    event('scoring_complete',arm=arm,rows=len(data))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['B','C','D']);main(p.parse_args().arm)
