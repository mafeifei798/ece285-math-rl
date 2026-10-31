"""Uniform held-out validation only; fixed prompts/verifier, greedy, no PRM."""
import argparse,fcntl,time,sys
from pilot_common import *
sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'))
from repairs import repaired_outcome

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['initial',*ARMS],required=True);a=p.parse_args()
    guard();release=verify();out=PAIR/'validation'/a.arm;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'summary.json').exists(),'Evaluation complete; do not regenerate'
    adapter=None if a.arm=='initial' else Path(read(PAIR/a.arm/'final_result.json')['adapter'])
    contract=dict(source_release_sha256=release,validation_sha256=sha(ROOT/'data/prepared/validation.jsonl'),arm=a.arm,
        adapter_sha256=sha(adapter/'adapter_model.safetensors') if adapter else None,
        seed=285,temperature=0.,cap=2048,prompt='unchanged data/prepared/validation.jsonl prompt',verifier='reward_repair_v3_20261021',
        repair_sha256=sha(ROOT/'experiments/reward_repair_v3_20261021/repairs.py'),vllm_cache_fraction=.15,batch_size=32)
    if (out/'contract.json').exists():assert read(out/'contract.json')==contract
    else:atomic(out/'contract.json',contract)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    assets=read(ROOT/'receipts/assets.json');started=time.time()
    llm=LLM(model=assets['model_path'],dtype='bfloat16',gpu_memory_utilization=.15,max_model_len=4096,
        max_num_seqs=32,max_num_batched_tokens=4096,seed=285,enable_lora=True,max_lora_rank=16,enforce_eager=True)
    request=LoRARequest(a.arm,1,str(adapter)) if adapter else None
    params=SamplingParams(temperature=0.,max_tokens=2048,top_p=1.,top_k=-1)
    questions=rows(ROOT/'data/prepared/validation.jsonl');assert len(questions)==500
    existing=rows(out/'predictions.jsonl') if (out/'predictions.jsonl').exists() else []
    done={r['problem_id']:r for r in existing};assert len(done)==len(existing)
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
    todo=[q for q in questions if q['problem_id'] not in done]
    for start in range(0,len(todo),32):
        batch=todo[start:start+32]
        generated=llm.generate([q['prompt'] for q in batch],params,lora_request=request,use_tqdm=False)
        for q,g in zip(batch,generated):
            c=g.outputs[0];judged=repaired_outcome(c.text,q['answer'],c.finish_reason=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
            r=dict(problem_id=q['problem_id'],question_sha256=q['question_sha256'],gold=q['answer'],completion=c.text,
                generated_tokens=len(c.token_ids),finish_reason=c.finish_reason,**judged)
            append(out/'predictions.jsonl',r);done[q['problem_id']]=r
        atomic(out/'progress.json',dict(done=len(done),total=500,seconds=time.time()-started,pid=os.getpid(),unix=time.time()))
        print(json.dumps(dict(event='validation',arm=a.arm,done=len(done),total=500)),flush=True)
    assert len(done)==500 and set(done)=={q['problem_id'] for q in questions}
    values=list(done.values());correct=sum(r['reward'] for r in values)
    summary=dict(passed=True,arm=a.arm,count=500,correct=correct,accuracy=correct/500,
        truncated=sum(r['finish_reason']!='stop' for r in values),empty=sum(not r['completion'].strip() for r in values),
        short_wrong=sum(r['generated_tokens']<=32 and not r['reward'] for r in values),
        no_extractable_answer=sum(r.get('answer') is None for r in values),
        mean_generated_tokens=sum(r['generated_tokens'] for r in values)/500,
        status_counts={s:sum(r['status']==s for r in values) for s in sorted({r['status'] for r in values})},
        predictions_sha256=sha(out/'predictions.jsonl'),contract_sha256=sha(out/'contract.json'),seconds=time.time()-started)
    atomic(out/'summary.json',summary)

if __name__=='__main__':main()
