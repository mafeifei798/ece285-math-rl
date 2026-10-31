"""Frozen initial actor: exactly four samples per selected problem, no optimizer."""
from common import *


def main():
    assert_root();verify_release()
    if (OUT/'buffer_manifest.json').exists():
        d=read(OUT/'buffer_manifest.json');assert sha(OUT/'buffer_rollouts.jsonl')==d['sha256'];return
    from vllm import LLM,SamplingParams
    questions=rows(OUT/'buffer_questions.jsonl');assert len(questions)==512
    assets=read(ROOT/'receipts/assets.json');assert assets['model_revision']==CFG['actor_revision']
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
    model=LLM(model=assets['model_path'],dtype='bfloat16',seed=CFG['generation_seed'],
        gpu_memory_utilization=.20,max_model_len=4096,max_num_seqs=32,max_num_batched_tokens=4096,enforce_eager=True)
    started=time.time();all_rows=[]
    for start in range(0,len(questions),CFG['generation_question_batch']):
        chunk=OUT/'buffer_chunks'/f'{start:06d}.jsonl';batch=questions[start:start+CFG['generation_question_batch']]
        if chunk.exists():
            saved=rows(chunk);assert len(saved)==len(batch)*4
            assert [r['problem_id'] for r in saved]==[q['problem_id'] for q in batch for _ in range(4)]
        else:
            params=SamplingParams(n=4,temperature=CFG['temperature'],top_p=1.,top_k=-1,max_tokens=2048,seed=CFG['generation_seed']+start,skip_special_tokens=True)
            answers=model.generate([q['prompt'] for q in batch],params,use_tqdm=False);saved=[]
            assert len(answers)==len(batch)
            for q,a in zip(batch,answers):
                assert len(a.outputs)==4
                for j,o in enumerate(a.outputs):
                    judged=repaired_outcome(o.text,q['answer'],o.finish_reason=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
                    saved.append(dict(sample_id=f'{q["problem_id"]}/initial/{j}',problem_id=q['problem_id'],question=q['problem'],question_sha256=q['question_sha256'],gold=q['answer'],
                        completion=o.text,completion_token_ids=list(o.token_ids),completion_tokens=len(o.token_ids),finish_reason=o.finish_reason,terminated=o.finish_reason=='stop',**judged))
            write_rows(chunk,saved)
        all_rows.extend(saved)
        event('generating_buffer',questions_done=start+len(batch),questions_total=512,generated_responses=len(all_rows),elapsed_seconds=time.time()-started)
    assert len(all_rows)==2048 and len({r['sample_id'] for r in all_rows})==2048
    write_rows(OUT/'buffer_rollouts.jsonl',all_rows)
    from collections import Counter
    atomic(OUT/'buffer_manifest.json',dict(sha256=sha(OUT/'buffer_rollouts.jsonl'),rows=len(all_rows),questions=512,samples_per_question=4,
        completion_tokens=sum(r['completion_tokens'] for r in all_rows),statuses=dict(Counter(r['status'] for r in all_rows)),policy_optimizer_updates=0,
        actor_revision=CFG['actor_revision'],generation='initial frozen actor; original prepared prompt; temperature1; cap2048; independently seeded saved chunks'))


if __name__=='__main__':main()
