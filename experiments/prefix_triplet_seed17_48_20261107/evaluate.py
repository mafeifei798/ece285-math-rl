"""Four adapters, existing validation only, one fixed greedy evaluation."""
from common import *
import fcntl,sys

def main():
    guard();verify();source_unchanged();out=RUN/'evaluation';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'summary.json').exists(),'Completed evaluation must not be regenerated'
    names=('source',)+ARMS;adapters={'source':CP}
    for arm in ARMS:
        result=read(RUN/arm/'result.json');assert result['passed'] and result['additional_updates']==48
        adapters[arm]=Path(result['checkpoint']);assert sha(adapters[arm]/'adapter_model.safetensors')==result['adapter_sha256']
    data_path=ROOT/'data/prepared/validation.jsonl';values=rows(data_path);assert len(values)==500
    contract={'source_release_sha256':verify(),'dataset':'validation','dataset_sha256':sha(data_path),'seed':285,
        'temperature':0.,'top_p':1.,'top_k':-1,'max_tokens':2048,'max_model_len':4096,'batch_size':64,
        'max_num_seqs':64,'max_num_batched_tokens':4096,'gpu_memory_utilization':.15,'dtype':'bfloat16',
        'enable_lora':True,'max_lora_rank':16,'max_loras':1,'max_cpu_loras':4,'enforce_eager':True,
        'engine':'V1','prompt':'unchanged prepared prompt','verifier':'reward-repair-v3-20261021',
        'repair_sha256':sha(ROOT/'experiments/reward_repair_v3_20261021/repairs.py'),
        'unit_metadata_sha256':sha(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json'),
        'adapters':{a:{'path':str(p),'sha256':sha(p/'adapter_model.safetensors'),'id':i+1} for i,(a,p) in enumerate(adapters.items())}}
    if (out/'contract.json').exists():assert read(out/'contract.json')==contract
    else:atomic(out/'contract.json',contract)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    llm=LLM(model=read(ROOT/'receipts/assets.json')['model_path'],dtype='bfloat16',gpu_memory_utilization=.15,max_model_len=4096,
        max_num_seqs=64,max_num_batched_tokens=4096,seed=285,enable_lora=True,max_lora_rank=16,max_loras=1,max_cpu_loras=4,enforce_eager=True)
    assert type(llm.llm_engine).__module__.startswith('vllm.v1.')
    params=SamplingParams(temperature=0.,top_p=1.,top_k=-1,max_tokens=2048,skip_special_tokens=True,include_stop_str_in_output=False)
    requests={a:LoRARequest(a,i+1,str(p)) for i,(a,p) in enumerate(adapters.items())}
    sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'));from repairs import repaired_outcome
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json');started=time.time();done=0
    for block,start in enumerate(range(0,len(values),64)):
        batch=values[start:start+64];order=names[block%4:]+names[:block%4]
        for arm in order:
            p=out/arm/'batches'/f'{start:06d}.json'
            if p.exists():
                saved=read(p);assert saved['contract_sha256']==sha(out/'contract.json')
                assert [r['problem_id'] for r in saved['records']]==[q['problem_id'] for q in batch]
                done+=len(batch);continue
            before=time.monotonic();generated=llm.generate([q['prompt'] for q in batch],params,lora_request=requests[arm],use_tqdm=False)
            seconds=time.monotonic()-before;records=[]
            for q,g in zip(batch,generated):
                v=g.outputs[0];judged=repaired_outcome(v.text,q['answer'],v.finish_reason=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
                records.append({'arm':arm,'problem_id':q['problem_id'],'question_sha256':q['question_sha256'],'gold':q['answer'],
                    'completion':v.text,'completion_token_ids':list(v.token_ids),'generated_tokens':len(v.token_ids),
                    'finish_reason':v.finish_reason,**judged})
            assert len(records)==len(batch)
            atomic(p,{'contract_sha256':sha(out/'contract.json'),'records':records,'inference_seconds':seconds});done+=len(records)
            atomic(out/'progress.json',{'done':done,'total':2000,'arm':arm,'seconds':time.time()-started,'unix':time.time(),'pid':os.getpid()})
            print(json.dumps({'event':'validation','done':done,'total':2000,'arm':arm}),flush=True)
    summaries={}
    for arm in names:
        batches=[read(out/arm/'batches'/f'{s:06d}.json') for s in range(0,len(values),64)]
        rr=[r for b in batches for r in b['records']];assert len(rr)==500 and len({r['problem_id'] for r in rr})==500
        p=out/arm/'predictions.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
        summaries[arm]={'n':500,'correct':sum(r['reward'] for r in rr),'accuracy':sum(r['reward'] for r in rr)/500,
            'tokens':sum(r['generated_tokens'] for r in rr),'truncated':sum(r['finish_reason']!='stop' for r in rr),
            'empty':sum(not r['completion'].strip() for r in rr),'short_wrong':sum(r['generated_tokens']<=32 and not r['reward'] for r in rr),
            'inference_seconds':sum(b['inference_seconds'] for b in batches),'predictions_sha256':sha(p)}
    verify();source_unchanged();assert sha(data_path)==contract['dataset_sha256']
    atomic(out/'summary.json',{'passed':True,'summary':summaries,'seconds':time.time()-started,'contract_sha256':sha(out/'contract.json'),
        'test_sets_evaluated':False,'PRM_at_evaluation':False})
if __name__=='__main__':main()
