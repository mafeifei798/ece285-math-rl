from common import *
import argparse,fcntl
from collections import Counter
from answer_format import describe,boxes

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['baseline','nine','final','endpoints_math','endpoints_gsm'],required=True)
    ap.add_argument('--repeat',type=int,choices=[0,1],required=True);a=ap.parse_args()
    guard();verify();source_unchanged();out=RUN/'evaluation'/a.cohort/f'repeat_{a.repeat}';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'summary.json').exists(),'Completed evaluation must not be regenerated'
    endpoint=a.cohort.startswith('endpoints_');dataset='math500' if a.cohort=='endpoints_math' else 'gsm8k' if a.cohort=='endpoints_gsm' else 'validation'
    step={'baseline':288,'nine':432,'final':576}.get(a.cohort)
    if endpoint:
        checkpoints={f'{arm}_{k}':SOURCE/arm/'checkpoint-288' if k==288 else RUN/arm/'milestones/checkpoint-576' for k in [288,576] for arm in ARMS}
    else:checkpoints={arm:SOURCE/arm/'checkpoint-288' if step==288 else RUN/arm/f'milestones/checkpoint-{step}' for arm in ARMS}
    datafile=ROOT/f'data/prepared/{dataset}.jsonl';data=rows(datafile)
    e=read(HERE/'protocol.json')['evaluation'];names=tuple(checkpoints)
    contract={'cohort':a.cohort,'repeat':a.repeat,'dataset':dataset,'dataset_sha256':sha(datafile),
        'settings':e,'verifier_sha256':sha(ROOT/'experiments/reward_repair_v3_20261021/repairs.py'),
        'unit_metadata_sha256':sha(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json'),
        'release_sha256':verify(),'models':{n:{'path':str(p),'sha256':sha(p/'adapter_model.safetensors'),'lora_id':i+1} for i,(n,p) in enumerate(checkpoints.items())}}
    if (out/'contract.json').exists():assert read(out/'contract.json')==contract
    else:atomic(out/'contract.json',contract)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    llm=LLM(model=read(ROOT/'receipts/assets.json')['model_path'],dtype='bfloat16',gpu_memory_utilization=.15,
        max_model_len=4096,max_num_seqs=64,max_num_batched_tokens=4096,seed=285,enable_lora=True,
        max_lora_rank=16,max_loras=1,max_cpu_loras=4,enforce_eager=True)
    assert type(llm.llm_engine).__module__.startswith('vllm.v1.')
    requests={n:LoRARequest(n,i+1,str(p)) for i,(n,p) in enumerate(checkpoints.items())}
    params=SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=2048,skip_special_tokens=True,include_stop_str_in_output=False)
    sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'));from repairs import repaired_outcome
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json');begun=time.time();done=0
    for block,start in enumerate(range(0,len(data),64)):
        batch=data[start:start+64];order=names[block%len(names):]+names[:block%len(names)]
        for name in order:
            path=out/name/'batches'/f'{start:06d}.json'
            if path.exists():
                saved=read(path);assert saved['contract_sha256']==sha(out/'contract.json')
                assert [r['problem_id'] for r in saved['records']]==[q['problem_id'] for q in batch]
                done+=len(batch);continue
            t=time.monotonic();generated=llm.generate([q['prompt'] for q in batch],params,lora_request=requests[name],use_tqdm=False)
            sec=time.monotonic()-t;rr=[]
            for q,g in zip(batch,generated):
                v=g.outputs[0];graded=repaired_outcome(v.text,q['answer'],v.finish_reason=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
                row={'model':name,'problem_id':q['problem_id'],'question_sha256':q['question_sha256'],'gold':q['answer'],
                    'completion':v.text,'completion_token_ids':list(v.token_ids),'generated_tokens':len(v.token_ids),'finish_reason':v.finish_reason,**graded}
                fmt=describe(row);bb=boxes(v.text)
                fmt['last_box_character_fraction']=bb[-1]['start']/max(1,len(v.text)) if bb else None
                fmt['no_nonplaceholder_complete_box']=not any(b['end'] is not None and not b['placeholder'] for b in bb)
                row['format_features']=fmt;rr.append(row)
            atomic(path,{'contract_sha256':sha(out/'contract.json'),'records':rr,'inference_seconds':sec})
            done+=len(rr);atomic(out/'progress.json',{'done':done,'total':len(data)*len(names),'model':name,'seconds':time.time()-begun,'pid':os.getpid(),'unix':time.time()})
    summary={}
    for name in names:
        rr=[r for start in range(0,len(data),64) for r in read(out/name/'batches'/f'{start:06d}.json')['records']]
        assert len(rr)==len(data) and len({r['problem_id'] for r in rr})==len(data)
        path=out/name/'predictions.jsonl';assert not path.exists()
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
        flags={k:sum(r['format_features'][k] for r in rr) for k,v in rr[0]['format_features'].items() if isinstance(v,bool)}
        summary[name]={'n':len(rr),'correct':sum(r['reward'] for r in rr),'accuracy':sum(r['reward'] for r in rr)/len(rr),
            'tokens':sum(r['generated_tokens'] for r in rr),'status_counts':dict(Counter(r['status'] for r in rr)),
            'format_counts':flags,'truncated':sum(r['finish_reason']!='stop' for r in rr),'predictions_sha256':sha(path)}
    verify();source_unchanged()
    atomic(out/'summary.json',{'passed':True,'summary':summary,'seconds':time.time()-begun,'contract_sha256':sha(out/'contract.json'),
        'no_PRM_at_evaluation':True,'no_optimizer_updates':True})

if __name__=='__main__':main()
