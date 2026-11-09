"""V1-only common inference at144 validation and288 fixed test checkpoints."""
from pilot_common import *
import argparse,fcntl,sys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=[144,288],required=True);a=parser.parse_args()
    guard();verify();unchanged();out=PAIR/'evaluation'/str(a.step);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'summary.json').exists(),'Completed evaluation must not be regenerated'
    datasets=['validation'] if a.step==144 else ['math500','gsm8k']
    adapters={arm:Path(read(PAIR/arm/f'stage_{a.step}_result.json')['checkpoint']) for arm in ARMS}
    contract={'source_release_sha256':verify(),'step':a.step,'engine':'V1','seed':285,'temperature':0.,'top_p':1.,'top_k':-1,
        'cap':2048,'max_model_len':4096,'batch_size':64,'max_num_seqs':64,'max_num_batched_tokens':4096,'gpu_memory_utilization':.15,
        'max_loras':1,'max_cpu_loras':3,'max_lora_rank':16,'dtype':'bfloat16','enforce_eager':True,'VLLM_ENABLE_V1_MULTIPROCESSING':'1',
        'prompt':'unchanged prepared prompts','verifier':'reward-repair-v3-20261021','repair_sha256':sha(ROOT/'experiments/reward_repair_v3_20261021/repairs.py'),
        'unit_metadata_sha256':sha(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json'),
        'dataset_sha256':{d:sha(ROOT/'data/prepared'/f'{d}.jsonl') for d in datasets},
        'adapters':{arm:{'path':str(p),'sha256':sha(p/'adapter_model.safetensors'),'id':i+1} for i,(arm,p) in enumerate(adapters.items())}}
    if (out/'contract.json').exists():assert read(out/'contract.json')==contract
    else:atomic(out/'contract.json',contract)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    assets=read(ROOT/'receipts/assets.json');started=time.time()
    llm=LLM(model=assets['model_path'],dtype='bfloat16',gpu_memory_utilization=.15,max_model_len=4096,max_num_seqs=64,max_num_batched_tokens=4096,
        seed=285,enable_lora=True,max_lora_rank=16,max_loras=1,max_cpu_loras=3,enforce_eager=True)
    assert type(llm.llm_engine).__module__.startswith('vllm.v1.'),type(llm.llm_engine).__module__
    requests={arm:LoRARequest(arm,i+1,str(p)) for i,(arm,p) in enumerate(adapters.items())}
    params=SamplingParams(temperature=0.,max_tokens=2048,top_p=1.,top_k=-1,skip_special_tokens=True,include_stop_str_in_output=False)
    sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'));from repairs import repaired_outcome
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json');summary={arm:{} for arm in ARMS}
    total=sum(len(rows(ROOT/'data/prepared'/f'{d}.jsonl')) for d in datasets)*3;done=0
    for dataset in datasets:
        values=rows(ROOT/'data/prepared'/f'{dataset}.jsonl')
        for block,start in enumerate(range(0,len(values),64)):
            batch=values[start:start+64];order=ARMS[block%3:]+ARMS[:block%3]
            for arm in order:
                p=out/arm/dataset/'batches'/f'{start:06d}.json'
                if p.exists():
                    d=read(p);assert d['contract_sha256']==sha(out/'contract.json') and [r['problem_id'] for r in d['records']]==[q['problem_id'] for q in batch]
                    done+=len(batch);continue
                t=time.time();gs=llm.generate([q['prompt'] for q in batch],params,lora_request=requests[arm],use_tqdm=False);seconds=time.time()-t
                rr=[]
                for q,g in zip(batch,gs):
                    x=g.outputs[0];judged=repaired_outcome(x.text,q['answer'],x.finish_reason=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
                    rr.append({'arm':arm,'problem_id':q['problem_id'],'question_sha256':q['question_sha256'],'gold':q['answer'],'completion':x.text,
                        'completion_token_ids':list(x.token_ids),'generated_tokens':len(x.token_ids),'finish_reason':x.finish_reason,
                        'adapter_sha256':contract['adapters'][arm]['sha256'],**judged})
                atomic(p,{'contract_sha256':sha(out/'contract.json'),'arm':arm,'records':rr,'inference_seconds':seconds});done+=len(rr)
                atomic(out/'progress.json',{'done':done,'total':total,'dataset':dataset,'arm':arm,'unix':time.time(),'pid':os.getpid(),'seconds':time.time()-started})
                print(json.dumps({'event':'evaluation','step':a.step,'done':done,'total':total,'arm':arm,'dataset':dataset}),flush=True)
        for arm in ARMS:
            batches=[read(out/arm/dataset/'batches'/f'{start:06d}.json') for start in range(0,len(values),64)]
            rr=[r for b in batches for r in b['records']];assert len(rr)==len(values)
            p=out/arm/f'{dataset}.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
            summary[arm][dataset]={'n':len(rr),'correct':sum(r['reward'] for r in rr),'accuracy':sum(r['reward'] for r in rr)/len(rr),
                'generated_tokens':sum(r['generated_tokens'] for r in rr),'truncated':sum(r['finish_reason']!='stop' for r in rr),
                'empty':sum(not r['completion'].strip() for r in rr),'short_wrong':sum(r['generated_tokens']<=32 and not r['reward'] for r in rr),
                'predictions_sha256':sha(p),'inference_seconds':sum(b['inference_seconds'] for b in batches)}
    unchanged();atomic(out/'summary.json',{'passed':True,'step':a.step,'summary':summary,'contract_sha256':sha(out/'contract.json'),'seconds':time.time()-started,'new_training':False,'new_PRM_calls':0})
if __name__=='__main__':main()
