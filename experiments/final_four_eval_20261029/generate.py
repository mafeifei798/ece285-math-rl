"""Fixed greedy evaluation only. Immutable per-batch commits support explicit resume."""
from common_eval import *
import sys,fcntl,subprocess,threading,traceback

def main():
    guard();release_check();unchanged()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    lock=(RUN/'generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'generation_complete.json').exists(),'Already completed; never regenerate'
    contract=read(RUN/'contract.json');assert contract['protocol']==PROTOCOL
    contract_sha=sha(RUN/'contract.json');started=time.time()
    free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=20000,('Insufficient shared GPU memory; do not kill other jobs',free)
    context={'arm':'startup','dataset':None};stop=threading.Event()
    def sampler():
        import psutil
        while not stop.is_set():
            try:
                owned={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)}
                gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
                memory=[]
                for line in compute.splitlines():
                    fields=[s.strip() for s in line.split(',')]
                    if int(fields[0]) in owned:memory.append({'pid':int(fields[0]),'MiB':int(fields[1])})
                append(RUN/'resource_samples.jsonl',{'unix':time.time(),**context,'whole_GPU1':gpu,'own_processes':memory,'own_total_MiB':sum(x['MiB'] for x in memory)})
            except Exception as exc:append(RUN/'resource_samples.jsonl',{'unix':time.time(),'monitor_error':str(exc)})
            stop.wait(15)
    monitor=threading.Thread(target=sampler,daemon=True);monitor.start()
    try:
        from vllm import LLM,SamplingParams
        from vllm.lora.request import LoRARequest
        sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'))
        from repairs import repaired_outcome
        units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
        event('loading_model',generations_total=7276)
        llm=LLM(model=contract['model_path'],dtype='bfloat16',gpu_memory_utilization=.15,
           max_model_len=4096,max_num_seqs=64,max_num_batched_tokens=4096,seed=285,
           enable_lora=True,max_lora_rank=16,max_loras=1,max_cpu_loras=3,enforce_eager=True)
        params=SamplingParams(temperature=0.,max_tokens=2048,top_p=1.,top_k=-1,
           skip_special_tokens=True,include_stop_str_in_output=False)
        requests={a:None if a=='initial' else LoRARequest(a,i,str(contract['states'][a]['adapter'])) for i,a in enumerate(ARMS)}
        startup_seconds=time.time()-started;counts={a:{n:0 for n in DATASETS} for a in ARMS}
        generated_new=0
        for dataset in DATASETS:
            values=rows(ROOT/'data/prepared'/f'{dataset}.jsonl')
            for block,offset in enumerate(range(0,len(values),64)):
                expected=values[offset:offset+64];order=ARMS[block%4:]+ARMS[:block%4]
                for arm in order:
                    p=batch_path(arm,dataset,offset)
                    if p.exists():
                        checked_batch(p,expected,arm,dataset,contract_sha);counts[arm][dataset]+=len(expected);continue
                    context.update(arm=arm,dataset=dataset);event('generating',arm=arm,dataset=dataset,counts=counts,generations_total=7276)
                    t=time.time();result=llm.generate([q['prompt'] for q in expected],params,lora_request=requests[arm],use_tqdm=False)
                    inference_seconds=time.time()-t;assert len(result)==len(expected);records=[];t=time.time()
                    for q,g in zip(expected,result):
                        answer=g.outputs[0]
                        judged=repaired_outcome(answer.text,q['answer'],answer.finish_reason=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
                        records.append({'arm':arm,'dataset':dataset,'problem_id':q['problem_id'],
                          'question_sha256':q['question_sha256'],'gold':q['answer'],'completion':answer.text,
                          'completion_token_ids':list(answer.token_ids),'generated_tokens':len(answer.token_ids),
                          'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,
                          'adapter_sha256':contract['states'][arm]['adapter_sha256'],**judged})
                    grading_seconds=time.time()-t
                    assert not p.exists()
                    atomic(p,{'arm':arm,'dataset':dataset,'offset':offset,'contract_sha256':contract_sha,
                        'adapter_id':0 if arm=='initial' else ARMS.index(arm),'records':records,
                        'inference_seconds':inference_seconds,'grading_seconds':grading_seconds,'completed_unix':time.time()})
                    checked_batch(p,expected,arm,dataset,contract_sha)
                    counts[arm][dataset]+=len(records);generated_new+=len(records)
                    total=sum(sum(d.values()) for d in counts.values())
                    progress={'done':total,'total':7276,'counts':counts,'arm':arm,'dataset':dataset,
                        'elapsed_seconds':time.time()-started,'unix':time.time(),'pid':os.getpid(),
                        'last_batch_inference_seconds':inference_seconds}
                    atomic(RUN/'progress.json',progress);print(json.dumps(progress),flush=True)
        summary={}
        for arm in ARMS:
            summary[arm]={}
            for dataset in DATASETS:
                values=rows(ROOT/'data/prepared'/f'{dataset}.jsonl');records=[];batches=[]
                for offset in range(0,len(values),64):
                    b=checked_batch(batch_path(arm,dataset,offset),values[offset:offset+64],arm,dataset,contract_sha)
                    batches.append(b);records.extend(b['records'])
                assert len(records)==len(values)
                p=RUN/arm/f'{dataset}.jsonl';assert not p.exists()
                tmp=p.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records));tmp.replace(p)
                summary[arm][dataset]={'n':len(records),'correct':sum(x['reward'] for x in records),
                   'accuracy':sum(x['reward'] for x in records)/len(records),'generated_tokens':sum(x['generated_tokens'] for x in records),
                   'inference_seconds':sum(x['inference_seconds'] for x in batches),'grading_seconds':sum(x['grading_seconds'] for x in batches),
                   'predictions_sha256':sha(p)}
        atomic(RUN/'generation_complete.json',{'passed':True,'summary':summary,'startup_seconds':startup_seconds,
             'elapsed_seconds':time.time()-started,'generated_new_this_invocation':generated_new,
             'generations_total':7276,'optimizer_steps':0,'PRM_calls':0,'contract_sha256':contract_sha,
             'finished_unix':time.time()})
        event('generation_complete_waiting_CPU_audit',summary=summary)
    except Exception as exc:
        atomic(RUN/'generation_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'unix':time.time(),'generated_outputs_preserved':True})
        raise
    finally:stop.set();monitor.join(timeout=12)
if __name__=='__main__':main()
