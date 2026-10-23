"""Four fixed inference cells on frozen validation data, resumable and locked."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/AB_prompt_validation_20261022'
sys.path.insert(0,str(ROOT/'training'))
from common import atomic_json,file_hash,read_jsonl
sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'))
from repairs import repaired_outcome

def verify_protection():
    manifest=json.loads((OUT/'protection_manifest.json').read_text())
    for rel,item in manifest.items():
        p=ROOT/rel
        assert p.is_file() and file_hash(p)==item['sha256'], 'Protected input changed: '+rel
    return {'passed':True,'files':len(manifest)}

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    lock=(OUT/'evaluation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'evaluation_summary.json').exists(), 'Already complete; do not re-run'
    release=json.loads((OUT/'release.json').read_text())
    for rel,h in release['input_sha256'].items():assert file_hash(ROOT/rel)==h,rel
    protection_before=verify_protection()
    protocol=json.loads((OUT/'protocol.json').read_text());cfg=protocol['generation']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    queries=read_jsonl(OUT/'questions.jsonl')
    units=json.loads((OUT/'unit_metadata.json').read_text())
    query_by_id={r['problem_id']:r for r in queries}
    rows=read_jsonl(OUT/'predictions.jsonl') if (OUT/'predictions.jsonl').exists() else []
    prior={}
    for row in rows:
        key=(row['cell'],row['problem_id']);assert key not in prior
        q=query_by_id[row['problem_id']]
        assert row['cell'] in protocol['cells'] and row['question_sha256']==q['question_sha256']
        assert row['release_sha256']==file_hash(OUT/'release.json')
        prior[key]=row
    started=time.time()
    state={'status':'preflight','pid':os.getpid(),'started_unix':started,'heartbeat_unix':started,
        'release_sha256':file_hash(OUT/'release.json'),'training':False,'completed_predictions':len(rows),
        'target_predictions':protocol['max_completions'],'gpu':1,'protection_before':protection_before}
    atomic_json(OUT/'state.json',state)
    try:
        gpu=subprocess.run(['nvidia-smi','-i','1','--query-gpu=memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
        free,total,util=map(int,gpu.split(','));assert free>=40000,'Not enough free memory; wait without disturbing other jobs'
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
        state.update(gpu_before=gpu,existing_gpu_processes=apps)
        # Preserve the known matching Python development-header configuration used by the old evaluation.
        hp=ROOT/'receipts/python_dev_headers.json'
        if hp.exists():
            hs=json.loads(hp.read_text());include=hs['include_paths'];old=os.environ.get('C_INCLUDE_PATH','')
            os.environ['C_INCLUDE_PATH']=os.pathsep.join(include+([old] if old else []))
        from vllm import LLM,SamplingParams
        from vllm.lora.request import LoRARequest
        assets=json.loads((ROOT/'receipts/assets.json').read_text())
        state.update(status='loading_model',heartbeat_unix=time.time());atomic_json(OUT/'state.json',state)
        llm=LLM(model=assets['model_path'],dtype=cfg['dtype'],gpu_memory_utilization=cfg['gpu_memory_utilization'],
            max_model_len=cfg['max_model_len'],max_num_seqs=cfg['max_num_seqs'],max_num_batched_tokens=cfg['max_num_batched_tokens'],
            seed=cfg['seed'],enable_lora=True,max_lora_rank=16,max_loras=cfg['max_loras'],enforce_eager=cfg['enforce_eager'])
        params=SamplingParams(temperature=cfg['temperature'],top_p=cfg['top_p'],top_k=cfg['top_k'],max_tokens=cfg['max_tokens'])
        requests={arm:LoRARequest('corrected_'+arm,i+1,str(ROOT/f'runs/{arm}_corrected_seed17_6M_20261022/final_adapter')) for i,arm in enumerate('AB')}
        cells=protocol['cells'];chunk=protocol['question_chunk_size']
        with (OUT/'predictions.jsonl').open('a') as stream:
            for offset in range(0,len(queries),chunk):
                batch=queries[offset:offset+chunk]
                rotation=(offset//chunk)%len(cells)
                order=cells[rotation:]+cells[:rotation]
                for cell in order:
                    arm,prompt=cell.split('_',1)
                    todo=[q for q in batch if (cell,q['problem_id']) not in prior]
                    if not todo:continue
                    state.update(status='generating',cell=cell,question_offset=offset,heartbeat_unix=time.time(),completed_predictions=len(prior))
                    atomic_json(OUT/'state.json',state)
                    begun=time.time()
                    generated=llm.generate([q['prompt_'+prompt] for q in todo],params,lora_request=requests[arm],use_tqdm=False)
                    assert len(generated)==len(todo)
                    for q,result in zip(todo,generated):
                        answer=result.outputs[0]
                        judged=repaired_outcome(answer.text,q['answer'],answer.finish_reason=='stop',
                            question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
                        assert judged['status']!='verifier_error'
                        record={'cell':cell,'arm':arm,'prompt_variant':prompt,'dataset':q['dataset'],
                            'problem_id':q['problem_id'],'question_sha256':q['question_sha256'],'gold':q['answer'],
                            'completion':answer.text,'generated_tokens':len(answer.token_ids),'finish_reason':answer.finish_reason,
                            'release_sha256':state['release_sha256'],**judged}
                        stream.write(json.dumps(record,ensure_ascii=False)+'\n');prior[(cell,q['problem_id'])]=record
                    stream.flush();os.fsync(stream.fileno())
                    state.update(completed_predictions=len(prior),heartbeat_unix=time.time(),last_batch_seconds=time.time()-begun,
                        generated_completion_tokens=sum(r['generated_tokens'] for r in prior.values()),elapsed_seconds=time.time()-started)
                    atomic_json(OUT/'state.json',state)
                    print(json.dumps({k:state[k] for k in ['cell','completed_predictions','generated_completion_tokens','last_batch_seconds','elapsed_seconds']}),flush=True)
        assert len(prior)==protocol['max_completions']
        assert sum(r['generated_tokens'] for r in prior.values())<=protocol['max_generated_completion_tokens']
        protection_after=verify_protection()
        for rel,h in release['input_sha256'].items():assert file_hash(ROOT/rel)==h,rel
        summary={'status':'completed','training':False,'finished_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds':time.time()-started,'release_sha256':state['release_sha256'],
            'predictions_sha256':file_hash(OUT/'predictions.jsonl'),'count':len(prior),
            'generated_completion_tokens':sum(r['generated_tokens'] for r in prior.values()),'protection_after':protection_after,'cells':{}}
        for cell in cells:
            summary['cells'][cell]={}
            for dataset in protocol['datasets']:
                part=[r for r in prior.values() if r['cell']==cell and r['dataset']==dataset]
                assert len(part)==protocol['datasets'][dataset]
                summary['cells'][cell][dataset]={'n':len(part),'correct':sum(r['reward'] for r in part),
                    'accuracy':sum(r['reward'] for r in part)/len(part),'truncated':sum(r['finish_reason']!='stop' for r in part),
                    'generated_tokens':sum(r['generated_tokens'] for r in part)}
        atomic_json(OUT/'evaluation_summary.json',summary)
        state.update(status='completed',finished_unix=time.time(),heartbeat_unix=time.time(),protection_after=protection_after)
        atomic_json(OUT/'state.json',state)
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc(),heartbeat_unix=time.time())
        atomic_json(OUT/'state.json',state)
        raise

if __name__=='__main__':main()
