"""CPU-only freeze: checkpoints, original results, data, verifier and configuration."""
from common_eval import *
import sys,importlib.metadata,subprocess

def main():
    guard();release=release_check()
    assert not (RUN/'contract.json').exists(),'Already admitted; inspect existing lock/receipts before any continuation'
    assert read(PILOT/'controller.json')['stage']=='complete'
    source=ROOT/'experiments/c_bounded_pilot_20261029_v2'
    previous=read(source/'release.json')
    for n,h in previous['files'].items():assert sha(source/n)==h,n
    for n,h in previous['dependencies'].items():assert sha(ROOT/n)==h,n
    assets=read(ROOT/'receipts/assets.json');model=Path(assets['model_path'])
    for n,d in assets['files'].items():assert sha(model/n)==d['sha256'],n
    manifest=read(ROOT/'data/prepared/manifest.json')
    assert sha(ROOT/'training/baseline_config.json')==manifest['config_sha256']
    protected={}
    def keep(p):
        p=Path(p);protected[str(p.relative_to(ROOT))]=sha(p)
    old=read(ROOT/'runs/CD_PRM7B_pair_seed17_20261028/protected_originals.json')
    for n,h in old.items():assert sha(ROOT/n)==h;protected[n]=h
    for parent in [source,ROOT/'runs/CD_PRM7B_pair_seed17_20261028']:
        for p in parent.rglob('*'):
            if p.is_file() and (p.suffix in ['.py','.json','.jsonl','.safetensors']):keep(p)
    for p in PILOT.rglob('*'):
        if p.is_file() and p.suffix in ['.py','.json','.jsonl','.safetensors']:keep(p)
    for p in model.iterdir():
        if p.is_file():keep(p)
    for n in ['math500','gsm8k','train','validation']:
        p=ROOT/'data/prepared'/f'{n}.jsonl';assert sha(p)==manifest['files'][p.name]['sha256'];keep(p)
    for parent in ['experiments/reward_repair_v3_20261021','experiments/reward_repair_v2_20261021','training']:
        for p in (ROOT/parent).glob('*.py'):keep(p)
    units=ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json';keep(units)
    for p in [ROOT/'data/prepared/manifest.json',ROOT/'receipts/assets.json',ROOT/'training/baseline_config.json']:keep(p)
    states={'initial':{'adapter':None,'adapter_sha256':None,'training_generated_tokens':0}}
    for arm in ARMS[1:]:
        result=read(PILOT/arm/'final_result.json');assert result['passed'] and result['global_step']==48
        adapter=Path(result['adapter']);assert sha(adapter/'adapter_model.safetensors')==result['adapter_sha256']
        checkpoint=Path(result['checkpoint']);receipt=read(checkpoint/'pilot_checkpoint.json')
        for n,h in receipt['files_sha256'].items():assert sha(checkpoint/n)==h;keep(checkpoint/n)
        assert sha(checkpoint/'adapter_model.safetensors')==result['adapter_sha256']
        assert read(PILOT/arm/'resume_acceptance.json')['passed']
        states[arm]={'adapter':str(adapter),'adapter_sha256':result['adapter_sha256'],
            'adapter_config_sha256':sha(adapter/'adapter_config.json'),
            'training_generated_tokens':result['accounting']['generated_completion_tokens'],
            'training_updates':result['global_step'],'source_release_sha256':result['source_release_sha256']}
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    data={};all_ids=set()
    for name,n in [('math500',500),('gsm8k',1319)]:
        values=rows(ROOT/'data/prepared'/f'{name}.jsonl')
        assert len(values)==n and len({x['problem_id'] for x in values})==n
        assert not (all_ids & {x['problem_id'] for x in values});all_ids.update(x['problem_id'] for x in values)
        lengths=[len(tok.encode(x['prompt'])) for x in values];assert max(lengths)<=2048
        data[name]={'n':n,'sha256':manifest['files'][f'{name}.jsonl']['sha256'],'max_prompt_tokens':max(lengths)}
    contract={'release_sha256':release,'protocol':PROTOCOL,'model_id':assets['model_id'],
       'model_revision':assets['model_revision'],'model_path':str(model),'states':states,'datasets':data,
       'verifier_sha256':sha(ROOT/'experiments/reward_repair_v3_20261021/repairs.py'),
       'units_sha256':sha(units),'versions':{x:importlib.metadata.version(x) for x in ['torch','vllm','transformers','peft','trl','math-verify']},
       'known_overlap':manifest['prm800k_known_question_overlap'],'contamination_limit':manifest['limitation'],
       'tests_have_historical_results':True,'new_training':False,'new_PRM_inference':False}
    atomic(AUDIT/'protected_before.json',protected);atomic(RUN/'contract.json',contract)
    atomic(AUDIT/'preflight.json',{'passed':True,'protected_files':len(protected),'model_file_hashes_checked':True,
       'checkpoint_optimizer_and_adapter_hashes_checked':True,'questions':1819,'generations':7276,
       'CUDA_initialized':False,'release_sha256':release})
    event('admitted_CPU_only',generations=7276)
    print(json.dumps({'passed':True,'protected_files':len(protected),'states':states,'data':data},ensure_ascii=False))
if __name__=='__main__':main()
