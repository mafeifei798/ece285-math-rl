"""Admit exactly one historical checkpoint under the completed evaluation protocol."""
from common_eval import *
import importlib.metadata

def main():
    guard();release=release_check()
    assert not (RUN/'contract.json').exists(),'Existing admission; inspect, never overwrite'
    assert read(PREVIOUS/'controller.json')['stage']=='complete'
    assert read(PREVIOUS/'acceptance.json')['passed']
    prior=read(PREVIOUS/'contract.json');old=read(HISTORICAL/'evaluation/evaluation_summary.json')
    old_contract=read(HISTORICAL/'evaluation/evaluation_config.json')
    assert prior['model_revision']==old['model_revision']
    assert sha(CHECKPOINT/'adapter_model.safetensors')==old_contract['adapter_sha256']
    assert Path(old['adapter']).resolve()==CHECKPOINT.resolve()
    assert read(CHECKPOINT/'trainer_state.json')['global_step']==284
    accounting=read(CHECKPOINT/'completion_accounting.json')
    assert 6000000<=accounting['generated_completion_tokens']<6100000
    old_cfg=read(HISTORICAL/'run_config.json')
    current_cfg=read(ROOT/'runs/C_bounded_triplet_seed17_1M_20261029_v2/A/run_config.json')
    assert old_cfg['actor_adapter_initial_sha256']==current_cfg['initial_adapter_digest']
    assert old_cfg['data_manifest_sha256']==sha(ROOT/'data/prepared/manifest.json')
    assert old_contract['repair_source_sha256']==prior['verifier_sha256']
    assert old_contract['unit_metadata_sha256']==prior['units_sha256']
    assert sha(ROOT/'experiments/reward_repair_v3_20261021/repairs.py')==prior['verifier_sha256']
    assert sha(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')==prior['units_sha256']
    assert set(PROTOCOL)==set(prior['protocol'])
    for k,v in PROTOCOL.items():
        if k!='ordering':assert v==prior['protocol'][k],('Protocol changed',k)
    versions={p:importlib.metadata.version(p) for p in prior['versions']}
    assert versions==prior['versions'],('Environment drift',versions)
    protected={}
    def keep(p):
        p=Path(p);protected[str(p.relative_to(ROOT))]=sha(p)
    for n,h in read(ROOT/'audits/Final_four_eval_seed17_1M_20261029/protected_before.json').items():
        assert sha(ROOT/n)==h,('Old protected source changed',n)
        protected[n]=h
    for parent in [PREVIOUS,HISTORICAL,ROOT/'experiments/final_four_eval_20261029',ROOT/'experiments/ab_corrected_6m_seed17_20261022']:
        for p in parent.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.jsonl','.py','.safetensors']:keep(p)
    for p in CHECKPOINT.iterdir():
        if p.is_file():keep(p)
    assets=read(ROOT/'receipts/assets.json');model=Path(prior['model_path'])
    for n,d in assets['files'].items():assert sha(model/n)==d['sha256'];keep(model/n)
    data={}
    for dataset,n in [('math500',500),('gsm8k',1319)]:
        p=ROOT/'data/prepared'/f'{dataset}.jsonl';assert sha(p)==prior['datasets'][dataset]['sha256'];keep(p)
        values=rows(p);assert len(values)==n==len({q['problem_id'] for q in values})
        for label,path in [('historical',HISTORICAL/'evaluation'/f'{dataset}.jsonl'),('current_A',PREVIOUS/'A'/f'{dataset}.jsonl'),('current_initial',PREVIOUS/'initial'/f'{dataset}.jsonl')]:
            rr=rows(path);assert len(rr)==n and [r['problem_id'] for r in rr]==[q['problem_id'] for q in values]
            for r,q in zip(rr,values):assert r['question_sha256']==q['question_sha256'] and r['gold']==q['answer']
            keep(path)
        data[dataset]=prior['datasets'][dataset]
    contract={
        'release_sha256':release,'protocol':PROTOCOL,'model_id':prior['model_id'],
        'model_revision':prior['model_revision'],'model_path':prior['model_path'],
        'states':{'A_6M':{'adapter':str(CHECKPOINT),'adapter_sha256':sha(CHECKPOINT/'adapter_model.safetensors'),
            'adapter_config_sha256':sha(CHECKPOINT/'adapter_config.json'),'adapter_id':1,
            'training_updates':284,'training_generated_tokens':accounting['generated_completion_tokens']}},
        'datasets':data,'verifier_sha256':prior['verifier_sha256'],'units_sha256':prior['units_sha256'],'versions':versions,
        'reference_contract_sha256':sha(PREVIOUS/'contract.json'),'reference_comparison_sha256':sha(PREVIOUS/'comparison.json'),
        'matched_protocol_keys':[k for k in PROTOCOL if k!='ordering'],
        'scheduling_difference':'Only historical A_6M is generated, so four-model rotation is not applicable. All inference/LoRA/batch/verifier settings are retained.',
        'historical_summary_sha256':sha(HISTORICAL/'evaluation/evaluation_summary.json'),
        'historical_training_vllm_memory_utilization':old_cfg['shared_training_arguments']['vllm_gpu_memory_utilization'],
        'current_training_vllm_memory_utilization':current_cfg['settings']['vllm_gpu_memory_utilization'],
        'known_overlap':prior['known_overlap'],'new_training':False,'new_PRM_inference':False}
    atomic(AUDIT/'protected_before.json',protected);atomic(RUN/'contract.json',contract)
    atomic(AUDIT/'preflight.json',{'passed':True,'protected_files':len(protected),'questions':1819,'generations':1819,
        'historical_checkpoint_sha256_verified':True,'same_base_and_initial_LoRA_verified':True,
        'all_reference_questions_gold_and_verifier_identical':True,'matched_inference_protocol_except_scheduling':True,
        'new_training':False,'CUDA_initialized':False,'source_release_sha256':release})
    event('admitted_CPU_only',generations=1819)
    print(json.dumps({'passed':True,'protected_files':len(protected),'states':contract['states']}))
if __name__=='__main__':main()
