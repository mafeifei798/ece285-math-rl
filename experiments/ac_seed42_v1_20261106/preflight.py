"""CPU admission for a new seed, preserving scientific code and old artifacts."""
from pilot_common import *
import ast,sys,subprocess,fcntl,importlib.metadata,dataclasses
import torch

def main():
    guard();release=verify();assert not (AUDIT/'preflight.json').exists()
    prior=ROOT/'runs/ABC_seed17_resumed_V1_20261104';prior_audit=ROOT/'audits/ABC_seed17_resumed_V1_20261104'
    assert read(prior/'controller.json')['stage']=='complete' and read(prior/'comparison.json')['passed']
    with (ROOT/'receipts/abc_seed17_resume_v1_gpu1.lock').open('r') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','trl','peft','vllm']}
    expected=['2.8.0','4.56.2','0.23.1','0.17.1','0.10.2']
    assert [v.split('+')[0] for v in versions.values()]==expected,versions
    old=ROOT/'experiments/abc_seed17_resumed_v1_20261104'
    def functions(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    original=functions(old/'runtime.py');new=functions(HERE/'runtime.py')
    for name in ['PilotMixin','OutcomeLedger','components_for_arm','new_accounting','trainer_class']:assert original[name]==new[name],name
    assert sha(old/'advantages.py')==sha(HERE/'advantages.py')
    from runtime import trainer_class,components_for_arm
    trainer_class()
    from advantages import compute_advantages
    from cpu_reference import bounded_correct_process
    checked=0
    for group in rows(ROOT/'runs/CD_PRM7B_pair_seed17_20261028/group_advantages.jsonl'):
        if group['arm']!='C':continue
        y=group.get('outcomes',group.get('outcome_rewards'));m=group['min_scores']
        got=compute_advantages(y,m)['advantages'].tolist();ref=bounded_correct_process(y,m)['advantage']
        assert max(abs(a-b) for a,b in zip(got,ref))<3e-7
        assert all(a<=0 for a,v in zip(got,y) if not v);checked+=8
    probe=components_for_arm('A',[1,0,1,0,0,1,0,0],[0.]*8,[False]*8)
    torch.testing.assert_close(probe['advantages'],probe['outcome_component'],atol=0,rtol=0)
    protected={}
    for n,h in read(prior_audit/'protected_before.json').items():
        assert sha(ROOT/n)==h,('Historical protected artifact changed',n);protected[n]=h
    for folder in [prior,old,prior_audit]:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix not in ['.lock','.pyc'] and '__pycache__' not in p.parts:protected[str(p.relative_to(ROOT))]=sha(p)
    manifest=read(ROOT/'data/prepared/manifest.json')
    for n,v in manifest['files'].items():assert sha(ROOT/'data/prepared'/n)==v['sha256']
    train=rows(ROOT/'data/prepared/train.jsonl');val=rows(ROOT/'data/prepared/validation.jsonl')
    assert len(val)==500 and not {x['question_sha256'] for x in train}&{x['question_sha256'] for x in val}
    # Actual TRL sampler uses args.seed. Verify the full 288-update schedule against it.
    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(42)).tolist()
    schedule=[[train[i]['problem_id'] for i in order[k:k+4]] for k in range(0,len(order)-3,4)]
    from trl.trainer.utils import RepeatSampler
    sampler=RepeatSampler(train,mini_repeat_count=8,batch_size=4,repeat_count=32,shuffle=True,seed=42)
    from itertools import islice
    stream=list(islice(sampler,288*32*32));assert len(schedule)>288
    assert [[train[i]['problem_id'] for i in stream[k*32*32:k*32*32+32:8]] for k in range(288)]==schedule[:288]
    assets=read(ROOT/'receipts/assets.json');model=Path(assets['model_path']);base_hashes={}
    for n,v in assets['files'].items():
        assert (model/n).stat().st_size==v['bytes'] and sha(model/n)==v['sha256'];base_hashes[n]=v['sha256']
    prm=ROOT/'models/qwen25_math_prm_7b';idx=read(prm/'model.safetensors.index.json')
    prm_hashes={n:sha(prm/n) for n in sorted(set(idx['weight_map'].values()))}
    for p in prm.iterdir():
        if p.is_file() and p.suffix in ['.json','.py','.txt']:prm_hashes[p.name]=sha(p)
    from audit_common import C_ADAPTER
    assert sha(C_ADAPTER/'adapter_model.safetensors')==read(HERE/'protocol.json')['C_adapter_sha256']
    protected.update({str((prm/n).relative_to(ROOT)):h for n,h in prm_hashes.items() if not n.endswith('.safetensors')})
    for arm in ARMS:
        run=PAIR/arm;assert not run.exists();run.mkdir();(run/'milestones').mkdir()
    assert not torch.cuda.is_initialized()
    atomic(AUDIT/'protected_before.json',protected)
    receipt={'passed':True,'source_release_sha256':release,'start_steps':{'A':0,'C_bounded':0},
        'protected_files':len(protected),'CPU_reference_rows':checked,'scientific_runtime_AST_identical':True,'advantages_bytes_identical':True,
        'first_epoch_question_groups':schedule,'protocol':read(HERE/'protocol.json'),'CUDA_initialized':False,'versions':versions,
        'dataset_hashes':{n:v['sha256'] for n,v in manifest['files'].items()},'base_model_hashes':base_hashes,'PRM_file_hashes':prm_hashes,
        'seed42_prompt_schedule_verified_against_installed_TRL':True,'validation_train_overlap':0,'PRM_adaptation_changed':False,
        'notes':['PRM anchor set retains the previously disclosed six same-input conflicts; no adaptation in this experiment.',
                 'Blind audit found no new scorer implementation mismatch; correct answers do not certify valid derivations.']}
    atomic(AUDIT/'preflight.json',receipt);event('admitted_CPU_only',seed=42,protected_files=len(protected))
    print(json.dumps({k:receipt[k] for k in ['passed','protected_files','CPU_reference_rows','CUDA_initialized','versions','seed42_prompt_schedule_verified_against_installed_TRL']}))
if __name__=='__main__':main()
