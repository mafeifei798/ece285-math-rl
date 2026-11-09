"""CPU-only admission of complete paused states; never regenerates a rollout."""
from pilot_common import *
import ast, shutil, sys, subprocess, fcntl, importlib.metadata
import torch

def main():
    guard();release=verify()
    assert not (AUDIT/'preflight.json').exists(),'Admission already exists; inspect rather than repeat'
    pause=read(PAUSED/'pause_receipt.json')
    assert pause['paused_by_user'] and not pause['automatic_resume']
    assert read(PAUSED/'controller.json')['stage']=='paused_by_user'
    for path in [PAUSED/'controller.lock',ROOT/'receipts/abc_seed17_resume_v1_gpu1.lock']:
        with path.open('r') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(f,fcntl.LOCK_UN)
    gpu=read(HERE/'protocol.json')['gpu_uuid']
    actual=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    assert gpu in actual and 'A100' in actual
    expected_versions={'torch':'2.8.0','transformers':'4.56.2','trl':'0.23.1','peft':'0.17.1','vllm':'0.10.2'}
    versions={n:importlib.metadata.version(n) for n in expected_versions}
    assert all(versions[n].split('+')[0]==v for n,v in expected_versions.items()),versions
    assert sha(HERE/'runtime.py')==sha(PAUSED_SOURCE/'runtime.py')
    assert sha(HERE/'advantages.py')==sha(PAUSED_SOURCE/'advantages.py')
    assert sha(HERE/'evaluate.py')==sha(PAUSED_SOURCE/'evaluate.py')
    def functions(path):
        return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    assert functions(HERE/'train_arm.py')['Acceptance']==functions(PAUSED_SOURCE/'train_arm.py')['Acceptance']
    from runtime import trainer_class,components_for_arm
    trainer_class()  # Verifies exact installed TRL source without constructing a model.
    from advantages import compute_advantages
    sys.path.insert(0,str(ORIGIN));from cpu_reference import bounded_correct_process
    checked=0
    for group in rows(ROOT/'runs/CD_PRM7B_pair_seed17_20261028/group_advantages.jsonl'):
        if group['arm']!='C':continue
        y=group.get('outcomes',group.get('outcome_rewards'));m=group['min_scores']
        got=compute_advantages(y,m)['advantages'].tolist();ref=bounded_correct_process(y,m)['advantage']
        assert max(abs(a-b) for a,b in zip(got,ref))<3e-7;checked+=8
    before=read(PAUSED_AUDIT/'preflight.json');protected={}
    for n,h in read(PAUSED_AUDIT/'protected_before.json').items():
        assert (ROOT/n).is_file() and sha(ROOT/n)==h,('Previous protected file changed',n)
        protected[n]=h
    for folder in [PAUSED,PAUSED_SOURCE,PAUSED_AUDIT]:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix not in ['.lock','.pyc'] and '__pycache__' not in p.parts:
                protected[str(p.relative_to(ROOT))]=sha(p)
    schedule=before['first_epoch_question_groups'];assert len(schedule)>288
    manifest=read(ROOT/'data/prepared/manifest.json')
    for n,v in manifest['files'].items():assert sha(ROOT/'data/prepared'/n)==v['sha256']
    assets=read(ROOT/'receipts/assets.json');model=Path(assets['model_path'])
    for n,v in assets['files'].items():
        assert (model/n).is_file() and (model/n).stat().st_size==v['bytes']
    prm=ROOT/'models/qwen25_math_prm_7b'
    idx=read(prm/'model.safetensors.index.json')
    shards=sorted(set(idx['weight_map'].values()))
    for n in shards:assert (prm/n).is_file() and (prm/n).stat().st_size>0
    from audit_common import C_ADAPTER
    assert sha(C_ADAPTER/'adapter_model.safetensors')==read(HERE/'protocol.json')['C_adapter_sha256']
    origins={};start_steps={};prior_cost={}
    for arm in ARMS:
        v=pause['arms'][arm];start=v['checkpoint_step'];cp=Path(v['resume_checkpoint'])
        assert start==read(HERE/'protocol.json')['start_steps'][arm]
        assert v['completed_updates_beyond_checkpoint']==v['rollout_rows_beyond_checkpoint']==0
        assert sha(cp/'pilot_checkpoint.json')==v['checkpoint_contract_sha256']
        contract=read(cp/'pilot_checkpoint.json')
        for n,h in contract['files_sha256'].items():assert sha(cp/n)==h
        assert read(cp/'trainer_state.json')['global_step']==start
        account=read(cp/'completion_accounting.json')
        assert sha(cp/'completion_accounting.json')==contract['accounting_sha256']
        assert account['generated_completions']==account['loss_rows']==start*32
        oldrun=PAUSED/arm;rr=rows(oldrun/'rollouts.jsonl')
        assert len(rr)==start*32
        assert sum(r['completion_tokens'] for r in rr)==account['generated_completion_tokens']
        for k in range(start):
            block=rr[k*32:(k+1)*32]
            assert [r['problem_id'] for r in block[::8]]==schedule[k]
            assert all(r['optimizer_step_before_update']==k for r in block)
        assert not [p.name for p in oldrun.glob('raw_generation_step-*') if int(p.stem.split('-')[-1])>=start]
        assert read(oldrun/'progress.json')['global_step']==start
        run=PAIR/arm;run.mkdir();(run/'milestones').mkdir()
        target=run/f'milestones/checkpoint-{start}';shutil.copytree(cp,target)
        for p in cp.iterdir():
            if p.is_file():assert not p.is_symlink() and sha(target/p.name)==sha(p)
        prefixes={}
        for n in ['rollouts.jsonl','metrics.jsonl','update_audits.jsonl','generation_checks.jsonl']:
            shutil.copyfile(oldrun/n,run/n)
            assert sha(run/n)==sha(oldrun/n)
            prefixes[n]={'bytes':(run/n).stat().st_size,'sha256':sha(run/n)}
        atomic(run/'origin_prefix.json',{'source':str(oldrun),'source_checkpoint':str(cp),'checkpoint':str(target),
            'rows':start*32,'accounting':account,'prefix_files':prefixes})
        start_steps[arm]=start
        origins[arm]={'source':str(cp),'checkpoint':str(target),'step':start,'tokens':account['generated_completion_tokens'],
                      'adapter_sha256':sha(target/'adapter_model.safetensors'),'prefix_files':prefixes}
        prior_cost[arm]={'original_1M_result':str(ORIGINAL/arm/'final_result.json'),
            'previous_stage50_result':read(oldrun/'stage_50_result.json'),
            'previous_last_progress':read(oldrun/'progress.json'),
            'note':'Historical wall times are retained separately; resumed stage times do not include pre-pause work or idle days.'}
    atomic(PAIR/'acceptance_gate.json',{'passed':True,'inherited':str(PAUSED/'acceptance_gate.json'),
        'inherited_sha256':sha(PAUSED/'acceptance_gate.json'),'all_three_true_continuations':True,
        'origin_note':'Previous common48-to50 model checks passed; current start steps differ due user pause. New first-update checks remain mandatory.'})
    assert read(PAUSED/'acceptance_gate.json')['passed']
    assert not torch.cuda.is_initialized()
    atomic(AUDIT/'protected_before.json',protected)
    receipt={'passed':True,'source_release_sha256':release,'origins':origins,'start_steps':start_steps,
        'protected_files':len(protected),'CPU_reference_rows':checked,'runtime_bytes_identical':True,
        'advantages_bytes_identical':True,'evaluation_bytes_identical':True,'Acceptance_callback_AST_identical':True,
        'first_epoch_question_groups':schedule,'protocol':read(HERE/'protocol.json'),
        'original_training_settings':before['original_training_settings'],'CUDA_initialized':False,
        'dataset_hashes':{n:v['sha256'] for n,v in manifest['files'].items()},'gpu_identity':actual,
        'PRM_shards_present':shards,'prior_cost_evidence':prior_cost,'versions':versions}
    atomic(AUDIT/'preflight.json',receipt)
    event('admitted_CPU_only',protected_files=len(protected),start_steps=start_steps)
    print(json.dumps({'passed':True,'start_steps':start_steps,'protected_files':len(protected),'CPU_reference_rows':checked,'CUDA_initialized':False,'gpu_identity':actual}))

if __name__=='__main__':main()
