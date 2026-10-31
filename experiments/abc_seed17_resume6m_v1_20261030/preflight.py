"""CPU admission; preserve originals and copy exact continuation states once."""
from pilot_common import *
import ast,shutil,sys,torch

def main():
    guard();release=verify();assert not (AUDIT/'preflight.json').exists(),'Existing admission; inspect first'
    assert read(ORIGINAL/'controller.json')['stage']=='complete'
    prior=read(ROOT/'audits/C_bounded_triplet_seed17_1M_20261029_v2/preflight.json')
    previous_source=ROOT/'experiments/c_bounded_pilot_20261029_v2'
    assert sha(HERE/'advantages.py')==sha(previous_source/'advantages.py')
    def selected(path):
        return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,(ast.ClassDef,ast.FunctionDef))}
    old,new=selected(previous_source/'runtime.py'),selected(HERE/'runtime.py')
    for name in ['components_for_arm','OutcomeLedger','new_accounting','PilotMixin','trainer_class']:
        assert old[name]==new[name],('Reward/loss runtime changed',name)
    from runtime import trainer_class,components_for_arm
    trainer_class()
    from advantages import compute_advantages
    sys.path.insert(0,str(ORIGIN));from cpu_reference import bounded_correct_process
    checked=0
    for group in rows(ROOT/'runs/CD_PRM7B_pair_seed17_20261028/group_advantages.jsonl'):
        if group['arm']!='C':continue
        y=group.get('outcomes',group.get('outcome_rewards'));m=group['min_scores']
        got=compute_advantages(y,m)['advantages'].tolist();ref=bounded_correct_process(y,m)['advantage']
        assert max(abs(a-b) for a,b in zip(got,ref))<3e-7;checked+=8
    probe=components_for_arm('A',[1,0,1,0,0,1,0,0],[0.]*8,[False]*8)
    assert torch.equal(probe['advantages'],probe['outcome_component'])
    protected={}
    inherited=ROOT/'audits/A6M_protocol_recheck_20261029/protected_before.json'
    for n,h in read(inherited).items():assert sha(ROOT/n)==h;protected[n]=h
    def keep(path):protected[str(path.relative_to(ROOT))]=sha(path)
    for parent in [ORIGINAL,ROOT/'runs/A6M_protocol_recheck_20261029',ROOT/'experiments/a6m_protocol_recheck_20261029']:
        for p in parent.rglob('*'):
            if p.is_file() and p.suffix!='.lock':keep(p)
    manifest=read(ROOT/'data/prepared/manifest.json')
    for name,item in manifest['files'].items():assert sha(ROOT/'data/prepared'/name)==item['sha256']
    keep(ROOT/'data/prepared/manifest.json')
    schedule=prior['first_epoch_question_groups'];assert len(schedule)>288
    initial_specs=[];origins={}
    for arm in ARMS:
        oldrun=ORIGINAL/arm;final=read(oldrun/'final_result.json');assert final['passed'] and final['global_step']==48
        cp=Path(final['checkpoint']);assert cp==oldrun/'checkpoint-48'
        contract=read(cp/'pilot_checkpoint.json')
        for n,h in contract['files_sha256'].items():assert sha(cp/n)==h
        assert sha(cp/'adapter_model.safetensors')==final['adapter_sha256']
        account=read(cp/'completion_accounting.json');assert account==final['accounting']
        assert account['generated_completions']==account['loss_rows']==48*32
        run=PAIR/arm;run.mkdir();(run/'milestones').mkdir()
        target=run/'milestones/checkpoint-48';shutil.copytree(cp,target)
        for p in cp.iterdir():
            if p.is_file():assert not p.is_symlink() and sha(target/p.name)==sha(p)
        prefixes={}
        for name in ['rollouts.jsonl','metrics.jsonl','update_audits.jsonl','generation_checks.jsonl']:
            shutil.copyfile(oldrun/name,run/name);assert sha(run/name)==sha(oldrun/name);prefixes[name]={'sha256':sha(run/name),'bytes':(run/name).stat().st_size}
        rr=rows(run/'rollouts.jsonl');assert len(rr)==1536
        for step in range(48):assert [r['problem_id'] for r in rr[step*32:(step+1)*32:8]]==schedule[step]
        for src,dst in [('run_config.json','origin_run_config.json'),('final_result.json','origin_final_result.json')]:shutil.copyfile(oldrun/src,run/dst)
        atomic(run/'origin_prefix.json',{'source':str(oldrun),'prefix_files':prefixes,'rows':1536,'source_checkpoint':str(cp),'copied_checkpoint':str(target),'accounting':account})
        origins[arm]={'checkpoint':str(target),'original':str(cp),'adapter_sha256':final['adapter_sha256'],'tokens':account['generated_completion_tokens']}
        s=read(oldrun/'run_config.json')['settings'];initial_specs.append({k:v for k,v in s.items() if k not in ['output_dir','logging_dir','run_name']})
    assert all(s==initial_specs[0] for s in initial_specs)
    assert not torch.cuda.is_initialized()
    atomic(AUDIT/'protected_before.json',protected)
    receipt={'passed':True,'source_release_sha256':release,'originals':origins,'protected_files':len(protected),'CPU_reference_rows':checked,
        'reward_loss_functions_AST_identical':True,'advantage_source_identical':True,'first_epoch_question_groups':schedule,
        'protocol':read(HERE/'protocol.json'),'original_training_settings':initial_specs[0],'CUDA_initialized':False,
        'dataset_hashes':{k:v['sha256'] for k,v in manifest['files'].items()}}
    atomic(AUDIT/'preflight.json',receipt)
    event('admitted_CPU_only',protected_files=len(protected),starting_step=48)
    print(json.dumps({'passed':True,'protected_files':len(protected),'CPU_reference_rows':checked,'originals':origins}))
if __name__=='__main__':main()
