"""CPU-only admission; source weights and fixed schedule remain unchanged."""
from common import *
import sys, inspect, importlib.metadata, itertools

def main():
    verify();source_unchanged()
    acceptance=read(ROOT/'audits/prefix_protection_20261107/ACCEPTANCE.json')
    assert acceptance['passed'] and acceptance['optimizer_steps']==0
    for n,h in acceptance['source_sha256'].items():assert sha(VALIDATED/n)==h,n
    full=read(ROOT/'audits/prefix_protection_20261107/native_trainer_eager/result.json')
    assert full['passed'] and full['actor_base_unchanged'] and full['actor_adapter_unchanged'] and full['frozen_PRM']
    hot=read(ROOT/'audits/prefix_protection_20261107/native_trainer_eager/hot_reload_probe.json')
    assert all(hot[k]==0 for k in ['a_same_adapter_repeat','b_same_adapter_repeat','new_adapter_id','return_old_id'])
    contract=read(CP/'pilot_checkpoint.json');assert contract['global_step']==START
    protocol=read(VALIDATED/'pilot_protocol.json');assert protocol['source_adapter_sha256']==sha(CP/'adapter_model.safetensors')
    schedule=read(ROOT/'audits/ABC_seed17_resumed_V1_20261104/preflight.json')['first_epoch_question_groups'][START:TARGET]
    assert len(schedule)==48 and all(len(q)==4 for q in schedule)
    train=rows(ROOT/'data/prepared/train.jsonl');validation=rows(ROOT/'data/prepared/validation.jsonl')
    ids={r['problem_id'] for r in train};assert len(ids)==len(train)==6993 and len(validation)==500
    assert all(q in ids for group in schedule for q in group)
    assert not ids.intersection(r['problem_id'] for r in validation)
    expected={'torch':'2.8.0','transformers':'4.56.2','trl':'0.23.1','peft':'0.17.1','vllm':'0.10.2'}
    versions={n:importlib.metadata.version(n) for n in expected}
    assert all(versions[n].split('+')[0]==v for n,v in expected.items())
    from trl.trainer.grpo_trainer import GRPOTrainer
    from trl.trainer.utils import RepeatSampler
    assert sha(inspect.getsourcefile(GRPOTrainer))=='f217183c84be9b38b23454920eb1b914726aa3c4b9dd4c076d455bdaa67bf1b8'
    sampler=iter(RepeatSampler(train,mini_repeat_count=8,batch_size=4,repeat_count=32,shuffle=True,seed=17))
    observed=[]
    for step in range(TARGET):
        block=list(itertools.islice(sampler,1024));assert len(block)==1024
        observed.append([train[i]['problem_id'] for i in block[:32:8]])
    assert observed[START:TARGET]==schedule,'Current sampler differs from frozen question schedule'
    from analyze import paired
    toy=[{'problem_id':str(i),'reward':y} for i,y in enumerate([1,0,1])]
    same=paired(toy,toy);assert same['difference_percentage_points']==0 and same['paired_bootstrap_95CI_pp']==[0,0]
    for arm in ARMS:
        p=RUN/arm;assert not (p/'intent.json').exists(),'Existing attempt requires inspection, not automatic restart'
        p.mkdir(parents=True,exist_ok=True)
    atomic(AUDIT/'preflight.json',{'passed':True,'source_checkpoint':str(CP),'source_contract':contract,'question_groups':schedule,
        'train_sha256':sha(ROOT/'data/prepared/train.jsonl'),'validation_sha256':sha(ROOT/'data/prepared/validation.jsonl'),
        'source_release_sha256':verify(),'validated_acceptance_sha256':sha(ROOT/'audits/prefix_protection_20261107/ACCEPTANCE.json'),
        'seed':17,'variants':ARMS,'start_step':START,'target_step':TARGET,'updates_per_arm':48,'evaluations':['validation'],'versions':versions,
        'live_sampler_matches_frozen_schedule':True,'paired_statistics_identity_check':True,
        'same_optimizer_and_scheduler_source':True,'vllm_rng_note':'Each arm starts a new V1 engine with seed0; historical vLLM RNG is not resumed bitwise.'})
    print(json.dumps({'preflight_passed':True,'new_updates_per_arm':48,'same_questions':192,'validation_questions':500}))
if __name__=='__main__':main()
