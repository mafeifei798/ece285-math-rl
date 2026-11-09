"""One explicit recovery after pre-generation Triton cache failure; fixed pipeline."""
from pathlib import Path
import sys,os,fcntl,time,traceback,json

ROOT=Path('/path/to/ece285_math_rl_project')
SOURCE=ROOT/'experiments/c_bounded_pilot_20261029_v2'
sys.path.insert(0,str(SOURCE))
from pilot_common import *
from controller import run

def main():
    guard();release=verify()
    lock=(PAIR/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (PAIR/'environment_recovery_started.json').exists(),'Recovery was already attempted; inspect rather than redispatch'
    assert read(PAIR/'controller.json')['stage']=='failed_stopped'
    assert 'InductorError' in read(PAIR/'A/failure.json')['traceback']
    assert not (PAIR/'A/run_config.json').exists() and not list((PAIR/'A').glob('checkpoint-*'))
    assert not (PAIR/'A/raw_generation_step-0.pt').exists() and not (PAIR/'A/update_audits.jsonl').exists()
    assert read(AUDIT/'preflight.json')['source_release_sha256']==release
    assert read(AUDIT/'cache_migration.json')['source_unchanged']
    archive=AUDIT/'initialization_failure';archive.mkdir(exist_ok=False)
    for name,path in [('A_failure.json',PAIR/'A/failure.json'),('A_status.json',PAIR/'A/status.json'),
                      ('controller_failure.json',PAIR/'controller_failure.json'),('controller_status.json',PAIR/'controller.json')]:
        (archive/name).write_bytes(path.read_bytes())
    atomic(PAIR/'environment_recovery_started.json',dict(pid=os.getpid(),unix=time.time(),source_release_sha256=release,
        prior_generation_calls=0,prior_optimizer_updates=0,prior_checkpoint_absent=True,
        change='Only same-environment compiled-cache migration; released reward/trainer/seed/budget files unchanged'))
    try:
        for a in ARMS:run('train_arm.py','--arm',a,'--phase','smoke',stage=a+'_two_updates_env_recovery')
        smoke={a:read(PAIR/a/'smoke_result.json') for a in ARMS}
        assert all(s['passed'] and s['global_step']==2 for s in smoke.values())
        specs=[read(PAIR/a/'run_config.json') for a in ARMS]
        assert len({s['initial_adapter_digest'] for s in specs})==1
        normalized=[{k:v for k,v in s['settings'].items() if k not in ['output_dir','logging_dir','run_name']} for s in specs]
        assert all(s==normalized[0] for s in normalized)
        initial=[[(r['problem_id'],r['completion_token_ids']) for r in rows(PAIR/a/'rollouts.jsonl')[:32]] for a in ARMS]
        assert all(v==initial[0] for v in initial)
        atomic(PAIR/'smoke_gate.json',dict(passed=True,source_release_sha256=release,all_three_real_updates=smoke,
            matched_initial_adapter=True,matched_training_arguments=True,matched_initial_raw_generation=True))
        run('evaluate.py','--arm','initial',stage='initial_validation')
        for a in ARMS:
            run('train_arm.py','--arm',a,'--phase','continue',stage=a+'_continue_to_1M')
            run('evaluate.py','--arm',a,stage=a+'_validation')
        run('analyze.py',stage='paired_analysis')
        event('complete',comparison=str(PAIR/'comparison.json'),no_6M_started=True,D_adaptation=False)
    except Exception as e:
        atomic(PAIR/'environment_recovery_failure.json',dict(error=str(e),traceback=traceback.format_exc(),unix=time.time()))
        event('failed_stopped',error=str(e),no_automatic_retry=True);raise

if __name__=='__main__':main()
