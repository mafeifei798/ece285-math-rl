"""Continue admitted smoke checkpoints after header-only environment repair."""
from pathlib import Path
import sys,os,fcntl,time,traceback
ROOT=Path('/path/to/ece285_math_rl_project')
sys.path.insert(0,str(ROOT/'experiments/c_bounded_pilot_20261029_v2'))
from pilot_common import *
from controller import run

def main():
    guard();release=verify()
    lock=(PAIR/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (PAIR/'header_recovery_started.json').exists(),'This recovery already attempted; inspect checkpoint before retry'
    assert read(PAIR/'smoke_gate.json')['passed']
    assert read(AUDIT/'smoke_acceptance_independent.json')['passed']
    header=read(AUDIT/'python_header_repair.json');assert header['passed'] and os.environ['CPATH']==header['CPATH']
    for a in ARMS:
        result=read(PAIR/a/'smoke_result.json');assert result['global_step']==2 and result['passed']
        assert not (PAIR/a/'resume_build.json').exists(),'Never start an already attempted continuation implicitly'
        c=Path(result['checkpoint'])
        for n,h in read(c/'pilot_checkpoint.json')['files_sha256'].items():assert sha(c/n)==h
    assert not (PAIR/'validation/initial/predictions.jsonl').exists()
    atomic(PAIR/'header_recovery_started.json',dict(pid=os.getpid(),unix=time.time(),source_release_sha256=release,
        resume_from_existing_smoke_checkpoints=True,reward_loss_data_seed_budget_unchanged=True,
        environment_change='Project-local CPython3.11 headers via CPATH; system Python untouched',
        scheduling_change='Run actual checkpoint continuation acceptance first; immutable initial validation follows A pilot, with unchanged checkpoint/prompt/verifier/seed'))
    try:
        # Prioritize the real optimizer/RNG/next-prompt restore acceptance. The
        # initial model remains immutable; no validation selects these settings.
        run('train_arm.py','--arm','A','--phase','continue',stage='A_continue_to_1M')
        run('evaluate.py','--arm','initial',stage='initial_validation_headers_recovered')
        run('evaluate.py','--arm','A',stage='A_validation')
        for a in ['C_mix','C_bounded']:
            run('train_arm.py','--arm',a,'--phase','continue',stage=a+'_continue_to_1M')
            run('evaluate.py','--arm',a,stage=a+'_validation')
        run('analyze.py',stage='paired_analysis')
        event('complete',comparison=str(PAIR/'comparison.json'),no_6M_started=True,D_adaptation=False)
    except Exception as e:
        atomic(PAIR/'header_recovery_failure.json',dict(error=str(e),traceback=traceback.format_exc(),unix=time.time()))
        event('failed_stopped',error=str(e),no_automatic_retry=True);raise

if __name__=='__main__':main()
