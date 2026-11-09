"""Audited pre-update recovery: restore existing CPATH, keep experiment immutable."""
from pathlib import Path
import sys,os,json,subprocess,time,fcntl,traceback
ROOT=Path('/path/to/ece285_math_rl_project')
sys.path.insert(0,str(ROOT/'experiments/length_norm_pair_seed17_48_20261107'))
from common import *


def main():
    guard();verify();source_unchanged()
    lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'header_recovery_started.json').exists(),'Recovery already attempted; inspect existing state'
    assert read(RUN/'controller_initial_header_failure.json')['stage']=='failed'
    assert read(AUDIT/'cpu_check.json')['passed']
    header=read(ROOT/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json')
    assert header['passed'] and os.environ['CPATH']==header['CPATH']
    for arm in ARMS:
        p=RUN/arm
        assert not (p/'intent.json').exists() and not list(p.glob('raw_step_*.pt')) and not list(p.glob('checkpoint-*'))
    atomic(RUN/'header_recovery_started.json',{'pid':os.getpid(),'unix':time.time(),
        'initial_release_sha256':verify(),'change':'Restore previously validated project-local CPATH only',
        'prior_training_updates':0,'prior_generated_training_batches':0,'settings_and_budget_unchanged':True})
    children=[]
    def child(label,script,args=(),cpu=False):
        verify();env=os.environ.copy()
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        else:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU job exists; preserve completed stages'
        cmd=[sys.executable,'-B','-u',str(HERE/script),*args]
        with (RUN/(label+'.log')).open('xb') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            v={'stage':label,'pid':p.pid,'command':cmd,'started_unix':time.time()};children.append(v)
            event(label,active_child=v,children=children,header_environment_recovery=True)
            code=p.wait();v.update(exit_code=code,finished_unix=time.time())
        atomic(RUN/'recovery_children.json',children)
        if code:raise RuntimeError(f'{label} failed with exitcode{code}; no automatic retry')
    try:
        child('no_update_header_recovered','smoke_no_update.py')
        assert read(AUDIT/'no_update/result.json')['passed']
        for arm in ARMS:child('train_'+arm,'train_arm.py',['--arm',arm])
        child('validation','evaluate.py')
        child('analysis','analyze.py',cpu=True)
        assert read(RUN/'comparison.json')['passed']
        event('complete',active_child=None,children=children,header_environment_recovery=True,comparison=str(RUN/'comparison.json'))
    except Exception as exc:
        event('failed',active_child=None,children=children,header_environment_recovery=True,error=str(exc),traceback=traceback.format_exc(),automatic_retry=False)
        raise


if __name__=='__main__':main()
