"""Manually admitted finite recovery. Never repeats the completed bounded arm."""
from common import *
import subprocess,sys,fcntl,traceback

def main():
    guard();verify();lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(RUN/'recovery_admission.json')['passed']
    assert read(RUN/'bounded/result.json')['passed']
    assert not (RUN/'recovery_dispatch_started.json').exists()
    atomic(RUN/'recovery_dispatch_started.json',{'pid':os.getpid(),'unix':time.time(),'release_sha256':verify()})
    children=[]
    def child(label,path,args=(),cpu=False):
        verify();env=os.environ.copy()
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        else:assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU job exists'
        command=[sys.executable,'-B','-u',str(path),*args];log=(RUN/f'{label}.log').open('xb')
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record={'stage':label,'pid':p.pid,'started_unix':time.time(),'command':command};children.append(record)
        event(label,active_child=record,children=children,recovery=True)
        code=p.wait();record.update(exit_code=code,finished_unix=time.time());log.close();atomic(RUN/'recovery_children.json',children)
        if code:raise RuntimeError(f'{label} failed with exit code {code}; no automatic retry')
        event(label+'_complete',active_child=None,children=children,recovery=True)
    try:
        child('recover_uniform',HERE/'train_arm.py',['--arm','uniform'])
        child('train_prefix_recovered_gate',HERE/'train_arm.py',['--arm','prefix'])
        child('validation',ORIGINAL_HERE/'evaluate.py')
        child('analysis',ORIGINAL_HERE/'analyze.py',cpu=True)
        x=read(RUN/'comparison.json');assert x['passed']
        x['recovery']={'reason':'Overstrict first stochastic generation bitwise-equality gate','bounded_rerun':False,
            'uniform_first_batch_reused_without_resampling':True,'uniform_updates_before_recovery':0,
            'uniform_sampling_rng_after_fault_bitwise_preserved':False,
            'uniform_failed_attempt_seconds':read(RUN/'recovery_admission.json')['uniform_failed_attempt_seconds'],
            'uniform_first_PRM_forward_seconds_unavailable':True,
            'seed_and_model_and_loss_and_question_order_settings_unchanged':True,
            'extra_limitation':'Fault recovery reused valid on-policy cached responses, but could not preserve post-generation sampling RNG exactly.'}
        x['training_costs']['uniform']['total_seconds']+=x['recovery']['uniform_failed_attempt_seconds']
        x['training_costs']['uniform']['PRM_seconds_incomplete']=True
        atomic(RUN/'comparison.json',x)
        event('complete',active_child=None,children=children,recovery=True,comparison=str(RUN/'comparison.json'))
    except Exception as exc:
        event('failed',active_child=None,children=children,recovery=True,error=str(exc),traceback=traceback.format_exc(),automatic_retry=False);raise
if __name__=='__main__':main()
