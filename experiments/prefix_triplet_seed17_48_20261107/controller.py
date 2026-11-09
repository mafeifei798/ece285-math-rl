"""Finite sequential 3x48-update job, followed by one validation and analysis."""
from common import *
import subprocess,sys,fcntl,traceback

def main():
    guard();verify();RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'dispatch_started.json').exists(),'Inspect an existing attempt; never redispatch automatically'
    atomic(RUN/'dispatch_started.json',{'pid':os.getpid(),'unix':time.time(),'release_sha256':verify()})
    children=[]
    def child(label,script,args=(),cpu=False):
        verify();env=os.environ.copy()
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        else:
            occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if occupied:raise RuntimeError('GPU acquired by another job; preserve completed stages and inspect before continuation')
        command=[sys.executable,'-B','-u',str(HERE/script),*args]
        log=(RUN/f'{label}.log').open('xb');p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record={'stage':label,'pid':p.pid,'started_unix':time.time(),'command':command};children.append(record)
        event(label,active_child=record,children=children)
        code=p.wait();record.update(exit_code=code,finished_unix=time.time());log.close()
        atomic(RUN/'children.json',children)
        if code:raise RuntimeError(f'{label} failed with exit code {code}; no automatic retry')
        event(label+'_complete',active_child=None,children=children)
    try:
        child('preflight','preflight.py',cpu=True)
        for arm in ARMS:child('train_'+arm,'train_arm.py',['--arm',arm])
        child('validation','evaluate.py')
        child('analysis','analyze.py',cpu=True)
        assert read(RUN/'comparison.json')['passed']
        event('complete',active_child=None,children=children,comparison=str(RUN/'comparison.json'))
    except Exception as exc:
        event('failed',active_child=None,children=children,error=str(exc),traceback=traceback.format_exc(),automatic_retry=False);raise
if __name__=='__main__':main()
