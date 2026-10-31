"""One finite paired-PRM job. No actor RL launch and no automatic retry."""
import fcntl
import subprocess
import traceback
from common import *


def main():
    assert_root();verify_release();OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'controller.json').exists():
        raise RuntimeError('Controller receipt already exists; inspect checkpoints instead of launching again')
    started=time.time();receipt=dict(status='running',pid=os.getpid(),started_unix=started,release_sha256=verify_release(),gpu=1,
        sharing_authorized=True,completed_stages=[],active_child=None,actor_optimizer_updates=0,automatic_RL_launch=False)
    atomic(OUT/'controller.json',receipt)
    stages=[('prepare',[str(HERE/'prepare.py'),'select'],False),('generate',[str(HERE/'generate.py')],True),
        ('schedule',[str(HERE/'prepare.py'),'schedule'],False),('preflight',[str(HERE/'adapt.py'),'preflight'],True),
        ('adapt_C',[str(HERE/'adapt.py'),'C'],True),('adapt_D',[str(HERE/'adapt.py'),'D'],True),
        ('paired_wiring',[str(HERE/'adapt.py'),'wiring'],False),
        ('score_B',[str(HERE/'score.py'),'B'],True),('score_C',[str(HERE/'score.py'),'C'],True),('score_D',[str(HERE/'score.py'),'D'],True),
        ('analyze',[str(HERE/'analyze.py')],False)]
    try:
        for name,args,gpu in stages:
            verify_release()
            if gpu:
                while True:
                    free=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
                    if int(free)>=CFG['minimum_free_GPU_MiB']:break
                    receipt.update(status='waiting_for_memory',next_stage=name,free_memory_MiB=int(free));atomic(OUT/'controller.json',receipt)
                    assert time.time()-started<CFG['maximum_controller_seconds'];time.sleep(30)
            log=OUT/f'{name}.log';f=log.open('a')
            child=subprocess.Popen([sys.executable,'-B','-u',*args],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=os.environ.copy())
            receipt.update(status='running',active_child=dict(stage=name,pid=child.pid,log=str(log)),stage_started_unix=time.time())
            atomic(OUT/'controller.json',receipt)
            try:code=child.wait(timeout=CFG['maximum_stage_seconds'])
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=60);raise RuntimeError(f'{name} timed out; only this job child was terminated')
            f.close()
            if code:raise RuntimeError(f'{name} exited {code}; no automatic retry; preserve checkpoint and inspect log')
            receipt['completed_stages'].append(name);receipt['active_child']=None;atomic(OUT/'controller.json',receipt)
        receipt.update(status='complete',finished_unix=time.time(),preservation=protect());atomic(OUT/'controller.json',receipt)
    except Exception as e:
        receipt.update(status='failed',failure=str(e),traceback=traceback.format_exc(),failed_unix=time.time())
        atomic(OUT/'controller.json',receipt);raise


if __name__=='__main__':main()
