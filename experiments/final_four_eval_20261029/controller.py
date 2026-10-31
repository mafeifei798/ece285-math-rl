"""Finite evaluation + CPU audit. No optimizer, retries, schedules or model selection."""
from common_eval import *
import sys,subprocess,fcntl,traceback
def main():
    guard();release_check();assert read(AUDIT/'preflight.json')['passed']
    lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'comparison.json').exists(),'Evaluation already complete'
    atomic(RUN/'controller.json',{'pid':os.getpid(),'stage':'starting','unix':time.time(),'training':False})
    try:
        for filename,stage,cuda in [('generate.py','generation','1'),('analyze.py','CPU_acceptance','')]:
            if filename=='generate.py' and (RUN/'generation_complete.json').exists():continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=cuda)
            log=RUN/(stage+'.log')
            with log.open('a') as out:
                child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/filename)],env=env,stdout=out,stderr=subprocess.STDOUT)
                atomic(RUN/'controller.json',{'pid':os.getpid(),'stage':stage,'child_pid':child.pid,'log':str(log),'unix':time.time(),'training':False})
                code=child.wait();assert code==0,(stage,code,'No automatic retry; preserve committed batches')
        atomic(RUN/'controller.json',{'pid':os.getpid(),'stage':'complete','unix':time.time(),'training':False,'comparison':str(RUN/'comparison.json')})
    except Exception as exc:
        atomic(RUN/'controller_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'unix':time.time(),'no_automatic_retry':True})
        event('failed_stopped',error=str(exc));raise
if __name__=='__main__':main()
