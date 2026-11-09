from common import *
import fcntl,subprocess,traceback

def main():
    verify();RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shared=(ROOT/'receipts/abc_seed17_resume_v1_gpu1.lock').open('a');fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'dispatch_started.json').exists(),'Existing attempt must be inspected'
    atomic(RUN/'dispatch_started.json',{'pid':os.getpid(),'unix':time.time(),'release_sha256':verify()})
    children=[];deadline=time.time()+72*3600
    def child(label,script,args=(),cpu=False,required_MiB=48000):
        verify();env=os.environ.copy()
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        else:
            guard()
            while True:
                assert time.time()<deadline,'Pipeline deadline reached; no new stages'
                free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=required_MiB:break
                event('waiting_for_capacity',pending=label,free_MiB=free,children=children);time.sleep(30)
        cmd=[sys.executable,'-B','-u',str(HERE/script),*args]
        log=(RUN/(label+'.log')).open('xb');p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        row={'stage':label,'pid':p.pid,'started_unix':time.time(),'command':cmd};children.append(row)
        event(label,children=children,active_child=row)
        rc=p.wait();row.update(exit_code=rc,finished_unix=time.time());log.close();atomic(RUN/'children.json',children)
        if rc:raise RuntimeError(label+' failed; preserve evidence, no automatic retry')
        event(label+'_complete',children=children,active_child=None)
    try:
        diagnostic=AUDIT/'placeholder_diagnostic/result.json'
        assert diagnostic.exists() and read(diagnostic)['passed'],'Complete independent diagnostic first; its numerical result does not change training'
        child('preflight','preflight.py',cpu=True)
        for arm in ARMS:child('no_update_'+arm,'smoke.py',['--arm',arm])
        for repeat in [0,1]:child(f'baseline_validation_repeat{repeat}','evaluate.py',['--cohort','baseline','--repeat',str(repeat)],required_MiB=20000)
        for arm in ARMS:child('train_'+arm+'_432','train_arm.py',['--arm',arm,'--target','432'])
        for repeat in [0,1]:child(f'nineM_validation_repeat{repeat}','evaluate.py',['--cohort','nine','--repeat',str(repeat)],required_MiB=20000)
        for arm in reversed(ARMS):child('train_'+arm+'_576','train_arm.py',['--arm',arm,'--target','576'])
        for repeat in [0,1]:child(f'final_validation_repeat{repeat}','evaluate.py',['--cohort','final','--repeat',str(repeat)],required_MiB=20000)
        child('freeze_seed42_eligibility','analysis.py',['--gate-only'],cpu=True)
        for repeat in [0,1]:child(f'common_endpoints_math_repeat{repeat}','evaluate.py',['--cohort','endpoints_math','--repeat',str(repeat)],required_MiB=20000)
        child('common_endpoints_gsm','evaluate.py',['--cohort','endpoints_gsm','--repeat','0'],required_MiB=20000)
        child('final_CPU_acceptance','analysis.py',cpu=True)
        assert read(RUN/'comparison.json')['passed']
        event('complete',children=children,active_child=None,comparison=str(RUN/'comparison.json'))
    except Exception as e:event('failed',children=children,error=str(e),traceback=traceback.format_exc(),automatic_retry=False);raise

if __name__=='__main__':main()
