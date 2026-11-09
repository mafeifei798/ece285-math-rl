"""Two finite no-update stages, each with a fresh model process."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'audits/prefix_protection_20261107'
if __name__=='__main__':
    lock=(OUT/'validation_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'no_update_result.json').exists()
    for name in ['no_update.py','merged_probe.py']:
        with (OUT/(name+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).with_name(name))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            (OUT/'controller.json').write_text(json.dumps({'stage':name,'pid':os.getpid(),'child_pid':child.pid,'unix':time.time(),'optimizer_steps':0})+'\n')
            code=child.wait()
        if code:
            (OUT/'controller_failure.json').write_text(json.dumps({'stage':name,'exit_code':code,'unix':time.time(),'optimizer_steps':0})+'\n')
            raise SystemExit(code)
    assert json.loads((OUT/'no_update_result.json').read_text())['completed']
    (OUT/'controller.json').write_text(json.dumps({'stage':'complete','pid':os.getpid(),'unix':time.time(),'optimizer_steps':0})+'\n')
