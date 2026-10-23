"""One-shot inference+CPU-analysis controller; never calls a training entrypoint."""
from pathlib import Path
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'runs/AB_prompt_validation_20261022'
sys.path.insert(0,str(ROOT/'training'))
from common import atomic_json,file_hash
sys.path.insert(0,str(ROOT/'operations'))
from run_continuation import environment

def main():
    locks=[]
    for path in [OUT/'controller.lock',ROOT/'receipts/ab_corrected_gpu1.lock']:
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert not (OUT/'controller.json').exists(),'Controller receipt exists; inspect, do not duplicate launch'
    assert not (OUT/'state.json').exists() and not (OUT/'predictions.jsonl').exists()
    release=json.loads((OUT/'release.json').read_text())
    for rel,h in release['input_sha256'].items():assert file_hash(ROOT/rel)==h
    state={'status':'starting','pid':os.getpid(),'started_unix':time.time(),'heartbeat_unix':time.time(),
        'training':False,'scope':'Existing corrected A/B checkpoints, fixed2x2 prompt comparison;998 held-out questions; no training or PRM scoring.',
        'release_sha256':file_hash(OUT/'release.json'),'completed_stages':[]}
    atomic_json(OUT/'controller.json',state)
    def record(**kw):
        state.update(kw,heartbeat_unix=time.time());atomic_json(OUT/'controller.json',state)
    try:
        env=environment(1);env.update(HF_DATASETS_OFFLINE='1')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));env['MASTER_PORT']=str(sock.getsockname()[1])
        record(master_port=env['MASTER_PORT'])
        for stage in ['evaluate','analyze']:
            log=OUT/(stage+'.log')
            with log.open('x') as f:
                child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE/(stage+'.py'))],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                record(status=stage,child_pid=child.pid,log=str(log))
                while child.poll() is None:
                    time.sleep(15);record()
                record(child_pid=None,last_exit_code=child.returncode)
                assert child.returncode==0,f'{stage} failed; no automatic restart'
            state['completed_stages'].append(stage);record()
        analysis=json.loads((OUT/'analysis.json').read_text());assert analysis['passed'] and analysis['independently_reverified']==3992
        record(status='completed',finished_unix=time.time(),all_predictions_reverified=True,
            old_results_preserved=True,new_training=False)
    except BaseException as exc:
        record(status='failed',error=str(exc),traceback=traceback.format_exc(),finished_unix=time.time());raise

if __name__=='__main__':main()
