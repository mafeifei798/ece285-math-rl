"""Operational queue only: wait for GPU1 to be unused, then run frozen inference."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'runs/AB_prompt_validation_20261022'
sys.path.insert(0,str(ROOT/'training'))
from common import atomic_json,file_hash

def inspect_gpu():
    ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    row=next(r.split(',') for r in ids.splitlines() if r.split(',')[0].strip()=='1')
    uuid=row[1].strip()
    allapps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    active=[line for line in allapps.splitlines() if line.split(',')[0].strip()==uuid]
    return {'gpu':1,'memory_used_MiB':int(row[2]),'utilization_percent':int(row[3]),'active_compute_processes':active,
        'idle':not active and int(row[2])<1000 and int(row[3])<10}

def main():
    lock=(OUT/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'dispatch.json').exists(),'Dispatch receipt already exists; do not duplicate'
    assert not (OUT/'controller.json').exists() and not (OUT/'state.json').exists()
    state={'status':'waiting_for_gpu','pid':os.getpid(),'started_unix':time.time(),'heartbeat_unix':time.time(),
        'script_sha256':file_hash(Path(__file__)),'release_sha256':file_hash(OUT/'release.json'),
        'training':False,'scope':'No generation until GPU1 is unused. Then exactly the frozen four-cell validation run and CPU analysis.',
        'maximum_wait_seconds':43200,'interval_seconds':60,'child_pid':None}
    def record(**kw):state.update(kw,heartbeat_unix=time.time());atomic_json(OUT/'dispatch.json',state)
    record()
    try:
        while True:
            gpu=inspect_gpu();record(gpu_check=gpu)
            if gpu['idle']:break
            if time.time()-state['started_unix']>state['maximum_wait_seconds']:
                record(status='gpu_wait_expired',finished_unix=time.time());return
            time.sleep(60)
        release=json.loads((OUT/'release.json').read_text())
        assert file_hash(OUT/'release.json')==state['release_sha256']
        for rel,h in release['input_sha256'].items():assert file_hash(ROOT/rel)==h
        with (OUT/'launch.log').open('x') as log:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'launch.py')],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            record(status='running',child_pid=child.pid)
            while child.poll() is None:time.sleep(30);record()
            record(child_pid=None,child_exit_code=child.returncode,finished_unix=time.time())
            assert child.returncode==0,'Frozen inference controller failed; evidence preserved, no automatic relaunch'
        record(status='completed')
    except BaseException as exc:
        record(status='failed',error=str(exc),traceback=traceback.format_exc(),finished_unix=time.time());raise

if __name__=='__main__':main()
