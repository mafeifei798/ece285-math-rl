"""One-shot receipt-gated frozen scoring and CPU analysis. Never trains."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'audits/PURE_official_prm_offline_20261022'
def atomic(name,data):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
def main():
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'scoring').exists(),'Inspect prior scoring before any restart'
    deadline=time.monotonic()+7200
    atomic('controller.json',dict(status='waiting_for_verified_weights',pid=os.getpid(),training=False))
    while not (OUT/'model_assets.json').exists():
        if time.monotonic()>deadline: raise TimeoutError('Asset wait expired; no GPU task started')
        pid=json.loads((OUT/'ranges_pid.json').read_text())['pid']
        if not Path(f'/proc/{pid}').exists(): raise RuntimeError('Download exited without verified assets')
        time.sleep(15)
    assets=json.loads((OUT/'model_assets.json').read_text());assert assets['complete']
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free>=24000:break
        atomic('controller.json',dict(status='waiting_for_24GB_free_on_shared_GPU1',pid=os.getpid(),free_MiB=free))
        if time.monotonic()>deadline:raise TimeoutError('Shared GPU memory unavailable')
        time.sleep(15)
    atomic('controller.json',dict(status='scoring',pid=os.getpid(),gpu=1,free_MiB_at_start=free,training=False))
    source=ROOT/'experiments/pure_official_prm_offline_20261022'
    input_path=ROOT/'audits/PURE_empty_boundary_20261022/inputs.jsonl'
    qwen_path=ROOT/'audits/PURE_empty_boundary_20261022/frozen_scoring_v2/scored.jsonl'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2'}
    with (OUT/'scoring.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(source/'score_existing.py'),'--input',str(input_path),
            '--qwen',str(qwen_path),'--model',str(ROOT/'models/pure_official_prm_7b'),
            '--assets',str(OUT/'model_assets.json'),'--output',str(OUT/'scoring')],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        atomic('scoring_pid.json',dict(pid=child.pid,controller_pid=os.getpid()))
        code=child.wait()
        if code:raise RuntimeError(f'Frozen scoring exited {code}; inspect saved evidence')
    atomic('controller.json',dict(status='CPU_analysis',pid=os.getpid(),training=False))
    (OUT/'analysis').mkdir(exist_ok=False)
    with (OUT/'analysis.log').open('x') as log:
        subprocess.run([sys.executable,'-u',str(source/'analyze.py'),'--output',str(OUT/'analysis'),
            '--inputs',str(input_path),'--qwen-scores',str(qwen_path),
            '--pure-scores',str(OUT/'scoring/scored.jsonl'),'--evidence',str(OUT/'scoring/result.json')],
            cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic('controller.json',dict(status='completed',pid=os.getpid(),training=False,optimizer_steps=0,new_generations=0))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic('controller.json',dict(status='failed',pid=os.getpid(),error=repr(exc),training=False));raise
