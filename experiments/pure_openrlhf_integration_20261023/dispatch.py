"""Receipt-gated dispatch of exactly one authorized no-update probe."""
import fcntl
import os
import subprocess
import sys
import time
from common import ROOT,HERE,read,atomic,verify_release,sha


def main():
    assert sys.argv[1:]==['--authorized-no-update']
    verify_release()
    audit=ROOT/'audits/PURE_OpenRLHF_integration_20261023'
    cpu=read(audit/'cpu_checks.json')
    assert cpu['passed']
    for name,expected in cpu['source_files'].items():
        assert sha(HERE/name)==expected,('Stale CPU receipt',name)
    lock=(audit/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=audit/'dispatch.json';run=ROOT/'runs/PURE_OpenRLHF_no_update_20261023_v1'
    assert not receipt.exists() and not run.exists(),'Existing run/receipt; inspect rather than launch twice'
    ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    active=[line for line in ps.splitlines() if str(HERE/'no_update.py') in line and 'python' in line]
    assert not active,active
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpu=1
    line=[line for line in snapshot.splitlines() if line.split(',')[0].strip()==str(gpu)][0]
    assert int(line.split(',')[2])>=56000,'Insufficient headroom; no launch'
    env=os.environ.copy()
    for key in list(env):
        if key.lower() in {'http_proxy','https_proxy','all_proxy','no_proxy','pip_proxy'}:env.pop(key)
    headers=read(ROOT/'receipts/python_dev_headers.json')
    env.update(PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=str(gpu),PURE_PHYSICAL_GPU=str(gpu),
        HF_HOME=str(ROOT/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        VLLM_CACHE_ROOT=str(ROOT/'cache/vllm'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'cache/torchinductor'),VLLM_USE_V1='0',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',MASTER_ADDR='127.0.0.1',
        MASTER_PORT='29643',DO_NOT_TRACK='1',WANDB_MODE='disabled',
        C_INCLUDE_PATH=os.pathsep.join(headers['include_paths']))
    command=[str(ROOT/'.venv/bin/python'),'-B','-u',str(HERE/'no_update.py'),
             '--run',str(run),'--authorized-no-update']
    log=audit/'runner.log'
    with log.open('x') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,cwd=ROOT,env=env,start_new_session=True)
    atomic(receipt,dict(pid=child.pid,run=str(run),log=str(log),command=command,
        launched=time.time(),gpu_snapshot=snapshot,shared_gpu=True,formal_training=False,
        optimizer_updates_authorized=False,release_sha256=sha(HERE/'release.json')))
    print(receipt.read_text(),flush=True)


if __name__=='__main__':main()
