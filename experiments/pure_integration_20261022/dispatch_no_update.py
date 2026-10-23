"""One idempotent dispatch for the explicitly authorized no-update check."""
import fcntl
import os
import subprocess
import sys
import time
from common import ROOT, HERE, read, atomic, verify_release

def main():
    assert sys.argv[1:] == ['--authorized-no-update'], 'Explicit bounded dispatch flag required'
    audit=ROOT/'audits/PURE_integration_20261022_cpu_v3'
    assert read(audit/'core_checks.json')['passed']
    assert read(audit/'tokenizer_checks.json')['passed']
    # Historical reconstruction failures are reported, never promoted to exact
    # generated-ID evidence; real preflight must independently pass exact IDs.
    verify_release()
    path=ROOT/'audits/PURE_integration_20261022_dispatch_v3'
    path.mkdir(exist_ok=True)
    lock=(path/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=path/'dispatch.json';run=ROOT/'runs/PURE_no_update_20261022_v3'
    assert not receipt.exists() and not run.exists(), 'Existing receipt/run: inspect, do not duplicate'
    procs=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    active=[l for l in procs.splitlines() if 'pure_integration_20261022/no_update.py' in l and 'python' in l]
    assert not active, active
    gpu=1
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    line=[l for l in snapshot.splitlines() if l.split(',')[0].strip()==str(gpu)][0]
    assert int(line.split(',')[2]) >= 56000, 'Insufficient shared-device headroom; do not launch'
    headers=read(ROOT/'receipts/python_dev_headers.json')
    env=os.environ.copy()
    for k in list(env):
        if k.lower() in {'http_proxy','https_proxy','all_proxy','no_proxy','pip_proxy'}:env.pop(k)
    env.update(PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=str(gpu),PURE_PHYSICAL_GPU=str(gpu),
        HF_HOME=str(ROOT/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        VLLM_CACHE_ROOT=str(ROOT/'cache/vllm'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'cache/torchinductor'),VLLM_USE_V1='0',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',
        MASTER_ADDR='127.0.0.1',MASTER_PORT='29629',DO_NOT_TRACK='1',WANDB_MODE='disabled',
        C_INCLUDE_PATH=os.pathsep.join(headers['include_paths']))
    command=[str(ROOT/'.venv/bin/python'),str(HERE/'no_update.py'),'--run',str(run),'--i-understand-no-updates',
        '--reuse-exact-rollouts',str(ROOT/'runs/PURE_no_update_20261022_v2/generated_exact_ids.jsonl')]
    log=path/'runner.log'
    with log.open('x') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
    atomic(receipt,dict(pid=child.pid,run=str(run),log=str(log),command=command,
        launched=time.time(),gpu_snapshot=snapshot,shared_gpu=True,formal_training=False,
        optimizer_updates_authorized=False))
    print(receipt.read_text(),flush=True)

if __name__=='__main__':main()
