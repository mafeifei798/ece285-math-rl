"""Launch exactly one bounded pilot controller after explicit authorization."""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time

from support import ROOT, HERE, atomic, read, sha, verify_release


def main():
    assert sys.argv[1:] == ['--authorized-pilot']
    verify_release()
    protocol = read(HERE / 'protocol.json')
    assert protocol['stage_targets'] == [2, 10, 20, 40]
    audit = ROOT / 'audits/PURE_OpenRLHF_pilot_20261023'
    audit.mkdir(parents=True, exist_ok=True)
    gate = read(audit / 'preflight.json')
    assert gate['passed'] is True and gate['source_files'], 'Pilot CPU/source preflight is required'
    for name, expected in gate['source_files'].items():
        assert sha(ROOT / name) == expected, ('Stale pilot preflight', name)
    lock = (audit / 'dispatch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = audit / 'dispatch.json'
    directory = ROOT / protocol['controller']
    assert not receipt.exists() and not directory.exists(), 'Existing controller/receipt: inspect instead of duplicate launch'
    assert all(not (ROOT / relative).exists() for relative in protocol['runs'].values()), 'Existing arm state; do not restart it'
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    active = [line for line in processes.splitlines()
              if any(str(HERE / script) in line for script in ('controller.py', 'train_arm.py', 'evaluate.py')) and 'python' in line]
    assert not active, active
    # The previous real-model no-update integration is an admission gate, not a
    # pilot training result; the controller is still capped at 40 updates/arm.
    no_update = ROOT / 'runs/PURE_OpenRLHF_no_update_20261023_v1'
    preflight = read(no_update / 'result.json')
    assert preflight['passed'] and preflight['optimizer_steps'] == 0
    assert preflight['actor_and_prm_parameters_unchanged'] and preflight['original_AB_and_verl_preserved']
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy', 'pip_proxy'}:
            env.pop(key)
    headers = read(ROOT / 'receipts/python_dev_headers.json')
    env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=str(protocol['gpu']),
        PURE_PHYSICAL_GPU=str(protocol['gpu']), HF_HOME=str(ROOT / 'cache/huggingface'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_XET='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        VLLM_CACHE_ROOT=str(ROOT / 'cache/vllm'), TRITON_CACHE_DIR=str(ROOT / 'cache/triton'),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache/torchinductor'), VLLM_USE_V1='0',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1', MASTER_ADDR='127.0.0.1',
        MASTER_PORT='29647', DO_NOT_TRACK='1', WANDB_MODE='disabled',
        C_INCLUDE_PATH=os.pathsep.join(headers['include_paths']))
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(HERE / 'controller.py'), '--authorized-pilot']
    log = audit / 'controller.log'
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    atomic(receipt, dict(pid=child.pid, controller=str(directory), command=command, log=str(log),
        launched_unix=time.time(), release_sha256=sha(HERE / 'release.json'),
        pilot_preflight_sha256=sha(audit / 'preflight.json'),
        no_update_preflight_sha256=sha(no_update / 'result.json'),
        stages=[2, 10, 20, 40], max_updates_per_arm=40,
        max_generated_completion_tokens_per_arm=2621440, formal_6M_authorized=False,
        shared_gpu_authorized=True, next_action='controller waits for sufficient free memory before each worker'))
    print(receipt.read_text(), flush=True)


if __name__ == '__main__':
    main()
