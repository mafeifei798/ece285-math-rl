"""User-authorized shared-GPU admission for the existing frozen inference run."""
from pathlib import Path
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'runs/AB_prompt_validation_20261022'
sys.path.insert(0, str(ROOT / 'training'))
from common import atomic_json, file_hash


def gpu_snapshot():
    text = subprocess.check_output([
        'nvidia-smi', '-i', '1',
        '--query-gpu=memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    free, used, util = map(int, text.split(','))
    return {'free_MiB': free, 'used_MiB': used, 'utilization_percent': util,
            'admitted': free >= 40000}


def main():
    lock = (OUT / 'shared_dispatch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'shared_dispatch.json').exists(), 'Inspect existing shared dispatch; no duplicate'
    if any((OUT / f).exists() for f in ['controller.json', 'state.json', 'predictions.jsonl']):
        raise RuntimeError('Existing inference evidence: inspect without relaunching')
    old = json.loads((OUT / 'dispatch.json').read_text())
    assert old['status'] == 'waiting_for_gpu' and old.get('child_pid') is None
    pid = old['pid']
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    expected = str(HERE / 'dispatch_when_idle.py').encode()
    assert expected in cmd, 'Dispatcher identity changed'
    children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    assert not children, 'Dispatcher has a child; inspect instead of terminating'
    assert file_hash(HERE / 'dispatch_when_idle.py') == old['script_sha256']
    with (OUT / 'dispatch_before_shared.json').open('x') as f:
        json.dump(old, f, indent=2)
    auth = {
        'user_answer': '允许共享GPU立即评测',
        'recorded_unix': time.time(), 'gpu': 1,
        'scope': 'Existing frozen four-cell inference only; no training, no protocol or grading changes',
        'admission_change': 'Allow other GPU processes and GPU utilization; retain evaluator free>=40000 MiB gate',
        'frozen_release_sha256': file_hash(OUT / 'release.json')}
    with (OUT / 'shared_gpu_authorization.json').open('x') as f:
        json.dump(auth, f, ensure_ascii=False, indent=2)
    # This exact process is our idle dispatcher; never signal its process group or any GPU job.
    os.kill(pid, signal.SIGTERM)
    old_lock = (OUT / 'dispatch.lock').open('a')
    for _ in range(100):
        try:
            fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(.1)
    else:
        raise RuntimeError('Old dispatcher did not release its lock; no launch')
    assert not any((OUT / f).exists() for f in ['controller.json', 'state.json', 'predictions.jsonl']), 'Old dispatcher raced to launch; preserve existing run'
    old.update(status='superseded_by_user_authorized_shared_dispatch', superseded_unix=time.time(),
               replacement_script=str(Path(__file__)), original_receipt='dispatch_before_shared.json')
    atomic_json(OUT / 'dispatch.json', old)
    state = {'status': 'waiting_for_free_memory', 'pid': os.getpid(), 'child_pid': None,
             'started_unix': time.time(), 'heartbeat_unix': time.time(), 'training': False,
             'script_sha256': file_hash(Path(__file__)),
             'release_sha256': auth['frozen_release_sha256'],
             'minimum_free_MiB': 40000, 'poll_seconds': 30, 'maximum_wait_seconds': 43200}

    def record(**kw):
        state.update(kw, heartbeat_unix=time.time())
        atomic_json(OUT / 'shared_dispatch.json', state)

    record()
    try:
        while True:
            gpu = gpu_snapshot()
            record(gpu_check=gpu)
            if gpu['admitted']:
                break
            if time.time() - state['started_unix'] > state['maximum_wait_seconds']:
                record(status='memory_wait_expired', finished_unix=time.time())
                return
            time.sleep(30)
        release = json.loads((OUT / 'release.json').read_text())
        assert file_hash(OUT / 'release.json') == state['release_sha256']
        for rel, sha in release['input_sha256'].items():
            assert file_hash(ROOT / rel) == sha, rel
        with (OUT / 'shared_launch.log').open('x') as log:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-B', str(HERE / 'launch.py')],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            record(status='running', child_pid=child.pid)
            while child.poll() is None:
                time.sleep(30)
                record()
            record(child_pid=None, child_exit_code=child.returncode, finished_unix=time.time())
            assert child.returncode == 0, 'Existing controller failed; preserve evidence, no automatic retry'
        record(status='completed')
    except BaseException as exc:
        record(status='failed', error=str(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        raise


if __name__ == '__main__':
    main()
