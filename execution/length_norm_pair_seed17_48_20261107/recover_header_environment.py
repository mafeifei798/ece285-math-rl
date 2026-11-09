"""Preserve a verified zero-update failure and relaunch with the old CPATH."""
from prepare_remote import *


def main():
    files={p.name:p.read_text() for p in (ROOT/'experiments/length_norm_header_recovery_20261107').iterdir() if p.is_file()}
    code=r'''
from pathlib import Path
import json,os,time,hashlib,subprocess,psutil,fcntl
r=Path('/path/to/ece285_math_rl_project');run=r/'runs/Length_norm_pair_seed17_48_20261107';audit=r/'audits/Length_norm_pair_seed17_48_20261107';src=r/'experiments/length_norm_header_recovery_20261107'
lock=(run/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not src.exists() and not (run/'recovery_launch_receipt.json').exists()
receipt=json.loads((run/'launch_receipt.json').read_text());pid=receipt['pid']
assert not (psutil.pid_exists(pid) and abs(psutil.Process(pid).create_time()-receipt['create_time'])<.1 and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE)
assert json.loads((run/'controller.json').read_text())['stage']=='failed'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for arm in ['grpo','fixed_length']:
 p=run/arm;assert not (p/'intent.json').exists() and not list(p.glob('raw_step_*.pt')) and not list(p.glob('checkpoint-*'))
assert not list((audit/'no_update').glob('raw_step_*.pt')) and not (audit/'no_update/rollouts.jsonl').exists()
header=json.loads((r/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json').read_text());assert header['passed']
assert all(Path(p).exists() for p in header['CPATH'].split(':'))
assert any((Path(p)/'Python.h').exists() for p in header['CPATH'].split(':'))
files=FILES;src.mkdir()
for n,s in files.items():assert Path(n).name==n;(src/n).write_text(s)
for n in ['controller.json','children.json']:
 p=run/n
 if p.exists():p.rename(run/(p.stem+'_initial_header_failure'+p.suffix))
(audit/'no_update').rename(audit/'no_update_initial_header_failure')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',CPATH=header['CPATH'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
cmd=[str(r/'.venv/bin/python'),'-B','-u',str(src/'controller.py')]
with (run/'header_recovery_controller.log').open('xb') as log:
 child=subprocess.Popen(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
out={'pid':child.pid,'create_time':psutil.Process(child.pid).create_time(),'unix':time.time(),'command':cmd,
     'change':'Add existing verified project-local CPATH only','prior_training_updates':0,'prior_emitted_training_batches':0,
     'initial_release_sha256':receipt['release_sha256'],'recovery_controller_sha256':hashlib.sha256((src/'controller.py').read_bytes()).hexdigest(),
     'original_failure_preserved':True,'same48_updates_per_arm':True,'same_model_reward_loss_seed_budget':True}
(run/'recovery_launch_receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''.replace('FILES',repr(files))
    r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=code,text=True,capture_output=True,timeout=45)
    if r.returncode:raise RuntimeError(r.stderr[-5000:])
    x=json.loads(r.stdout);(HERE/'recovery_launch_receipt.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))


if __name__=='__main__':main()
