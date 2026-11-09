"""Independent, fixed-budget continuation. Importing does not start work."""
from pathlib import Path
import hashlib, json, os, time, importlib.util
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
os.environ['VLLM_USE_V1']='1'

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
VALIDATED=ROOT/'experiments/prefix_protection_20261107'
RUN=ROOT/'runs/Tree_credit_triplet_seed17_48_20261108'
AUDIT=ROOT/'audits/Tree_credit_triplet_seed17_48_20261108'
CP=ROOT/'runs/ABC_seed17_resumed_V1_20261104/C_bounded/checkpoint-288'
ARMS=('chain_cb','tree_vr','tree_cb')
START,TARGET=288,336
GPU_UUID='GPU-9fa4fe23-1ce5-d116-daab-3271dcbaa0ac'

spec=importlib.util.spec_from_file_location('frozen_resume_utilities',ROOT/'experiments/abc_seed17_resumed_v1_20261104/pilot_common.py')
util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
read,rows,sha,atomic,append,adapter_digest,tree_digest=util.read,util.rows,util.sha,util.atomic,util.append,util.adapter_digest,util.tree_digest

def verify():
    release=read(HERE/'release.json')
    for name,h in release['files'].items():assert sha(HERE/name)==h,('Changed experiment source',name)
    for name,h in release['dependencies'].items():assert sha(ROOT/name)==h,('Changed dependency',name)
    return sha(HERE/'release.json')

def source_unchanged():
    for name,h in read(CP/'pilot_checkpoint.json')['files_sha256'].items():assert sha(CP/name)==h,('Changed source checkpoint',name)
    return True

def guard():
    import subprocess
    assert ROOT.resolve().is_relative_to(Path('/path/to/project_parent'))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('VLLM_USE_V1')=='1'
    assert subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==GPU_UUID

def event(stage,**values):atomic(RUN/'controller.json',{'stage':stage,'pid':os.getpid(),'unix':time.time(),**values})
