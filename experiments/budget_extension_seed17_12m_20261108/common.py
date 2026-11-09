from pathlib import Path
import hashlib,json,os,time,importlib.util,sys
ROOT=Path('/path/to/ece285_math_rl_project')
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/Budget_extension_seed17_12M_20261108'
AUDIT=ROOT/'audits/Budget_extension_seed17_12M_20261108'
OLD=ROOT/'experiments/abc_seed17_resumed_v1_20261104'
SOURCE=ROOT/'runs/ABC_seed17_resumed_V1_20261104'
ARMS=('A','C_bounded')
spec=importlib.util.spec_from_file_location('budget_original_utilities',OLD/'pilot_common.py')
util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
read,rows,sha,atomic,append,adapter_digest,tree_digest=util.read,util.rows,util.sha,util.atomic,util.append,util.adapter_digest,util.tree_digest
def verify():
    r=read(HERE/'release.json')
    for n,h in r['files'].items():assert sha(HERE/n)==h,('Changed budget source',n)
    for n,h in r['dependencies'].items():assert sha(ROOT/n)==h,('Changed frozen dependency',n)
    return sha(HERE/'release.json')
def guard():
    import subprocess
    assert ROOT.resolve().is_relative_to(Path('/path/to/project_parent'))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==read(HERE/'protocol.json')['GPU_UUID']
def source_unchanged():
    for arm,files in read(AUDIT/'source_checkpoints.json').items():
        for name,h in files.items():assert sha(SOURCE/arm/'checkpoint-288'/name)==h,(arm,name)
def protected_unchanged():
    for name,h in read(AUDIT/'protected_before.json').items():assert sha(ROOT/name)==h,('Historical file changed',name)
def event(stage,**kw):atomic(RUN/'controller.json',{'stage':stage,'pid':os.getpid(),'unix':time.time(),**kw})
