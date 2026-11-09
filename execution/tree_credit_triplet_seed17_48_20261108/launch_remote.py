"""Launch this authorized finite release once; fail closed on existing state."""
from pathlib import Path
import runpy,subprocess,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
CODE=r"""
from pathlib import Path
import json,subprocess,os,time,psutil,hashlib
root=Path('/path/to/ece285_math_rl_project');source=root/'experiments/tree_credit_triplet_seed17_48_20261108'
run=root/'runs/Tree_credit_triplet_seed17_48_20261108';audit=root/'audits/Tree_credit_triplet_seed17_48_20261108'
assert not (run/'launch_receipt.json').exists() and not (run/'dispatch_started.json').exists(),'Inspect existing job; do not duplicate'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied; do not interrupt'
release=json.loads((source/'release.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in release['files'].items():assert sha(source/n)==h,n
for n,h in release['dependencies'].items():assert sha(root/n)==h,n
header=json.loads((root/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json').read_text());assert header['passed']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
 CPATH=header['CPATH'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',WANDB_DISABLED='true',
 TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
 HF_HOME=str(root/'cache/huggingface'),HF_DATASETS_CACHE=str(root/'cache/huggingface/datasets'),
 TRITON_CACHE_DIR=str(root/'cache/pilot_v2/triton'),TORCHINDUCTOR_CACHE_DIR=str(root/'cache/pilot_v2/torchinductor'),
 VLLM_CACHE_ROOT=str(root/'cache/pilot_v2/vllm'),TORCH_HOME=str(root/'cache/torch'))
for x in header['CPATH'].split(':'):assert Path(x).is_dir()
cmd=[str(root/'.venv/bin/python'),'-B','-u',str(source/'controller.py')]
with (run/'controller.log').open('xb') as f:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'create_time':psutil.Process(p.pid).create_time(),'unix':time.time(),'command':cmd,
 'release_sha256':sha(source/'release.json'),'authorized_arms':['chain_cb','tree_vr','tree_cb'],'updates_per_arm':48,'seed':17,
 'zero_update_gate_required':True,'automatic_expansion':False,'header_receipt_sha256':sha(root/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json')}
(run/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
"""
if __name__=='__main__':
 r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=CODE,text=True,capture_output=True,timeout=55)
 if r.returncode:raise RuntimeError(r.stderr[-5000:])
 x=json.loads(r.stdout);(HERE/'launch_receipt.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
