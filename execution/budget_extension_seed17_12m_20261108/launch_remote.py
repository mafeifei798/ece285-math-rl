from pathlib import Path
import json,runpy,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
CODE=r'''
from pathlib import Path
import json,hashlib,os,subprocess,psutil,time
r=Path('/path/to/ece285_math_rl_project');s=r/'experiments/budget_extension_seed17_12m_20261108';run=r/'runs/Budget_extension_seed17_12M_20261108';audit=r/'audits/Budget_extension_seed17_12M_20261108'
assert not (run/'launch_receipt.json').exists() and not (run/'dispatch_started.json').exists(),'Existing launch: inspect'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
release=json.loads((s/'release.json').read_text())
for n,h in release['files'].items():assert sha(s/n)==h
for n,h in release['dependencies'].items():assert sha(r/n)==h
header=json.loads((r/'audits/C_bounded_triplet_seed17_1M_20261029_v2/python_header_repair.json').read_text());assert header['passed']
for p in header['CPATH'].split(':'):assert Path(p).is_dir()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
 CPATH=header['CPATH'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',WANDB_DISABLED='true',
 TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
 HF_HOME=str(r/'cache/huggingface'),HF_DATASETS_CACHE=str(r/'cache/huggingface/datasets'),
 TRITON_CACHE_DIR=str(r/'cache/pilot_v2/triton'),TORCHINDUCTOR_CACHE_DIR=str(r/'cache/pilot_v2/torchinductor'),
 VLLM_CACHE_ROOT=str(r/'cache/pilot_v2/vllm'),TORCH_HOME=str(r/'cache/torch'))
cmd=[str(r/'.venv/bin/python'),'-B','-u',str(s/'controller.py')]
with (run/'controller.log').open('xb') as log:p=subprocess.Popen(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
x={'pid':p.pid,'create_time':psutil.Process(p.pid).create_time(),'unix':time.time(),'command':cmd,'release_sha256':sha(s/'release.json'),
 'seed':17,'arms':['A','C_bounded'],'source_step':288,'final_step':576,'zero_update_gate':True,'automatic_seed42_dispatch':False}
(run/'launch_receipt.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
'''
def main():
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=CODE,text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr[-6000:])
    x=json.loads(p.stdout);(HERE/'launch_receipt.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
if __name__=='__main__':main()
