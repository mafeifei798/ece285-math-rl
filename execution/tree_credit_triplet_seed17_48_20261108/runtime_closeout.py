"""Final read-only PID/GPU and optional-importance-audit coverage check."""
from pathlib import Path
import json,subprocess,runpy

HERE=Path(__file__).resolve().parent
SSH=runpy.run_path(str(HERE.parents[1]/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
CODE=r'''
from pathlib import Path
import json,subprocess,hashlib,os,time
import psutil,torch
r=Path('/path/to/ece285_math_rl_project')
p=r/'runs/Tree_credit_triplet_seed17_48_20261108'
c=json.loads((p/'controller.json').read_text());assert c['stage']=='complete'
assert not torch.cuda.is_initialized()
hashes=json.loads((r/'audits/Tree_credit_triplet_seed17_48_20261108/independent_tensor_acceptance.json').read_text())['raw_tensor_sha256']
arms={}
for arm in ('chain_cb','tree_vr','tree_cb'):
 settings=json.loads((p/arm/'build.json').read_text())['settings']
 assert settings['vllm_importance_sampling_correction'] is False
 assert settings['num_iterations']==1 and settings['steps_per_generation']==settings['gradient_accumulation_steps']==32
 counts={'batches':0,'old_logps_absent':0,'sampling_ratio_absent':0}
 for step in range(288,336):
  path=p/arm/f'raw_step_{step}.pt'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[str(path.relative_to(r))]
  x=torch.load(path,map_location='cpu',weights_only=False)
  counts['batches']+=1
  counts['old_logps_absent']+=x.get('old_per_token_logps') is None
  counts['sampling_ratio_absent']+=x.get('importance_sampling_ratio') is None
 assert counts['old_logps_absent']==counts['sampling_ratio_absent']==48
 arms[arm]={'vllm_sampling_correction_enabled':False,**counts,
  'sampling_ratio_reconstructions_applicable':0,
  'note':'Optional ratio branch in the prior independent verifier was not applicable; its max-error0 is not evidence of reconstructed sampling ratios. Current single-iteration loss uses detached current log-probabilities when cached old log-probabilities are absent.'}
alive=[]
for pid in [c['pid']]+[x['pid'] for x in c.get('children',[])]:
 try:
  proc=psutil.Process(pid)
  if proc.status()!=psutil.STATUS_ZOMBIE and any('tree_credit_triplet_seed17_48_20261108' in x for x in proc.cmdline()):alive.append(pid)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
assert not alive
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
assert not torch.cuda.is_initialized()
print(json.dumps({'passed':True,'unix':time.time(),'controller_stage':'complete',
 'our_live_controller_or_children':alive,'GPU_apps_after_closeout':gpu,
 'CUDA_initialized_by_check':False,'sampling_correction_coverage':arms}))
'''

def main():
    assert json.loads((HERE/'completion_verification.json').read_text())['passed']
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=CODE,
        text=True,capture_output=True,timeout=90)
    if p.returncode:raise RuntimeError(p.stderr[-5000:])
    result=json.loads(p.stdout)
    (HERE/'runtime_closeout.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
