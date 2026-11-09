"""Read-only remote receipt/process inventory; never dispatches work."""
from pathlib import Path
import json, subprocess
HERE=Path(__file__).resolve().parent
SSH=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','USER@GPU_HOST']
REMOTE=r'''
from pathlib import Path
import os,json,subprocess,time
r=Path('/path/to/ece285_math_rl_project')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (ValueError,TypeError,ProcessLookupError):return False
run=r/'runs/A6M_protocol_recheck_20261029'
out={'unix':time.time(),'new_attempt_exists':run.exists(),'new_attempt':{n:read(run/(n+'.json')) for n in ['dispatch','controller','progress','generation_complete','acceptance','comparison','controller_failure','generation_failure']},
 'previous_controller':read(r/'runs/Final_four_eval_seed17_1M_20261029/controller.json'),
 'historical_A_receipt':read(r/'runs/A_corrected_seed17_6M_20261022/final_result.json'),
 'historical_eval':read(r/'runs/A_corrected_seed17_6M_20261022/evaluation/evaluation_summary.json')}
out['old_checkpoint_files']=[str(p.relative_to(r)) for p in (r/'runs/A_corrected_seed17_6M_20261022/checkpoint-284').glob('*') if p.is_file()]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
out['compute_apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
out['processes']=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,comm'],text=True)
out['instructions']={str(p):p.read_text() for p in [Path('/AGENTS.md'),Path('/yimin/AGENTS.md'),Path('/yimin/lzx/AGENTS.md'),Path('/path/to/project_parent/AGENTS.md'),r/'AGENTS.md',r/'experiments/AGENTS.md',r/'runs/AGENTS.md',r/'audits/AGENTS.md'] if p.exists()}
for k in ['controller','dispatch']:
 d=out['new_attempt'].get(k) or {};out[k+'_alive']=alive(d.get('pid'))
log=run/'generation.log';out['log_tail']=log.read_text(errors='replace')[-1500:] if log.exists() else ''
print(json.dumps(out))
'''
if __name__=='__main__':
 s=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=REMOTE,text=True,capture_output=True,check=True)
 d=json.loads(s.stdout);(HERE/'latest_snapshot.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:d[k] for k in ['new_attempt_exists','previous_controller','old_checkpoint_files','gpu','compute_apps','instructions','controller_alive','dispatch_alive']},ensure_ascii=False,indent=2))
 print(json.dumps({'current':{k:v for k,v in d['new_attempt'].items() if k not in ['comparison','generation_complete']},'log_tail':d['log_tail']},ensure_ascii=False,indent=2))
