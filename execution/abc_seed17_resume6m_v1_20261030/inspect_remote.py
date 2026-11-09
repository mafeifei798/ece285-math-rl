"""Read-only inventory; never launches or changes remote jobs."""
from pathlib import Path
import subprocess,json
HERE=Path(__file__).resolve().parent
SSH=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','USER@GPU_HOST']
REMOTE=r'''
from pathlib import Path
import os,json,subprocess,time
r=Path('/path/to/ece285_math_rl_project');p=r/'runs/ABC_seed17_resume6M_V1_20261030';old=r/'runs/C_bounded_triplet_seed17_1M_20261029_v2'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (ValueError,TypeError,ProcessLookupError):return False
d={'unix':time.time(),'attempt_exists':p.exists(),'controller':read(p/'controller.json'),'dispatch':read(p/'dispatch.json'),'failure':read(p/'controller_failure.json'),
 'old_controller':read(old/'controller.json'),'arms':{},'instructions':{}}
for a in ['A','C_mix','C_bounded']:
 d['arms'][a]={'progress':read(p/a/'progress.json'),'stage_result50':read(p/a/'stage_50_result.json'),'stage_result144':read(p/a/'stage_144_result.json'),'stage_result288':read(p/a/'stage_288_result.json'),
   'last_checkpoint':read(p/a/'last_checkpoint.json'),'old_final':read(old/a/'final_result.json'),'old_checkpoint':read(old/a/'last_checkpoint.json'),'failure':read(p/a/'failure.json')}
d['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
d['apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
d['disk']=subprocess.check_output(['df','-h',str(r)],text=True)
for x in [Path('/AGENTS.md'),Path('/yimin/AGENTS.md'),Path('/yimin/lzx/AGENTS.md'),Path('/path/to/project_parent/AGENTS.md'),r/'AGENTS.md',r/'experiments/AGENTS.md',r/'runs/AGENTS.md',r/'audits/AGENTS.md']:
 if x.exists():d['instructions'][str(x)]=x.read_text()
d['controller_alive']=alive((d['dispatch'] or {}).get('pid'));d['child_alive']=alive((d['controller'] or {}).get('child_pid'))
cp=Path((d['controller'] or {}).get('log',str(p/'controller.log')));d['log_tail']=cp.read_text(errors='replace')[-2200:] if cp.exists() else ''
print(json.dumps(d))
'''
if __name__=='__main__':
 s=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=REMOTE,text=True,capture_output=True,check=True)
 d=json.loads(s.stdout);(HERE/'latest_snapshot.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'attempt_exists':d['attempt_exists'],'controller':d['controller'],'controller_alive':d['controller_alive'],'child_alive':d['child_alive'],'gpu':d['gpu'],'apps':d['apps'],'disk':d['disk'],'instructions':d['instructions'],'arms':{a:{'old_final_step':v['old_final']['global_step'] if v['old_final'] else None,'old_checkpoint':v['old_checkpoint'],'progress':v['progress'],'failure':v['failure']} for a,v in d['arms'].items()},'failure':d['failure'],'log_tail':d['log_tail']},ensure_ascii=False,indent=2))
