"""Read-only snapshot of the existing, bounded pilot. Never launches work."""
import datetime,json,pathlib,subprocess

HERE=pathlib.Path(__file__).resolve().parent
REMOTE=r'''
import json,pathlib,subprocess,time
r=pathlib.Path('/path/to/ece285_math_rl_project')
def load(p):
 return json.loads(p.read_text()) if p.exists() else None
c=r/'runs/PURE_OpenRLHF_pair_pilot_seed17_20261023'
result=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),controller=load(c/'controller.json'),arms={})
for arm,label in [('P_VR','PVR'),('P_PRM','PPRM')]:
 d=r/f'runs/PURE_OpenRLHF_{label}_pilot_seed17_20261023'
 a=dict(status=load(d/'status.json'),latest=load(d/'latest.json'),stages={},validation={},diagnostics=[],failures={})
 for p in d.glob('stage_*.json'):a['stages'][p.name]=load(p)
 for p in d.glob('validation_*.json'):a['validation'][p.name]={k:v for k,v in load(p).items() if k!='rows'}
 if (d/'diagnostics.jsonl').exists():a['diagnostics']=[json.loads(x) for x in (d/'diagnostics.jsonl').read_text().splitlines()]
 for p in d.glob('runtime*/failure.json'):a['failures'][str(p.relative_to(d))]=load(p)
 a['evaluation']={p.parent.name:load(p) for p in d.glob('evaluation*/evaluation_summary.json')}
 result['arms'][arm]=a
result['gates']={p.name:load(p) for p in c.glob('gate_*.json')}
result['comparison']=load(c/'comparison.json')
result['logs']={p.name:p.read_text()[-3000:] for p in c.glob('*.log')}
pids=[str(result['controller']['pid'])]
child=result['controller'].get('active_child')
if child:pids.append(str(child['pid']))
result['processes']=subprocess.run(['ps','-p',','.join(pids),'-o','pid,ppid,stat,etime,time,%cpu,args'],capture_output=True,text=True).stdout
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
result['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(result,ensure_ascii=False))
'''
command=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','USER@GPU_HOST','/path/to/ece285_math_rl_project/.venv/bin/python -']
data=json.loads(subprocess.check_output(command,input=REMOTE,text=True,timeout=60))
stamp=data['utc'].replace(':','').replace('-','')
destination=HERE/'snapshots'/f'{stamp}.json';destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(HERE/'latest_snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(saved=str(destination),utc=data['utc'],status=data['controller']['status'],active_child=data['controller'].get('active_child'),arms={k:dict(status=v['status'],validation=v['validation'],diagnostics_count=len(v['diagnostics']),failures=v['failures']) for k,v in data['arms'].items()}),ensure_ascii=False,indent=2))
