"""Read-only remote status collection; no training/model imports or remote writes."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess

SSH = ['ssh', '-p', '11032', '-i', '/home/USER/.ssh/fym_125_76_110_84_ed25519',
       '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ConnectTimeout=12', 'USER@GPU_HOST',
       '/path/to/ece285_math_rl_project/.venv/bin/python -']
REMOTE = r'''
import json,pathlib,subprocess,time
root=pathlib.Path('/path/to/ece285_math_rl_project')
out={'observed_unix':time.time(),'files':{},'missing':[]}
names=['runs/AB_corrected_seed17_6M_20261022/'+x for x in
       ['controller.json','comparison.json','pilot_gate.json','stages.jsonl']]
for arm in ['A','B']:
 names+=['runs/'+arm+'_corrected_seed17_6M_20261022/'+x for x in
         ['status.json','progress.json','failure.json','live_audits.json',
          'evaluation/evaluation_config.json','evaluation/progress.json','evaluation/evaluation_summary.json']]
for name in names:
 p=root/name
 if not p.exists():out['missing'].append(name);continue
 data=p.read_text()
 out['files'][name]=[json.loads(x) for x in data.splitlines() if x.strip()] if name.endswith('.jsonl') else json.loads(data)
controller=out['files'].get('runs/AB_corrected_seed17_6M_20261022/controller.json',{})
ids=[str(controller[k]) for k in ['pid','child_pid'] if controller.get(k)]
if ids:out['processes']=subprocess.run(['ps','-p',','.join(ids),'-o','pid,ppid,etime,stat,args'],capture_output=True,text=True).stdout
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
log=controller.get('log')
if log and pathlib.Path(log).exists():
 p=pathlib.Path(log)
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));out['recent_log']=f.read().decode('utf8','replace')
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
'''

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'snapshots')
    args=p.parse_args()
    result=subprocess.run(SSH,input=REMOTE,text=True,capture_output=True,timeout=90)
    if result.returncode:
        raise RuntimeError('Read-only SSH snapshot failed; this does not prove remote job failure: '+result.stderr[-2000:])
    snapshot=json.loads(result.stdout)
    snapshot['client_observed_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/(stamp+'.json')
    with output.open('x') as f:json.dump(snapshot,f,ensure_ascii=False,indent=2,allow_nan=False)
    brief={'snapshot':str(output),'controller':snapshot['files'].get('runs/AB_corrected_seed17_6M_20261022/controller.json')}
    for arm in ['A','B']:
        prefix='runs/'+arm+'_corrected_seed17_6M_20261022/'
        brief[arm]={k:snapshot['files'].get(prefix+k+'.json') for k in ['status','progress']}
    print(json.dumps(brief,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
