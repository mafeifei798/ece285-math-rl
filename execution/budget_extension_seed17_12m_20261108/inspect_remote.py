from pathlib import Path
import json,runpy,subprocess,html
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']
CODE=r'''
from pathlib import Path
import json,time,psutil,subprocess
r=Path('/path/to/ece285_math_rl_project');run=r/'runs/Budget_extension_seed17_12M_20261108';audit=r/'audits/Budget_extension_seed17_12M_20261108'
def read(p):return json.loads(p.read_text()) if p.exists() else None
c=read(run/'controller.json');launch=read(run/'launch_receipt.json');alive=False
if launch:
 try:
  proc=psutil.Process(launch['pid']);alive=abs(proc.create_time()-launch['create_time'])<.1 and proc.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:pass
x={'unix':time.time(),'controller':c,'launch':launch,'controller_alive':alive,'preflight':read(audit/'preflight.json') is not None,'arms':{},'evaluations':{}}
for arm in ['A','C_bounded']:
 p=run/arm;v={'progress':read(p/'progress.json'),'smoke':read(audit/'no_update'/arm/'result.json'),'smoke_build':(audit/'no_update'/arm/'build_smoke.json').exists()}
 for step in [432,576]:
  v['first_update_'+str(step)]=read(p/f'first_update_acceptance_{step}.json');v['result_'+str(step)]=read(p/f'stage_{step}_result.json');v['failure_'+str(step)]=read(p/f'failure_{step}.json')
 x['arms'][arm]=v
for cohort in ['baseline','nine','final','endpoints_math','endpoints_gsm']:
 x['evaluations'][cohort]={}
 for repeat in [0,1]:
  p=run/'evaluation'/cohort/f'repeat_{repeat}'
  if p.exists():x['evaluations'][cohort][str(repeat)]={'progress':read(p/'progress.json'),'summary':read(p/'summary.json')}
x['seed42_eligibility']=read(run/'seed42_eligibility.json')
x['comparison_passed']=(read(run/'comparison.json') or {}).get('passed')
x['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
if c and (c.get('stage')=='failed' or not alive):
 x['controller_log_tail']=(run/'controller.log').read_text()[-5000:]
 active=(c.get('children') or [])[-1:]
 if active:
  p=run/(active[0]['stage']+'.log')
  if p.exists():x['last_child_log_tail']=p.read_text()[-6500:]
print(json.dumps(x))
'''
def snapshot():
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=CODE,text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr[-4000:])
    x=json.loads(p.stdout);(HERE/'status.json').write_text(json.dumps(x,indent=2)+'\n');return x
def compact(x):
    return {'stage':(x['controller'] or {}).get('stage'),'alive':x['controller_alive'],'preflight':x['preflight'],
        'arms':{a:{'step':(v['progress'] or {}).get('global_step'),'smoke_passed':bool(v['smoke'] and v['smoke']['passed']),
          'first_update432':bool(v['first_update_432']),'complete432':bool(v['result_432']),'first_update576':bool(v['first_update_576']),'complete576':bool(v['result_576'])} for a,v in x['arms'].items()},
        'evaluation':{c:{k:({'complete':True,'accuracy':{a:b['accuracy'] for a,b in v['summary']['summary'].items()}} if v['summary'] else v['progress']) for k,v in reps.items()} for c,reps in x['evaluations'].items() if reps},
        'comparison_passed':x['comparison_passed'],'GPU':x['gpu']}
def render(x):
    data=json.dumps(compact(x),ensure_ascii=False,indent=2)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>A/C 6M→12M 固定预算续训</title><style>body{font:16px/1.8 -apple-system,"PingFang SC",sans-serif;background:#f3f6fa;color:#20334a}main{max-width:1000px;background:white;padding:32px;margin:30px auto;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:20px}a{color:#2862a5}</style><main><h1>A/C-bounded · 6M→12M 固定预算续训</h1><p>seed17；原始checkpoint288→432→576；奖励、loss、验答规则不变。固定格式监控与重复验证。旧结果保留。</p><p>这是带时间戳的实际状态快照；启动不等于已完成训练。</p><pre>'+html.escape(data)+'</pre><p><a href="status.json">真实状态JSON</a> · <a href="CONTINUATION.txt">执行范围</a> · <a href="diagnostic_status.json">冻结PRM格式对照</a></p></main></html>'
    (HERE/'review.html').write_text(page)
if __name__=='__main__':
    x=snapshot();render(x);print(json.dumps(compact(x),ensure_ascii=False,indent=2))
