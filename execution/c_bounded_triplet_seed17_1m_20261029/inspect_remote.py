"""Read-only finite-pipeline snapshot. Does not launch, stop, or retry jobs."""
from pathlib import Path
import subprocess,json,html,datetime

HERE=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import json,subprocess,time,os
r=Path('/path/to/ece285_math_rl_project');p=r/'runs/C_bounded_triplet_seed17_1M_20261029_v2';a=r/'audits/C_bounded_triplet_seed17_1M_20261029_v2'
def load(f):return json.loads(f.read_text()) if f.exists() else None
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (ProcessLookupError,TypeError,ValueError):return False
out=dict(unix=time.time(),dispatch=load(a/'dispatch.json'),controller=load(p/'controller.json'),child=load(p/'child.json'),
 smoke_gate=load(p/'smoke_gate.json'),failure=((load(p/'header_recovery_failure.json') or load(p/'environment_recovery_failure.json') or load(p/'controller_failure.json')) if (load(p/'controller.json') or {}).get('stage')=='failed_stopped' else None),resource=load(p/'resource_snapshot.json'),
 comparison=load(p/'comparison.json'),arms={},validation={})
out['controller_alive']=alive((out['dispatch'] or {}).get('pid'))
out['child_alive']=alive((out['child'] or {}).get('pid'))
for arm in ['A','C_mix','C_bounded']:
 q=p/arm
 out['arms'][arm]={n:load(q/(n+'.json')) for n in ['status','progress','smoke_result','resume_loaded','resume_acceptance','final_result','failure','last_checkpoint']}
for arm in ['initial','A','C_mix','C_bounded']:
 q=p/'validation'/arm;out['validation'][arm]={n:load(q/(n+'.json')) for n in ['progress','summary']}
child=out['child']
if child:
 f=Path(child['log']);out['active_log_tail']=f.read_text(errors='replace')[-6500:] if f.exists() else None
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
'''

def main():
    cmd=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        'USER@GPU_HOST','/path/to/ece285_math_rl_project/.venv/bin/python -B -']
    r=subprocess.run(cmd,input=REMOTE,text=True,capture_output=True,check=True)
    d=json.loads(r.stdout);(HERE/'latest_snapshot.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    if d['dispatch']:(HERE/'dispatch.json').write_text(json.dumps(d['dispatch'],ensure_ascii=False,indent=2)+'\n')
    stage=(d['controller'] or {}).get('stage','unknown')
    status=dict(status=stage,controller_alive=d['controller_alive'],child_alive=d['child_alive'],unix=d['unix'],
        controller_pid=(d['dispatch'] or {}).get('pid'),arms={a:{k:v for k,v in (d['arms'][a]['progress'] or {}).items()
        if k in ['phase','global_step','generated_completion_tokens','loss_wrong_positive','process_nonzero_rows','peak_reserved_GiB']} for a in d['arms']},
        original_ABCD_preserved=(d.get('smoke_gate') or {}).get('passed', 'verified_at_CPU_preflight'),D_adaptation=False,per_arm_budget=1000000,seed=17)
    (HERE/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    cells=[]
    for a,v in d['arms'].items():
        q=v['progress'] or {};s=v['status'] or {}
        cells.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [a,s.get('status','等待'),q.get('global_step',0),q.get('generated_completion_tokens',0),q.get('loss_wrong_positive','—'),q.get('peak_reserved_GiB','—')])+'</tr>')
    heading='三组小试 · '+stage
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三组小试实时验收</title><style>body{background:#f4f6f8;color:#17232e;font:16px/1.8 system-ui;margin:0}main{max-width:1050px;margin:40px auto;background:white;padding:36px;border-radius:16px}h1{font-size:28px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:16px}a{color:#235fb0}</style><main><h1>'''+html.escape(heading)+'''</h1><p>授权范围：A / C-mix / C-bounded，各约100万生成token，seed17，GPU1共享。先2次真实更新/保存，再从checkpoint恢复续跑。D、额外seed、600万均不启动。</p><p>controller存活：'''+str(d['controller_alive'])+'''；当前子进程存活：'''+str(d['child_alive'])+'''。时间：'''+datetime.datetime.fromtimestamp(d['unix'],datetime.timezone.utc).isoformat()+'''</p><table><tr><th>组别</th><th>状态</th><th>更新数</th><th>生成token</th><th>错答正优势累计</th><th>显存reserved GiB</th></tr>'''+''.join(cells)+'''</table><p>C-mix中可能有错答正优势，这是原奖励归一化机制的被测现象；A和C-bounded要求该项为0。</p><p>CPU536例复算通过；验证集500题与策略训练/C适配数据无题目重叠。无更新验收不能代替实际训练/恢复验收，当前仅按真实receipt报告。</p><p><a href="latest_snapshot.json">完整状态和验收证据</a> · <a href="preflight.json">CPU准入</a></p><h2>当前日志末尾</h2><pre>'''+html.escape(d.get('active_log_tail') or '')+'''</pre></main></html>'''
    (HERE/'review.html').write_text(doc)
    print(json.dumps(status,ensure_ascii=False,indent=2))
    print((d.get('active_log_tail') or '')[-3500:])
if __name__=='__main__':main()
