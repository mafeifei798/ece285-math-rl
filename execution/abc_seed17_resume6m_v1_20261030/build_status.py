"""Local status artifact from receipts, with planned and completed stages separated."""
from pathlib import Path
import json,html,hashlib
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else {}
snap=read(HERE/'latest_snapshot.json');dispatch=read(HERE/'dispatch.json');admission=read(HERE/'admission.json')
pause=read(HERE/'pause_receipt.json')
controller=snap.get('controller') or {'stage':'dispatched; waiting for refreshed snapshot'}
names={'A':'A：结果奖励','C_mix':'C-mix：直接混合','C_bounded':'C-bounded：有界过程项'}
rows=[]
for a,label in names.items():
 data=snap.get('arms',{}).get(a,{})
 current=data.get('progress') or {};origin=admission.get('origins',{}).get(a,{})
 rows.append([label,current.get('global_step',48),f'{current.get("generated_completion_tokens",origin.get("tokens",0)):,}',
    '通过' if (data.get('stage_result50') or {}).get('passed') else '待完成',
    '完成' if (data.get('stage_result144') or {}).get('passed') else '未完成',
    '完成' if (data.get('stage_result288') or {}).get('passed') else '未完成'])
body='<small>ECE285 · seed17独立续跑 · 2026-10-30 PT</small><h1>A／C-mix／C-bounded：约6M共同预算续跑</h1>'
body+='<div class="note">从三组既有checkpoint-48恢复模型、优化器、scheduler、RNG和数据位置。统一V1，奖励和loss未变。仅seed17。第144次更新留验证点，第288次更新后统一MATH500/GSM8K测试；3M/6M是近似token标记，实际计账。</div>'
body+='<p><strong>最近核验阶段：</strong>'+html.escape(str(controller.get('stage')))+'；controller PID '+str(dispatch.get('pid','待派发'))+'。这是最近一次人工刷新快照，不是实时页面。</p>'
body+='<table><tr>'+''.join('<th>'+s+'</th>' for s in ['组别','累计更新','累计训练生成token','48→50验收','144步','288步'])+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(s))+'</td>' for s in row)+'</tr>' for row in rows)+'</table>'
body+='<h2>控制器执行顺序</h2><ol><li>三组48→50，真实续跑、保存和恢复验收，计入总预算。</li><li>三组50→144；保留完整144步断点，在固定500题验证集上评测，不据测试调参。</li><li>三组144→288；保留完整终点，统一greedy评测MATH500和GSM8K，再CPU重算验答、检查题目顺序和hash。</li></ol>'
body+='<p>预计新增约15M训练回答token。按此前吞吐先估计约15–25小时（含评测），共享负载变化时以真实进度更新；这不是完成保证。有限控制器48小时预算，故障保存证据后停止，不自动重试、不从头训练、不扩预算。</p>'
body+='<p>GPU1共享；每次GPU阶段前核对显存，不停止其他作业。三组相同题目顺序、相同更新数、相同重启关口。vLLM采样RNG会在进程重启时重建，不能声称与不间断运行逐位一致。</p>'
body+='<h2>版本与证据</h2><p>远端：<code>/path/to/ece285_math_rl_project/runs/ABC_seed17_resume6M_V1_20261030</code></p><p><a href="admission.json">CPU预检</a> · <a href="dispatch.json">唯一派发记录</a> · <a href="latest_snapshot.json">最近状态与原始receipt</a> · <a href="CONTINUATION.txt">执行范围和恢复说明</a> · <a href="../../experiments/abc_seed17_resume6m_v1_20261030/protocol.json">固定协议</a>。</p>'
if pause.get('paused_by_user'):
 body='<div class="note"><strong>已按用户要求暂停。训练进程和调度器均已退出，GPU已释放，不会自动恢复。</strong></div>'+body
 body=body.replace('控制器执行顺序','原计划（已暂停，未继续执行）')
 body=body.replace('预计新增约15M训练回答token。按此前吞吐先估计约15–25小时（含评测），共享负载变化时以真实进度更新；这不是完成保证。有限控制器48小时预算，故障保存证据后停止，不自动重试、不从头训练、不扩预算。','以上训练与评测计划已暂停。后续只有收到明确恢复指令后，才能从已核验完整断点继续；当前没有完成3M验证或6M测试。')
 body+='<h2>暂停时已核验的完整断点</h2><table><tr><th>组别</th><th>可恢复更新次数</th><th>断点训练生成token</th><th>未保存的已完成更新</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [names[a],d['checkpoint_step'],f'{d["checkpoint_generated_tokens"]:,}',d['completed_updates_beyond_checkpoint']])+'</tr>' for a,d in pause['arms'].items())+'</table><p><a href="pause_receipt.json">完整断点核验</a> · <a href="pause_process_receipt.json">进程退出与GPU释放证据</a>。</p>'
if snap.get('failure'):body+='<pre>'+html.escape(json.dumps(snap['failure'],ensure_ascii=False,indent=2))+'</pre>'
css='body{margin:0;background:#f1f5f8;color:#1b2d3d;font:16px/1.85 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1100px;margin:28px auto;background:white;padding:32px 40px;border-radius:14px}h1{font-size:28px;line-height:1.5}h2{font-size:21px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #d9e3ec}th{background:#edf3f7}.note{padding:16px 20px;background:#edf6fb;border-left:4px solid #327da3}small{color:#607384}code{overflow-wrap:anywhere}a{color:#1c638d}'
(HERE/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>seed17三组续跑进度</title><style>'+css+'</style><main>'+body+'</main></html>')
print(json.dumps({'report':str(HERE/'review.html'),'stage':controller.get('stage'),'training_complete':controller.get('stage')=='complete'},ensure_ascii=False))
