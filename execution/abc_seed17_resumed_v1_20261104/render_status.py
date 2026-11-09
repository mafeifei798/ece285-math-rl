"""Render a dated local snapshot; never calls the server or changes training."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import html
import json

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / 'live_status.json').read_text())
if (d.get('comparison') or {}).get('passed') and (HERE / 'results.html').exists():
    (HERE / 'review.html').write_text((HERE / 'results.html').read_text())
    print(HERE / 'review.html')
    raise SystemExit(0)
dispatch = d['dispatch']
start_tokens = {'A': 2390451, 'C_mix': 1034873, 'C_bounded': 1038895}
rows = []
for arm, a in d['arms'].items():
    p = a['progress']
    resumed = a['resume_acceptance_144']
    state = '排队，尚未开始新的更新'
    if a['failure']:
        state = '失败，停止并保留证据'
    elif a['stage_288_result']:
        state = '训练至 288 步完成；评测状态见控制器'
    elif a['stage_144_result']:
        state = '144 步阶段完成'
    elif resumed and resumed.get('passed'):
        state = '已通过真实更新续训验收'
    step = p['global_step'] if p else dispatch['start_steps'][arm]
    tokens = p['generated_completion_tokens'] if p else start_tokens[arm]
    rows.append(f'<tr><td>{html.escape(arm)}</td><td>{dispatch["start_steps"][arm]}</td>'
                f'<td>{step}</td><td>{tokens:,}</td><td>{state}</td></tr>')
timestamp = datetime.fromtimestamp(d['unix'], timezone.utc).astimezone(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')
stage = html.escape((d.get('controller') or {}).get('stage', 'unknown'))
accepted = [arm for arm, a in d['arms'].items() if (a['resume_acceptance_144'] or {}).get('passed')]
accepted_text = '、'.join(accepted) if accepted else '暂无'
body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECE285 — A / C-mix / C-bounded 续训</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1040px;margin:40px auto;padding:0 24px;color:#183047;background:#f6f8fb}}h1{{font-size:28px}}section{{background:white;border:1px solid #dbe2ea;border-radius:12px;padding:20px 24px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e3e8ed}}th{{background:#eef3f8}}.muted{{color:#556879}}code{{overflow-wrap:anywhere}}a{{color:#175fa1}}</style>
<h1>A / C-mix / C-bounded 已恢复续训</h1>
<p class="muted">服务器证据时间（Los Angeles）：{html.escape(timestamp)}。这是快照，不是实时页面。</p>
<section><h2>当前状态</h2><p>控制器阶段：<strong>{stage}</strong>；PID：{dispatch['pid']}。从完整 checkpoint 接续，未从头重训。</p>
<table><thead><tr><th>组别</th><th>续训起点</th><th>最新完成步数</th><th>累计生成 token</th><th>状态</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>已通过本次真实模型更新验收的组别：{html.escape(accepted_text)}。各组按顺序接续；排队不等于已经跑完。</p></section>
<section><h2>执行范围</h2><p>三组先到共同第 144 步（约 300 万 token）并做统一验证，再到第 288 步（约 600 万 token），统一 greedy 评测 MATH500 / GSM8K。实际 token 依回答长度有所差异；三组固定题目顺序与更新次数相同。</p>
<p>seed17、V1、奖励、loss、PRM、提示词与验答规则保持原设置。一张共享 A100，三组顺序执行。所有新文件位于 <code>/path/to/ece285_math_rl_project/runs/ABC_seed17_resumed_V1_20261104</code>。</p></section>
<section><h2>已验收与限制</h2><p>CPU admission 通过：536 条参考计算，1,535 个受保护文件；训练信号与评测代码逐字节保持一致。A 首次真实续训验收确认权重、优化器、调度器、Torch CPU/CUDA 随机状态恢复，题目顺序和实际 loss 输入一致，未重复归一化。</p>
<p>vLLM 采样随机状态重新创建，因此不能声称与不中断训练逐位等同。A 在 115 步额外重启，另外两组从 50 步恢复，这项重启历史差异会保留在最终分析中。共享算力的完成时间仍会波动；单 seed 结果不证明稳定收益。</p>
<p>控制器有有限时限和失败停止机制，不会自动从头重跑。旧代码、checkpoint 和结果保持保留。</p></section>
<section><h2>证据与接续说明</h2><p><a href="CONTINUATION.txt">完整接续说明</a> · <a href="admission.json">CPU admission</a> · <a href="dispatch.json">启动 receipt</a> · <a href="live_status.json">实时核验快照</a></p></section></html>'''
(HERE / 'review.html').write_text(body)
print(HERE / 'review.html')
