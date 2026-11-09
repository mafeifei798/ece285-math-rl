from pathlib import Path
import json,html,datetime
HERE=Path(__file__).resolve().parent


def render(x):
    rows=[]
    for arm,v in x['arms'].items():
        progress=v['progress'] or {};result=v['result'] or {}
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(a))}</td>' for a in
           (arm,progress.get('additional_updates',0),progress.get('generated_tokens','—'),
            progress.get('expanded_leaf_tokens','—'),'完成' if result else '失败' if v['failure'] else '运行中' if progress else '等待'))+'</tr>')
    c=x['controller'] or {};phase=c.get('stage','未启动')
    summary=''
    if x.get('comparison'):
        z=x['comparison'];summary='<h2>统一验证集</h2><table><tr><th>模型</th><th>准确率</th><th>截断</th><th>缺框</th></tr>'
        for name,v in z['validation_summary'].items():
            summary+=f'<tr><td>{name}</td><td>{v["correct"]}/500 ({v["accuracy"]*100:.1f}%)</td><td>{v["truncated"]}</td><td>{v["missing_or_invalid_box"]}</td></tr>'
        summary+='</table><h2>配对比较</h2><pre>'+html.escape(json.dumps(z['paired_comparisons'],ensure_ascii=False,indent=2))+'</pre>'
    when=datetime.datetime.fromtimestamp(x['unix'],datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M:%S PST')
    gate=x['no_update']['result'] or {}
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>树状信用三组短试</title><style>body{font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f3f5f8;color:#22334b;margin:0}main{max-width:1000px;margin:32px auto;background:white;padding:36px;border-radius:12px}h1{font-size:28px}h2{font-size:21px;margin-top:30px}table{width:100%;border-collapse:collapse}th,td{padding:10px;text-align:left;border-bottom:1px solid #dde5ee}th{background:#f4f7fb}.state{background:#eaf3ff;padding:18px;border-left:4px solid #447ab7}.muted{color:#66778d;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:15px}a{color:#2463a6}</style><main><h1>树状信用分配 · 三组配对短试</h1>'''
    page+=f'<p class="muted">最后核对：{when} · seed17 · 同一 C-bounded checkpoint-288 → 336</p>'
    page+=f'<div class="state">阶段：{html.escape(phase)}<br>控制器存活：{bool(x["controller_alive"])}；CPU 门槛：{bool(x["cpu_check"] and x["cpu_check"].get("passed"))}；真实模型零更新验收：{bool(gate.get("passed"))}</div>'
    page+='<h2>训练进度（各 48 次更新）</h2><table><tr><th>组别</th><th>已更新</th><th>唯一生成 token</th><th>展开后的训练 token</th><th>状态</th></tr>'+''.join(rows)+'</table>'
    page+='<p>chain_cb：原链式 C-bounded。tree_vr：树状采样＋验答树信用，不加载 PRM。tree_cb：相同树状信用＋固定有界 PRM 偏好。</p>'
    page+='<p>树形状 (2,1,1,3)，每题八个叶子；完整路径上限 2048。错答的共享前缀可能得到正的结果信用；错答没有正的 PRM 项，独有错误后缀也不能拿正优势。保持原提示词、验答器、LR、KL 与 loss reduction。</p>'
    page+=summary
    if c.get('stage')=='failed':page+='<h2>故障证据</h2><pre>'+html.escape(x.get('child_log_tail',str(c)))+'</pre>'
    page+='<h2>资源快照</h2><pre>'+html.escape(x['gpu'])+'</pre><p class="muted">本轮只做验证集，不运行测试集或自动扩展预算。原始 A/B/C/D、PURE 和此前短试保留。配对题目区间不包含训练 seed 或运行间数值波动。</p></main></html>'
    (HERE/'review.html').write_text(page)


if __name__=='__main__':render(json.loads((HERE/'status.json').read_text()))
