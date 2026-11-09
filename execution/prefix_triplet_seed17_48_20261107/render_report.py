"""Render only verified completed validation results; no remote operations."""
from pathlib import Path
import json,html
HERE=Path(__file__).resolve().parent
LABELS={'source':'共同起点 checkpoint-288','bounded':'C-bounded 续跑','uniform':'整条均匀减弱负优势','prefix':'前缀保护'}
def main():
    e=HERE/'completed_evidence';x=json.loads((e/'comparison.json').read_text());assert x['passed']
    m=x['paired_comparisons']['prefix_minus_uniform'];lo,hi=m['paired_bootstrap_95CI_pp'];diff=m['difference_percentage_points']
    if lo>0:verdict='前缀保护在这次 seed17 验证集中高于均匀减罚；这仍是短程、单 seed 的结果。'
    elif hi<0:verdict='前缀保护在这次 seed17 验证集中低于均匀减罚，不支持扩展这项改动。'
    else:verdict='这次短试无法确认前缀保护优于均匀减罚；配对区间包含零。'
    lines=['三组配对短试：seed17，从同一 C-bounded checkpoint 续跑48次更新','',verdict,'',
        '准确率（原 validation.jsonl 的500题验证集，不是MATH500测试集；统一 V1、原生LoRA、greedy、2048 token上限）']
    table=[]
    for arm in ['source','bounded','uniform','prefix']:
        s=x['validation_summary'][arm];row=[LABELS[arm],f"{int(s['correct'])}/500",f"{s['accuracy']*100:.2f}%",f"{s['tokens']/s['n']:.0f}",str(s['truncated']),str(s['short_wrong'])]
        table.append(row);lines.append(' | '.join(row))
    lines+=['','主比较：前缀保护 − 均匀减罚',f'准确率差 {diff:+.2f} 个百分点；题目配对 bootstrap95%区间 [{lo:+.2f}, {hi:+.2f}]。',
        f"前缀保护独胜 {m['wins']} 题，均匀减罚独胜 {m['losses']} 题；McNemar exact p={m['mcnemar_exact_two_sided_p']:.4f}。",'',
        '训练成本（各48次新更新、1536条新回答；实际生成token不强行对齐）']
    costs=[]
    for arm in ['bounded','uniform','prefix']:
        s=x['training_costs'][arm];row=[LABELS[arm],f"{s['new_generation_tokens']:,}",f"{s['total_seconds']/3600:.2f}h",f"{s['peak_sampled_GPU_GiB']:.2f}GiB",str(s['truncated']),str(s['short_wrong'])]
        costs.append(row);lines.append(' | '.join(row))
    lines+=['','改动覆盖率（训练轨迹各自统计，不把不同模型的回答当作相同回答）']
    coverage=[]
    for arm in ['bounded','uniform','prefix']:
        s=x['training_costs'][arm];n=s['signal_statuses'].get('prefix_protected',0);g=s['group_types'];total=sum(g.values())
        row=[LABELS[arm],f"{n}/1536 ({100*n/1536:.1f}%)",str(s['protected_tokens_if_prefix']),f"{g.get('all_wrong',0)}/{total}",str(s['empty'])]
        coverage.append(row);lines.append(' | '.join(row))
    lines+=['','对照含义与限制',
        'prefix–uniform 检验局部位置是否比整体减罚更有用；对同一回答，两种规则的负优势绝对值之和相同，但实际训练轨迹和梯度范数不保证相同。prefix–bounded 单独不能回答这个问题。',
        '三组题序、更新次数和运行设置一致；每组均从同一adapter、optimizer、scheduler恢复。起点已有约6M训练，本次是继续训练的短试。',
        '只用seed17；bootstrap只反映这500题的波动，不包含训练seed方差。验证集此前使用过，并非全新盲测。',
        '保护位置由冻结PRM首次低于0.8的步骤确定，不等于人工确认的数学首错。全错组的策略优势仍为零，原KL项保留。',
        '没有评测或据此挑选 MATH500/GSM8K 测试结果；没有自动扩大训练预算。',
        f"原保护清单 {x['old_protected_files_verified']} 个文件核验未变。",'','详细数据见 completed_evidence/comparison.json，完整预测和胜负题也已保存。']
    limitation_lines=lines[lines.index('对照含义与限制')+1:-2]
    if x.get('recovery'):
        lines+=['','故障恢复说明：uniform 在首次更新前被过严的逐字一致检查拦住，已复用原32条缓存回答，没有重采样或重跑 bounded。',
            '恢复后仍为相同seed/模型/优化器/题序/奖励/loss，但故障时的生成后采样随机状态没有保存，因此不能声称随机采样轨迹逐位复现。',
            '成本已计入失败尝试墙钟时间；该首批PRM forward秒数未保存，uniform的PRM计时略有缺项。']
    check=json.loads((HERE/'completion_verification.json').read_text());assert check['passed']
    secondary=[]
    for key,label in [('prefix_minus_bounded','前缀保护 − C-bounded'),('uniform_minus_bounded','均匀减罚 − C-bounded')]:
        c=x['paired_comparisons'][key];a,b=c['paired_bootstrap_95CI_pp']
        secondary.append([label,f"{c['difference_percentage_points']:+.2f}pp",f"[{a:+.2f}, {b:+.2f}]pp"])
    decision='暂不扩大前缀保护训练，也不把它替换为主方法。前缀保护相对均匀减罚只多对1题，无法支持位置特有收益；两种减罚相对C-bounded的观察增益也都包含零，不能宣称已证实减罚有效。保留本轮作为机制消融。'
    lines+=['','次要对照（探索性）']+[' | '.join(r) for r in secondary]
    lines+=['','完成验收',
        '本地91个取回文件逐个SHA256核对通过；三组各48次更新、1536条回答、相同题序。错答正优势token与额外归一化次数均为0。',
        '前缀组1条回答未通过原始字节对齐检查，已按预设回退到原C-bounded优势，没有强行遮罩。',
        '三组训练累计4.30小时；四模型验证约50.4分钟。已确认GPU进程退出。',
        '', '定性案例（非盲审，不能据此估计过程有效率）']
    lines += [c['problem_id']+'：'+c['note'] for c in check['cases']]
    lines+=['','建议',decision]
    text='\n'.join(lines)+'\n';(HERE/'REPORT.txt').write_text(text)
    def tab(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table>'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>前缀保护短试结果</title><style>body{max-width:1050px;margin:48px auto;padding:0 24px;font:16px/1.65 system-ui;color:#183047;background:#f5f7fa}h1{font-size:28px}section{background:white;border:1px solid #dce2e9;border-radius:12px;padding:24px;margin:20px 0}table{width:100%;border-collapse:collapse}td,th{padding:12px 8px;border-bottom:1px solid #e5e9ed;text-align:left}th{color:#496178}p{margin:12px 0}.muted{color:#587080}</style><h1>前缀保护三组短试</h1><p class="muted">seed17 · 共同 checkpoint-288 · 各续跑48次更新 · 500题验证集（非MATH500）</p><section><strong>'+html.escape(verdict)+'</strong><p>'+html.escape(f'前缀保护 − 均匀减罚：{diff:+.2f} 个百分点，95%配对区间 [{lo:+.2f}, {hi:+.2f}]。')+'</p><p>'+html.escape(f"独胜题：前缀保护 {m['wins']}，均匀减罚 {m['losses']}。")+'</p></section><section><h2>验证集结果</h2>'+tab(['模型','正确题数','准确率','平均输出token','截断','≤32 token错答'],table)+'</section><section><h2>实际训练成本</h2>'+tab(['组别','新生成token','训练墙钟','采样峰值显存','训练截断','训练短错答'],costs)+'</section><section><h2>解释边界</h2>'+''.join('<p>'+html.escape(t)+'</p>' for t in limitation_lines)+'</section></html>'
    if x.get('recovery'):
        page=page.replace('</html>','<section><h2>恢复记录</h2><p>'+html.escape('Uniform 首次更新前被过严检查拦住。复用原32条缓存回答，无重采样，无基线重跑；种子和算法设置未变，但故障后的采样随机状态不能逐位恢复。该限制和额外运行成本均保留。')+'</p></section></html>')
    page=page.replace('<section><h2>解释边界</h2>', '<section><h2>改动覆盖率</h2>'+tab(['组别','满足保护条件的回答','对应前缀token','全错组','空回答'],coverage)+'<p>每组分别统计自身训练输出；均匀减罚组的前缀token只用于计算同一回答下的等量控制，不代表它执行了局部保护。</p></section><section><h2>解释边界</h2>')
    page=page.replace('</html>', '<section><h2>次要对照</h2>'+tab(['比较','准确率差','95%配对区间'],secondary)+'<p>这些是探索性比较，不能用其中较好的一项替代预先确定的主比较。</p></section><section><h2>验收与案例</h2><p>91个取回文件SHA256核验通过，三组题序与更新数相同。错答正优势及重复归一化均为0；1条字节对齐失败回答按预设回退。GPU已释放。</p>'+''.join('<details><summary>'+html.escape(c['problem_id'])+'</summary><p>'+html.escape(c['note'])+'</p></details>' for c in check['cases'])+'<p>案例仅作定性说明，不是盲审，也不是过程有效率估计。完整文本见 completion_verification.json。</p></section><section><h2>本轮建议</h2><p>'+html.escape(decision)+'</p><p><a href="completed_evidence/comparison.json">原始比较JSON</a> · <a href="completion_verification.json">验收与案例</a> · <a href="REPORT.txt">文本报告</a></p></section></html>')
    (HERE/'review.html').write_text(page);print(text)
if __name__=='__main__':main()
