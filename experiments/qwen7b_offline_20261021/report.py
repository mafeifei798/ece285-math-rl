"""Readable paired audit report; all figures are measured or explicit counts."""
import html
import json
from pathlib import Path
import sys
from core import read_jsonl


def main():
    root,out=map(lambda p:Path(p).resolve(),sys.argv[1:3])
    s=json.loads((out/'summary.json').read_text())
    r=s['review']['uniform_random'];t=s['review']['targeted_combined']
    lines=['Qwen 7B PRM 对现有 B 回答的独立离线评分', '',
           '本轮没有训练、生成新回答、更新任何模型权重或恢复 B。原 A/B 数据、代码和结果均保留。',
           '输入：B 最初 6M 的 9120 条回答、1140 个八回答组；7472 条完整可评分回答。',
           '主比较：同一原始双换行步骤、同一原始结果标签、同一 reward=outcome+0.2*min(step scores)，两模型均重新推理。',
           '副比较：两模型共同使用此前固定的 CRLF 视图和百分数判分修复，不混入主比较。',
           '各模型使用官方原生序列化；Qwen 使用聊天模板及每步 <extra_0>，H4 使用原生双换行前缀末端。',
           '不做分数校准、温度调整、阈值选择或奖励权重调参；分类阈值固定 0.5。', '',
           '一、已审阅回答：随机 30 条（1 条题面矛盾排除；2 条语义不确定单列）',
           '以下是固定小样本的描述性计数；逐步判断来自此前 Codex 单审阅，非独立人工金标准。']
    def table_summary(data):
        h,q=data['h4'],data['qwen7b']
        return [
            f"全程有效回答被拒绝（min<0.5）：H4 {h['clean_solutions_falsely_rejected_at_0p5']}/{data['clean_solutions']}；Qwen {q['clean_solutions_falsely_rejected_at_0p5']}/{data['clean_solutions']}",
            f"有明确错误但全程放过（min>=0.5）：H4 {h['erroneous_solutions_fully_passed_at_0p5']}/{data['erroneous_solutions']}；Qwen {q['erroneous_solutions_fully_passed_at_0p5']}/{data['erroneous_solutions']}",
            f"首个确定错误步骤漏判：H4 {h['first_error_steps_missed_at_0p5']}/{data['determinate_first_error_solutions']}；Qwen {q['first_error_steps_missed_at_0p5']}/{data['determinate_first_error_solutions']}",
            f"首错定位命中：H4 {h['first_error_location_exact_at_0p5']}/{data['determinate_first_error_solutions']}；Qwen {q['first_error_location_exact_at_0p5']}/{data['determinate_first_error_solutions']}",
            f"确定有效前缀中的步骤误罚：H4 {h['valid_prefix_steps_falsely_rejected_at_0p5']}/{data['certain_valid_prefix_steps']}；Qwen {q['valid_prefix_steps_falsely_rejected_at_0p5']}/{data['certain_valid_prefix_steps']}",
            f"有效回答负 advantage：H4 {h['clean_solutions_negative_advantage']}/{data['clean_solutions']}；Qwen {q['clean_solutions_negative_advantage']}/{data['clean_solutions']}"]
    lines+=table_summary(r)
    lines+=['','二、针对旧评分器选择的可疑 20 条','这些样本专门按旧分数选取，只能作故障回归，不能拿两模型的差距外推总体收益。']+table_summary(t)
    lines+=['','三、组内排序与实际 GRPO advantage（每组八个伙伴全部保留）']
    for key,title in [('model_only_raw','原始口径：更换评分器'),('model_only_repaired','共同修复口径：更换评分器'),('historical_H4_reinference_drift','历史 H4 与本轮 H4 重推的数值漂移（不算模型收益）')]:
        d=s['comparisons'][key]
        lines += [title,
                  f"共同保留 {d['groups']} 组/{d['rows']} 条；reward 排序改变 {d['reward_order_changed_groups']} 组；process 排序改变 {d['process_order_changed_groups']} 组。",
                  f"advantage 正负/零状态改变 {d['advantage_sign_flips']} 条；|Δadvantage|>0.01 为 {d['advantage_changed_rows_gt_0p01']} 条；平均绝对变化 {d['mean_abs_advantage_change']:.6f}。"]
        for kind,v in d['by_group_type'].items():
            lines.append(f"  {kind}: {v['groups']} 组，reward 排序变化 {v['reward_order_changed_groups']} 组，advantage 符号变化 {v['advantage_sign_flips']} 条。")
    lines+=['排序变化不等于数学上更好；全量伙伴没有步骤真值，方向只在已核对案例中判断。',
            '最终答错也可能包含有效步骤，因此答错回答获得正 advantage 并不自动构成错误奖励。',
            '全对/全错组里，0.2 系数仍会被标准差归一化近似抵消；更低的绝对分不保证更弱或更正确的训练信号。',
            '', '四、具体案例（原始口径，实际原八回答组）', 'ID | H4 min/rank/advantage | Qwen min/rank/advantage']
    cases=json.loads((out/'review_cases.json').read_text())
    for c in cases:
        if c['review_id'] in ['R01','R02','R04','R07','R14','S03','S08','S11','S19','S20']:
            lines.append(f"{c['review_id']} | {c['h4_min']:.6f}/{c['h4_rank']:.1f}/{c['h4_advantage']:+.6f} | {c['qwen7b_min']:.6f}/{c['qwen7b_rank']:.1f}/{c['qwen7b_advantage']:+.6f}")
    for c in s['known_error_cases']:
        lines.append(f"旧已知错误行 {c['source_row']} | {c['h4_min']:.6f}/{c['h4_rank']:.1f}/{c['h4_advantage']:+.6f} | {c['qwen7b_min']:.6f}/{c['qwen7b_rank']:.1f}/{c['qwen7b_advantage']:+.6f}")
    lines+=['R04 的原始结果标签存在百分数单位错误，原始口径特意保留；请结合共同修复口径看，不归咎于新旧 PRM。',
            '', '五、显存与吞吐（同一 A100 80GB、BF16、microbatch=8、按长度分桶）']
    for name,v in s['resources'].items():
        lines += [f"{name}: {v['total_views']} 次回答视图评分，{v['input_tokens']:,} 非 padding 输入 token；模型载入权重分配 {v['model_memory_allocated_GiB']:.3f} GiB。",
                  f"  生产推理 {v['forward_seconds']:.3f} 秒；{v['production_views_per_second']:.2f} 视图/秒；{v['unpadded_input_tokens_per_second']:,.0f} 非 padding 输入 token/秒。",
                  f"  PyTorch 分配峰值 {v['allocated_peak_GiB']:.3f} GiB；保留峰值 {v['reserved_peak_GiB']:.3f} GiB；含控制检查的墙钟时间 {v['wall_seconds_including_controls']:.3f} 秒。"]
    lines+=['这是独立评分器成本，不含 actor、生成 KV cache、RL 反向、优化器，也不代表完整 RL 的显存或吞吐。',
            '视图评分包含 CRLF 变体，不能把它当作生成新回答的速度。两个模型的原生模板 token 数不同，实际 token 成本分别记录。',
            '', '六、结论边界',
            '此次只能判断同一批旧 B 轨迹上评分器与学习信号是否改善，不能推出新 B 的策略准确率会提高。',
            '已有 50 条案例均看过且非独立人类标注，不能据此挑阈值、反复调参数或声称统计显著。',
            '详细字段、原文、步骤分数、排名与优势见 review_comparison.html、review_cases.csv、review_steps.csv、variants/。',
            '资源测量见 receipt_h4.json、receipt_qwen7b.json；校验见 analysis_QA.json、independent_QA.json 和原始文件保护回执。']
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    review_by_id={r['review_id']:r for r in json.loads((out/'reviewed_records.json').read_text())}
    scores={m:{(r['source_row'],r['view']):r for r in read_jsonl(out/f'scores_{m}.jsonl')} for m in ['h4','qwen7b']}
    esc=lambda x:html.escape(str(x))
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Qwen 7B / H4 离线对照</title><style>',
          'body{font:16px/1.65 system-ui,sans-serif;background:#f3f5f8;color:#182330;max-width:1250px;margin:30px auto;padding:0 24px}h1{font-size:28px}h2{font-size:21px}.note{background:#fff4d7;padding:18px;border-radius:12px}.case{background:white;margin:24px 0;padding:24px;border:1px solid #dce3ed;border-radius:14px}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border-bottom:1px solid #e1e5eb;padding:10px;text-align:left;vertical-align:top}th{background:#f5f7fb}.text{white-space:pre-wrap;overflow-wrap:anywhere}.score{font-variant-numeric:tabular-nums;white-space:nowrap}.E{background:#fff0f0}.U{background:#fff8e8}.meta{color:#536275}nav a{display:inline-block;margin:3px 8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}</style>',
          '<h1>同一批数学回答：Qwen 7B PRM 与旧 H4 对照</h1>',
          '<p class="note">冻结模型离线评分；零训练、零新生成。下列 C/E/U 来自之前的 Codex 单审阅，不是独立人工金标准。可疑样本按旧分数选择，只作故障回归。固定阈值为 0.5；排名与 advantage 来自原来的完整八回答组。</p>',
          '<details><summary>完整统计与成本报告</summary><pre>'+esc('\n'.join(lines))+'</pre></details><nav>']
    body += [f'<a href="#{c["review_id"]}">{c["review_id"]}</a>' for c in cases];body+=['</nav>']
    for c in cases:
        review=review_by_id[c['review_id']];i=c['source_row']
        body += [f'<section class="case" id="{c["review_id"]}"><h2>{c["review_id"]} · {esc(c["verdict"])}</h2>',
                 f'<p class="meta">原始行 {i} · {esc(c["category"])} · {esc(c["group_type"])} · 结果标签 {c["outcome_original"]}（共同修复后 {c["outcome_repaired"]}）</p>',
                 '<p class="text">'+esc(review['question'])+'</p>',
                 '<p>H4 min / rank / advantage：'+f'{c["h4_min"]:.6f} / {c["h4_rank"]:.1f} / {c["h4_advantage"]:+.6f}'+
                 '　Qwen：'+f'{c["qwen7b_min"]:.6f} / {c["qwen7b_rank"]:.1f} / {c["qwen7b_advantage"]:+.6f}'+'</p>',
                 '<table><thead><tr><th>步骤</th><th>原文</th><th>原审阅</th><th>H4</th><th>Qwen 7B</th></tr></thead><tbody>']
        for j,(text,label) in enumerate(zip(review['steps'],review['review']['step_labels'])):
            hp=scores['h4'][(i,'raw')]['step_scores'][j];qp=scores['qwen7b'][(i,'raw')]['step_scores'][j]
            body.append(f'<tr class="{label}"><td>{j+1}</td><td class="text">{esc(text)}</td><td>{label}</td><td class="score">{hp:.6f}</td><td class="score">{qp:.6f}</td></tr>')
        body+=['</tbody></table><p class="text">核对依据：'+esc(review['review']['verification'])+'</p></section>']
    body+=['</html>']
    (out/'review_comparison.html').write_text('\n'.join(body))
    print('Wrote REPORT.txt and review_comparison.html',flush=True)


if __name__=='__main__':main()
