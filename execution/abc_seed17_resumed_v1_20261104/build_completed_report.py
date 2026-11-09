"""Summarize verified completed evidence, preserving every original result."""
from pathlib import Path
import html, json
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
c=read(HERE/'completed_evidence/comparison.json')
v=read(HERE/'independent_acceptance.json')
assert c['passed'] and v['passed']
oldroot=HERE.parent/'final_four_eval_20261029'
old=read(oldroot/'comparison.json')
oc=read(oldroot/'evidence/runs/Final_four_eval_seed17_1M_20261029/contract.json')
nc=read(HERE/'completed_evidence/evaluation/288/contract.json')
assert oc['verifier_sha256']==nc['repair_sha256'] and oc['units_sha256']==nc['unit_metadata_sha256']
assert all(oc['datasets'][ds]['sha256']==nc['dataset_sha256'][ds] for ds in ['math500','gsm8k'])
for k in ['seed','temperature','top_p','top_k','max_model_len','batch_size','max_num_seqs','max_num_batched_tokens','dtype','enforce_eager']:
 assert oc['protocol'][k]==nc[k],k
assert oc['protocol']['max_tokens']==nc['cap']
table=[]
for a in ['A','C_mix','C_bounded']:
 ds=c['summary'][a];s=v['signals'][a];acc=c['accounting'][a]
 table.append(f"| {a} | {acc['generated_completion_tokens']/1e6:.3f}M | {ds['math500']['correct']:.0f}/500 ({ds['math500']['accuracy']*100:.2f}%) | {ds['gsm8k']['correct']:.0f}/1319 ({ds['gsm8k']['accuracy']*100:.2f}%) | {acc['loss_wrong_positive']} |")
text='''ECE285：约 6M token 三组对照完成
证据整理日期：2026-11-05（America/Los_Angeles）；单训练 seed17。

结论
C-bounded 是当前最值得确认的方案。此次准确率高于纯结果奖励 A；原始混合 C-mix 的提升较小。
可以开始收尾，下一项训练应是独立 seed 的小范围确认，而不是继续加方法或直接扩至 12M/24M。

| 组别 | 累计生成训练 token | MATH500 | GSM8K | 错答获正 advantage 的训练回答数 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

比较与证据
• C-bounded − A：MATH500 +4.80 个百分点，逐题配对 bootstrap 95% 区间 [1.40, 8.40]；GSM8K +2.50，[0.45, 4.55]。
• C-mix − A：MATH500 +1.00，区间 [−2.20, 4.20]；GSM8K +0.83，[−0.91, 2.58]。
• C-bounded − C-mix：MATH500 +3.80，区间 [0.395, 7.40]；GSM8K +1.67，[−0.30, 3.64]。
这些是未做多重比较校正、只覆盖题目抽样的区间，不覆盖训练 seed 变化，不能据此称多 seed 稳定提升。

训练信号检查
C-mix 的 526 条错答正优势中，524 条来自全错组，2 条来自混合组。归一化后的过程调整最大绝对值 2.47。
C-bounded 的错答正优势为 0，实际最大过程调整约 0.169，未超过预定上限 0.2；27,648 条训练记录核对无重复归一化。
零错答正优势是公式约束，不等于独立的数学能力证据，也不能单独证明它造成了准确率提升。

收益是否只来自截断或答案格式
MATH500：C-bounded 赢 A 54 题、输 30 题；两者均正常结束的 447 题里净多答对 23 题，全体净多 24 题。
GSM8K：赢 111 题、输 78 题；两者均正常结束的 1,282 题里净多答对 23 题，全体净多 33 题。
因此收益不能全部归为少截断，但正常结束子集是事后条件统计，不能用于无偏因果估计。
仍有答案框问题：GSM8K 的 78 个失分对中，C-bounded 有 36 个 missing_or_invalid_box；MATH500 对应 6 个。
所有最终评测都没有空回答，也没有 ≤32 token 的错误回答。训练期仍存在短错答，不能说病态输出完全消失。
没有做独立人工推导有效性审查；终答正确不代表推导正确，不改变现有验答规则来抬高分数。

与 1M 结果的关系
共同 1M 评测中 A 为 49.4% / 57.01%，C-bounded 为 50.2% / 57.16%；现在分别为 53.0% / 66.87% 和 57.8% / 69.37%。
数据、验答规则与核心推理参数已核对一致。更充分训练后差距扩大，与“1M 不足以暴露差异”相容，但单条训练轨迹不能证明稳定的规模规律。
约 3M 的 500 题是独立验证集（A 54.4%、C-mix 56.4%、C-bounded 55.6%），不能当作 MATH500 点连成同一测试曲线。

实际资源
只看三组共同的 144→288 步：A 3.11 小时，C-mix 3.67 小时，C-bounded 3.61 小时。
恢复后观测到的本任务进程显存峰值：A 25.62 GiB，两个过程奖励组均 34.71 GiB。
共享算力和答案长度不同，这些是实测墙钟费用而非隔离环境的算法速度比较；PRM 适配费用另计。
恢复前起点不同，不能把本次恢复总时长直接当三组端到端效率比较。三组同为 288 次更新、9,216 条回答；生成 token 约相差 0.9%。

完成的验收
本地核对 6,957 条预测（验证 1,500 + 测试 5,457）、全部预测文件哈希、题目 ID、答案标签、原始 token 长度及记录中的 adapter 哈希。
27,648 条训练信号核对同题顺序、步数、累计 token、分组正优势和预设约束。
服务器独立 CPU 验收已用冻结验答器重新评分全部 5,457 条最终测试预测，确认 1,535 个受保护文件未变。
当前控制器 complete，项目进程已退出；未启动额外 GPU 工作。

下一步（建议，尚未启动）
1. 冻结当前方法、提示词、权重系数和评测协议，整理课程报告主体。保留初始模型、1M、约3M验证和约6M结果。
2. 对 A 与 C-bounded 的胜负题做小规模盲审，区分数学错误、答案框错误和碰巧答对；定向抽样只做机制诊断，不估计全体错误率。
3. 资源允许时只补 A / C-bounded 两个 seed42 约6M训练：共同初始模型、相同题序与更新次数、对称重启安排，PRM保持固定，不从 seed17 checkpoint 接续冒充独立 seed。先不重跑 C-mix，不加新方法，不升到24M。
4. seed42 若方向一致，将项目写成“过程奖励的训练信号诊断与有界修正”；若不一致，保留单seed正结果并明确稳定性未证实。
若课程时间或显卡不足，可以直接以当前单seed研究交付，缩小结论，不把未来确认写成已完成。

保留的限制
A 在 115 步多了一次恢复；vLLM 采样随机状态重新创建，重启历史不对称。Torch/优化器状态虽精确恢复，但不能宣称逐位等同不中断训练。
固定 C PRM 的锚点中存在已知 6 组冲突标签；此次没有改变 PRM。这不破坏同一 PRM 下的 C-mix/C-bounded 控制，但限制 PRM 适配本身的结论。
历史测试已被查看；不能再在这些测试上挑参数。没有人工过程有效性或跨模型结论。
'''
(HERE/'RESULTS.txt').write_text(text)
tablehtml='<table><thead><tr><th>组别</th><th>训练 token</th><th>MATH500</th><th>GSM8K</th></tr></thead><tbody>'
for a in ['A','C_mix','C_bounded']:
 tablehtml+=f'<tr><td>{a}</td><td>{c["accounting"][a]["generated_completion_tokens"]:,}</td><td>{c["summary"][a]["math500"]["accuracy"]:.2%}</td><td>{c["summary"][a]["gsm8k"]["accuracy"]:.2%}</td></tr>'
tablehtml+='</tbody></table>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECE285 6M 三组对照：已完成</title><style>body{font:16px/1.75 system-ui,sans-serif;max-width:1000px;margin:36px auto;padding:0 22px;color:#193047;background:#f7f9fc}section{background:white;border:1px solid #dbe3eb;border-radius:12px;padding:20px 24px;margin:20px 0}h1{font-size:28px}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #e1e7ed;padding:12px}th{background:#eff5f8}pre{font:15px/1.85 system-ui,sans-serif;white-space:pre-wrap;overflow-wrap:anywhere}a{color:#125da4}.small{color:#546879;font-size:14px}</style><h1>约 6M token 三组对照已完成</h1><p class="small">2026-11-05 · seed17 · 统一 V1 / greedy · 原代码与结果保留</p><section><strong>C-bounded 是目前最值得做独立重复确认的方案。</strong><p>相对 A，MATH500 +4.80 个百分点，GSM8K +2.50 个百分点。当前是单 seed 的正结果，不是已证实的稳定收益。</p>'''+tablehtml+'''</section><section><p><a href="RESULTS.txt">完整结果与后续方案</a> · <a href="independent_acceptance.json">独立 CPU 复核</a> · <a href="completed_evidence/comparison.json">原始汇总</a> · <a href="retrieval_receipt.json">取回校验</a> · <a href="CONTINUATION.txt">接续记录</a></p></section><section><pre>'''+html.escape(text)+'''</pre></section></html>'''
(HERE/'results.html').write_text(page)
(HERE/'review.html').write_text(page)
print(HERE/'results.html')
