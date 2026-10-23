"""Build an evidence-linked audit; no GPU, model, optimizer or remote mutation."""
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def read(name):
    return json.loads((HERE / name).read_text())

decomp = read('prm_signal_decomposition.json')
batch = read('evidence/P_PRM_batch_0002.json')
runtime = read('official_runtime_audit.json')
replacement = read('prm_replacement_evidence.json')
protected = read('protected_before.json')
changed = [p for p, h in protected.items()
           if not (ROOT / p).is_file() or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
assert not changed, changed
assert runtime['upstream']['matches_pin']
assert decomp['passed']
outcome_checks = []
for item in decomp['rows']:
    row = batch['rows'][item['row']]
    group = [float(r['outcome']) for r in batch['rows'] if r['problem_id'] == row['problem_id']]
    loo = float(row['outcome']) - (sum(group) - float(row['outcome'])) / (len(group)-1)
    assert len(group) == 8 and not any(group) and loo == 0
    outcome_checks.append({'row': item['row'], 'group_outcomes': group, 'outcome_LOO': loo})

inputs = ['prm_signal_decomposition.json', 'evidence/P_PRM_batch_0002.json',
          'evidence/P_PRM_batch_0002_signals.pt', 'official_runtime_audit.json',
          'prm_replacement_evidence.json']
receipt = {
    'created_utc': datetime.now(timezone.utc).isoformat(), 'cpu_only': True,
    'training_changed': False, 'models_loaded': False, 'scoring_launched': False,
    'protected_files_checked': len(protected), 'protected_files_changed': changed,
    'same_batch_hypothetical_outcome_LOO': outcome_checks,
    'counterfactual_scope': 'Only the verifier LOO component is computed; no mixed-reward run or complete alternate normalized advantages are claimed.',
    'input_sha256': {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in inputs},
    'decision': 'Keep the fixed running pilot unchanged. No missing official hard safeguard found. Qualify a candidate PRM offline on full current groups with its native input adapter before any replacement training. No replacement scoring performed in this audit.',
}
(HERE/'official_failure_audit_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')

labels = ['只答 243 并带无关文字（标准答案 77）', '只答 9（标准答案 77）', '题目完整却要求提供更多背景']
rows = ''.join(f'<tr><td>{html.escape(label)}</td><td>{x["prm_positive_score"]:.4f}</td>'
               f'<td>{x["signed_prm_reward"]:+.4f}</td><td>{x["baseline_delta"]:+.6f}</td>'
               f'<td>{x["mean_raw_content_return"]:+.4f}</td><td>{x["loss_mean_advantage"]:+.4f}</td></tr>'
               for label, x in zip(labels, decomp['rows']))
source_rows = ''.join(f'<tr><td>{html.escape(x["name"])}</td><td>{html.escape(x["finding"])}</td>'
                      f'<td><a href="{html.escape(x["url"], quote=True)}">{html.escape(x["path"])}:{x["line_start"]}</a></td></tr>'
                      for x in runtime['findings'])
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PURE 短错答与拒答：官方机制及替换 PRM 决策</title>
<style>body{font:16px/1.8 system-ui,sans-serif;background:#f5f7fa;color:#182437;margin:0}main{max-width:1060px;margin:auto;padding:40px 26px 70px}h1{font-size:30px;line-height:1.4}h2{font-size:21px;margin-top:32px}a{color:#1764ac}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dfe6ee;text-align:left;vertical-align:top}th{background:#eaf0f6}.lead{padding:18px 22px;border-left:4px solid #2367a1;background:white}.note{color:#536174;font-size:14px}code{background:#e8edf3;padding:1px 4px}.wide{overflow:auto}li{margin-bottom:9px}</style>
<main><p class="note">2026-10-23 · 原始 rollout + 官方源码 + 论文 v3 · 仅本地 CPU 核对</p>
<h1>PURE 为什么仍会奖励短错答，下一步怎么处理</h1>
<p class="lead"><strong>结论：这三条的错误正信号主要来自 PRM 原始打分；没有发现漏接的官方拒答过滤器。</strong>
当前单步输出经过 softmin 后原分保留。保留正在运行的配对 pilot；不改奖励、提示词或验答规则，也不立即换模型重新训练。
如果坚持不加新奖励，优先验证替代 PRM 在当前输出上能否纠正打分及实际 advantage，而不是再改归一化。</p>

<h2>1. 实际信号：归一化之前就已经鼓励错误输出</h2>
<div class="wide"><table><tr><th>真实输出</th><th>PRM 正类概率</th><th>有符号奖励</th><th>baseline 增量</th><th>归一化前正文平均 return</th><th>loss 中回答平均 advantage</th></tr>__ROWS__</table></div>
<p>这三条都只有一个评分步骤，softmin 权重等于 1；其用途是抑制多个步骤奖励相加造成的偏差，不能纠正单个步骤的误判。
全批标准差为 0.462316，归一化带来约 2.163 倍的共同缩放，但没有制造这三条的正号。
EOS advantage 已为 0，正文仍得到正优势。独立 32 条重算最大误差 2.56×10⁻⁷，loss 没有再次归一化。</p>
<p>三条占该批正的“回答平均 advantage”总和的 27.91%，不是梯度范数占比；仅凭这批不能判定已出现训练崩溃或准确率下降。
<a href="signal_decomposition_review.html">完整逐项分解</a> · <a href="prm_signal_decomposition.json">原始计算结果</a></p>

<h2>2. 官方是否也会这样：会，作者明确记录了这一局限</h2>
<p>论文 §5.1 Case 3 记录无关客套和空输出得到高分。作者解释为因果 PRM 缺少“这已经是整份回答终点”的判断。
论文没有逐例报告我们的拒答和短错答；它们来自本地真实 rollout，属于与上述机制相符的观察。
<a href="https://arxiv.org/html/2504.15275v3#S5.SS1">论文 §5.1</a></p>
<p>官方 OpenRLHF README 也承认投机，并说明展示过的纯 PRM checkpoint 取自投机前的 step 100。
这解释了为何早期有收益与后期出现漏洞可以同时成立。官方边界清零只去掉 EOS 等位置的直接信号，没有给正文拒答判负。
<a href="https://github.com/CJReinforce/PURE/blob/41f9363b8cc4af044b7fc5f7f9bbbf62139e9553/README.md#L96-L102">官方说明</a></p>
<p>论文 Table 2 对 Qwen2.5-Math-1.5B 报告最佳保存 checkpoint：纯 PRM / 纯 VR 的 MATH500 为 75.2% / 74.2%，五集平均为 40.3% / 40.6%。纯 PRM 并非全面胜过结果奖励。
混合方式为 76.0% / 43.3%。这些是原论文结果，不是本项目测量，也不承诺在小规模 LoRA 中复现。
<a href="https://arxiv.org/html/2504.15275v3#S4.SS2">设置与结果表</a></p>

<h2>3. 官方混合奖励能缓解，但不是这三条的即时修复</h2>
<p>官方 PRM+VR 使用部分题目的标准答案辅助训练（约 10% 的题，非 VR 权重 0.1）。这是另一条实验臂，不是纯 PRM 漏接的代码。
<a href="https://github.com/CJReinforce/PURE/tree/openrlhf">官方实验说明</a></p>
<p>本批两个相关题目组都是 8 条全错，结果奖励留一项 <code>yᵢ − mean(yⱼ, j≠i)</code> 全为 0。
因此仅给这些组加上标准答案，结果奖励的这一项不会抵消三条 PRM 正信号。这里计算的是固定样本上的分量，不是声称完成了混合训练或完整混合归一化回放。
<a href="official_failure_audit_receipt.json">CPU 检查凭据</a></p>

<h2>4. 换 PRM：可以检验，不能预先保证更好</h2>
<p>优先候选仍是服务器已有的 <strong>Qwen2.5-Math-PRM-7B</strong>。旧离线审计中它也给 9/9 空回答高正类分数（0.9982–1.0000）；
部分非空短错答评分降低，另一些反而升高。目前三条的题目和输出都不在旧比较集里，因此不能把旧结果当作它能修复本批的证据。</p>
<p>旧审计用了重构 token，其中的 advantage 来自另一条回放路径；不能冒充本次原始 token OpenRLHF 的直接比较。
<a href="../pure_official_prm_offline_20261022/summary.json">旧证据</a> · <a href="prm_replacement_evidence.json">适用边界</a></p>
<ol>
<li>先固定当前完整 32 条批次，再预先选一个独立完整批次作为复核集。保留每题 8 条关系和原始 actor token、边界、old/reference log-prob。</li>
<li>各 PRM 用自己的官方输入接口。Qwen 需要 chat template 和 <code>&lt;extra_0&gt;</code>，不能照搬 PURE 的换行评分头。步骤映射失败必须报告，不能重写 actor token。</li>
<li>新分数通过相同 signed reward、softmin、baseline、KL、一次归一化和 loss 权重做 CPU 回放。模型及必要的输入适配器是唯一被替换的部分。</li>
<li>先要求三个确认无有效推理的坏例不再获正回答平均优势，同时核对正确短答和有效推导，防止只把所有分数压低。再看独立批次的错误强化、错罚和排序；样本太少则报告不确定。这里不要求所有终答错误的 CoT 都为负：有数学进展的错误解答仍可能值得相对鼓励，重点是已确认没有有效推理的输出。</li>
<li>通过后才值得提出新的配对小试验；未通过就不新开 PRM 训练。不能拿这些诊断样本挑完模型后，声称已证明泛化或下游准确率提升。</li>
</ol>
<p><a href="https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B">Qwen 官方接口</a>。本次没有下载或加载替代模型，没有启动以上离线评分。</p>

<h2>5. 复现边界和当前决定</h2>
<p>当前分支 HEAD 仍为 <code>41f9363b</code>，与本项目 pin 一致。奖励、baseline、边界与归一化遵循该 OpenRLHF 路径；
整个实验应称为<strong>早期官方 OpenRLHF 算法的单卡 LoRA 适配</strong>，不是论文 v3 的完整训练复现。
v3 使用 veRL；当前小批量、长度、LoRA 和提示词与原配方不同。尤其当前提示词保留本项目协议，没有官方显式要求双换行分段的句子。
这些差异会影响输出分布，不能声称官方完整实验会以相同频率产生这些坏例；当前不能中途更改。</p>
<p>同输入、logits、组和 mask 下，官方信号路径会保留观测到的正优势。我们已排查这些样本的计算链，尚不能把它升级为“所有性能差异都已排除实现影响”。
此前 LoRA 到 BF16 采样副本的微小更新舍入限制仍需在最终结果中披露。</p>
<p><strong>行动决定：</strong>完成已经批准的有上限配对 pilot 和共同评测，沿用既定停止规则；不加拒答罚分，不靠长度过滤，不改当前 PRM，不扩到 6M。
替换 PRM 的下一步是上述完整组离线准入检查。无可靠评分器时，不能把单纯改优势公式当作数学正确性的修复。</p>

<details><summary>逐项官方源码与审计证据</summary><div class="wide"><table><tr><th>环节</th><th>核实内容</th><th>固定版本来源</th></tr>__SOURCES__</table></div></details>
<p class="note">本地旧文件 hash 检查：__COUNT__ 个全部未变。审计只新增证据与报告。<a href="official_runtime_audit.json">源码审计</a> · <a href="official_failure_audit_receipt.json">本轮验收</a></p>
</main></html>'''
page = page.replace('__ROWS__', rows).replace('__SOURCES__', source_rows).replace('__COUNT__', str(len(protected)))
(HERE/'official_prm_reliability_review.html').write_text(page)
print(json.dumps({'report': str(HERE/'official_prm_reliability_review.html'), 'protected_checked': len(protected), 'changed': changed, 'all_wrong_group_LOO': outcome_checks}, ensure_ascii=False))
