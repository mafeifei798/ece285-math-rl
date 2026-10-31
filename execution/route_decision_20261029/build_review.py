"""Build a local decision report from saved evidence. Never imports a training runtime."""
from pathlib import Path
import json, hashlib, html

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
current=json.loads((HERE/'completed_pilot_independent_check.json').read_text())
snapshot=json.loads((HERE.parent/'c_bounded_triplet_seed17_1m_20261029/latest_snapshot.json').read_text())
assert current['passed'] and snapshot['comparison']['passed']
original=snapshot['comparison']

decision={
 'status':'ANALYSIS_ONLY_NO_NEW_TRAINING',
 'recommendation':'One budget-limited test of frozen-C process-signal scaling before considering a route change.',
 'resources_assumed':'one shared A100; existing local configuration uses an 80GiB A100',
 'current_triplet_complete':True,
 'new_training_started':False,'new_model_calls':0,'remote_mutations':0,
 'proposed_stage1_actor_generated_tokens':2000000,
 'optional_stage2_actor_generated_tokens':6000000,
 'all_future_training_requires_explicit_budget_authorization':True,
 'PRM_input_tokens_and_evaluation_generation_are_additional':True,
 'full_official_PAPO_reproduction':False,
 'new_algorithm_priority_claim':False,
 'stage1_arms':[
   {'arm':'A','budget':1000000,'state':'existing_completed'},
   {'arm':'PAPO-standard-controlled','coefficient':1.0,'budget':1000000,'state':'proposed_not_started'},
   {'arm':'PAPO-magnitude-matched','coefficient_candidate':0.052956140041351324,'budget':1000000,'state':'proposed_not_started'},
   {'arm':'C-bounded','coefficient':0.2,'budget':1000000,'state':'existing_completed'}],
 'implementation_note':'V3 currently has CPU acceptance only. Its old proposal used PAPO coefficient0.2; this decision proposes comparing default1.0 and a training-only calibrated coefficient. No source/protocol/launcher was changed to execute this proposal.',
 'small_effects_or_nonsignificance_do_not_establish_equivalence':True,
}
(HERE/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')

rows=[]
for arm,label in [('initial','初始模型'),('A','A：结果奖励'),('C_mix','C-mix：直接混合'),('C_bounded','C-bounded：保守过程项')]:
 d=current['arms'][arm]; train=d.get('training',{}).get('accounting',{})
 tok=train.get('generated_completion_tokens')
 rows.append(f'<tr><td>{label}</td><td>{tok:,}</td><td>{d["correct"]:.0f}/500</td><td>{d["accuracy"]:.1%}</td><td>{train.get("loss_wrong_positive","—")}</td></tr>' if tok else
             f'<tr><td>{label}</td><td>无训练</td><td>{d["correct"]:.0f}/500</td><td>{d["accuracy"]:.1%}</td><td>—</td></tr>')
comparison_rows=''.join(rows)
alternatives=[
 ('冻结C＋保守过程项','已有本地小试、冻结评分器、独立V3','方法增量小，必须排除全局减权解释','优先做一次有限验证'),
 ('NSR / W-REINFORCE','无PRM，代码改动较小','只用错答和调正负权重已有完整工作；本地1.5B收益未知','显存进一步紧张时的备选'),
 ('DBB历史成功率','无PRM，少rollout统计估计','当前192题均只访问一次；历史机制和折扣已被研究','暂不迁移'),
 ('ThinkTwice自纠错','不需要过程judge或错误位置提示','要改双阶段上下文、训练与评测；需重新核算单卡成本','当前不优先'),
 ('3PO参数探索','探索机制明确，方法技术深','IVON、参数同步、分组及Seq-MIS；迁移明显更大','当前不优先'),
 ('PRIME / 继续D适配','过程奖励学习是明确主题','在线评分器增加成本；当前D已有退化证据','不继续扩展')]
alternative_rows=''.join('<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in row)+'</tr>' for row in alternatives)
sources=json.loads((HERE/'sources.json').read_text())['sources']
source_items=''.join(f'<li><strong>{html.escape(s["id"])}</strong> · <a href="{s.get("paper",s.get("local_report","#"))}">论文 / 审查</a>'+
    (f' · <a href="{s["repository"]}">官方仓库</a>' if 'repository' in s else '')+
    (f'<small>固定代码：{html.escape(s["commit"])}</small>' if 'commit' in s else '')+'</li>' for s in sources)

doc=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>数学推理RL：PRM还是换路线？</title><style>
:root{color-scheme:light;--ink:#172a3a;--muted:#546779;--line:#dde5ec;--blue:#175e8e}*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:var(--ink);font:16px/1.85 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1120px;margin:36px auto;padding:38px 46px;background:white;border-radius:15px;box-shadow:0 5px 25px #17344b09}h1{font-size:30px;line-height:1.4;margin:12px 0}h2{font-size:22px;margin-top:36px}p{margin:14px 0}a{color:var(--blue)}small{display:block;font-size:12px;color:var(--muted);overflow-wrap:anywhere}.meta{color:var(--muted);font-size:13px}.lead{padding:18px 22px;border-left:5px solid var(--blue);background:#edf5fb;border-radius:5px}.note{background:#fff7e7;border-left:4px solid #c79227;padding:13px 18px}.formula{padding:15px 19px;background:#f4f7fa;font:17px/1.8 ui-monospace,monospace;overflow-wrap:anywhere}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:11px 12px}th{background:#eef2f6;white-space:nowrap}ul,ol{padding-left:25px}.links{display:flex;flex-wrap:wrap;gap:10px 24px}code{font-family:ui-monospace,monospace;font-size:.9em;overflow-wrap:anywhere}@media(max-width:700px){main{margin:0;padding:24px 18px;border-radius:0}h1{font-size:25px}}
</style></head><body><main>
<p class="meta">ECE285 · 路线比较与缩减预算 · 2026-10-29 · 一张共享A100</p>
<h1>PRM继续，还是换数学推理RL路线？</h1>
<div class="lead"><strong>推荐：保留冻结C，把当前方案收缩为一次有预算上限的过程信号验证。</strong><br>先补两个最有解释力的PAPO式对照，新增约200万训练生成token即可作下一步预算决定。取消D、在线PRM、新评分器和长预算扩张。当前不切换到自纠错或参数扰动训练栈。</div>
<p>理由是最新小试有积极趋势，且迁移到其他路线目前并没有更清晰的小创新。这个推荐不把已有投入当理由，也不承诺有界优势一定胜出。<strong>本轮没有启动任何新训练或模型调用。</strong></p>

<h2>1. 最新结果：有趋势，远未到宣布成功</h2>
<div class="table"><table><thead><tr><th>组别</th><th>实际训练生成token</th><th>验证题正确数</th><th>准确率</th><th>训练错答正优势</th></tr></thead><tbody>__RESULTS__</tbody></table></div>
<p>这是固定的500题<strong>MATH验证集</strong>，不是MATH500测试集。三臂各48次更新，均已完成；各192道独立训练题，每题8条。C-bounded相对A为<strong>+2.6pp</strong>，原配对95%区间为<strong>[−0.2,+5.2]pp</strong>；相对C-mix为+1.2pp，区间[−1.4,+3.8]pp。相对初始模型仅多对4题（+0.8pp）。</p>
<p>只读独立检查重数2000条保存标签、预测hash和共同协议；没有重新判断数学正确性。另一个bootstrap随机数得到的边界略有不同，不能选有利区间。精确双侧McNemar p≈0.072，训练seed仍只有一个。</p>
<div class="note">错答正优势归零是门控的结构保证，PAPO同样能做到；它不是我们独有的提升证据。当前结果支持一次有限验证，不支持直接扩到每组600万/2400万token。</div>
<p>历史6M的只纠错A/B：MATH500为54.0%/53.8%，GSM8K为67.02%/64.97%，未建立B的优势。C的PRM诊断改善不确定；D的诊断显著退化。旧PURE小试在P-PRM续到第40批阶段失败，没有完整最终准确率对照，不能用其不完整结果判定PURE数学表现。</p>

<h2>2. 替代路线认真比较</h2>
<div class="table"><table><thead><tr><th>路线</th><th>可取之处</th><th>当前主要代价 / 新颖性边界</th><th>判断</th></tr></thead><tbody>__ALTERNATIVES__</tbody></table></div>
<p>NSR/W-REINFORCE是最轻的备选，但现成论文已经研究只惩罚错答和正负信号权重。DBB已经研究折扣历史成功率。ThinkTwice已经用相同结果奖励联合训练解题与修改。3PO已经覆盖多种参数扰动与分组。迁移本身都不能算自己的创新。</p>
<p>官方多卡配方不等于单卡改写不可能；这里比较的是<strong>我们还要补多少实现、对照和证据</strong>。NSR核心函数已核对；其他新路线的审查范围与未审查部分见文献清单，未冒称整仓库复现。</p>

<h2>3. 收敛成一个具体项目</h2>
<p><strong>When Does a Frozen PRM Help Small-Model RLVR?</strong><br>固定过程评分器何时能帮助小模型数学RL？过程学习信号的受控研究。</p>
<p>大白话：答案对错负责学会做对题。PRM只在答对的解答之间表达偏好；评分只差一点点时，就先让它只产生一点点影响，检验这种做法能否提供额外的数学收益。</p>
<div class="formula">A = A<sub>out</sub> + 0.2 · y · (m − μ<sub>correct</sub>)</div>
<p>y是固定验答器的0/1结果，m是冻结C的最低步骤概率。结果优势保持原GRPO；正确回答不足两条时过程项为0；合成后不再归一化。步骤切分、PRM输入、EOS、LoRA、KL和clipping都保持共同协议。</p>
<p><strong>它可能有用的前提：</strong>PRM在答对回答内部仍能区分严谨推导与有缺陷的推导。正确答案不等于正确过程，小分差也不自动等于噪声；原始概率分差可能没有良好校准。这些都需要独立评价。</p>
<p><strong>贡献定位：</strong>门控和解耦优势已有PAPO；不归一化过程项也有相关工作。因此不以“一条公式首次提出”作为卖点。项目价值来自在真实小模型、固定判别式PRM和低预算下，把评分错误、信号缩放和数学收益分开测清。</p>

<h2>4. 四组对照，专门排除“只是减小权重”</h2>
<div class="table"><table><thead><tr><th>组别</th><th>设置</th><th>目的</th></tr></thead><tbody>
<tr><td>A</td><td>纯结果优势</td><td>PRM是否真的有额外收益；复用已有同预算结果</td></tr>
<tr><td>PAPO-standard-controlled</td><td>同C、正确子集标准化，过程系数1.0</td><td>最接近方法的受控移植；不是完整论文复现</td></tr>
<tr><td>PAPO-magnitude-matched</td><td>同一标准化过程项，用训练数据固定校准全局系数</td><td>排除收益仅来自整体减弱PRM；现有候选系数约0.053</td></tr>
<tr><td>C-bounded</td><td>原始分差，系数0.2</td><td>我们的小改动；复用已有同预算结果</td></tr>
</tbody></table></div>
<p>幅度匹配只用已冻结的训练回答，不能按测试成绩选权重。平均绝对优势相同不等于参数梯度范数相同；后续策略变化也可能使幅度偏离，须报告实际值。原C-mix作为已有诊断保留，不再重复跑。</p>
<p>PAPO原论文使用rubric生成式grader，我们统一换为同一个冻结C，目的是隔离过程信号构造。当前V3已有CPU接线验收，训练入口仍锁定；这里新增的系数安排只是计划，没有修改运行代码或启动器。</p>

<h2>5. 缩减预算与止损</h2>
<ol><li><strong>最低版：新增约2M。</strong>两个PAPO对照各1M、seed17；A与C-bounded已有结果。先做真实模型无更新联调和协议兼容性检查，再考虑执行。若运行语义不同，不可硬拼旧结果。</li>
<li><strong>可选确认版：再加6M。</strong>锁定较强的PAPO配置后，A/PAPO/C-bounded各1M，两个新seed42/83。新增训练合计约8M，不是每组24M。</li>
<li><strong>只评共同最终checkpoint。</strong>先用验证集作设计决定；最终固定greedy MATH500/GSM8K，不做pass@256、72B judge或反复找最佳checkpoint。历史测试结果已被看过，不能称从未查看的封存测试。</li>
<li><strong>失败就收束。</strong>若幅度匹配PAPO已解释收益，或确认seed没有稳定收益，不再换PRM、试第三个λ或延长某一臂。小试不显著不是等效证明，但可以成为停止投入的预算决定。</li></ol>
<p>PRM输入、评测生成和失败/截断token全部另列账。<strong>8M只是拟议新增训练生成token，绝不是全项目计算总量。</strong>相同token不保证同更新数；两者都报告。</p>
<p>本轮记录中，A/C的训练allocator峰值reserved约20.6/34.1GiB。C每1M额外PRM输入约0.88M、评分约3分钟，训练continuation总时长约32–39分钟；A当时约74分钟。共享负载不同，这些时长不能作为算法速度结论或未来工期保证。共享时仍按原显存准入等待。</p>

<h2>6. 什么结果算有含金量？</h2>
<p>有价值的正结果：在同预算、同评分器下，较A和幅度匹配PAPO都表现出可重复收益，且独立推导检查支持数学质量改善。只有病态优势减少、准确率没有收益时，定位为修复诊断；只有减权PAPO就能达到同样效果时，定位为权重与尺度问题。</p>
<p>建议用现有独立标签诊断，并做60条输出盲审，其中20条双人复核。小人工样本只支持有限结论；训练PRM自己的高分不算独立过程评价。当前没有新人工训练标签。</p>
<div class="lead"><strong>为什么现在选择这一条：</strong>它是当前证据和单卡资源下最容易做出干净结论的一条。算法增量有限，实证项目价值可以做扎实；没有充分理由现在重做自纠错或参数探索训练栈。先花有限预算尝试推翻自己的假设。</div>

<h2>原始资料与核查范围</h2><ul>__SOURCES__</ul>
<p class="links"><a href="PLAN.txt">完整方案与三人分工</a><a href="decision.json">机器可读决策 / 未授权预算</a><a href="completed_pilot_independent_check.json">只读独立核对</a><a href="paired_exact_and_cost_check.json">配对统计</a><a href="legacy_pure_status.json">旧PURE完成范围</a><a href="sources.json">文献与代码版本</a></p>
<p class="meta">实际执行：读取既有receipt与预测标签、CPU统计、论文/官方代码检索、本地报告整理。新增模型调用0、优化器更新0、远端写入0、训练启动0。原A/B/C/D与现有结果保留。</p>
</main></body></html>'''
doc=doc.replace('__RESULTS__',comparison_rows).replace('__ALTERNATIVES__',alternative_rows).replace('__SOURCES__',source_items)
(HERE/'review.html').write_text(doc)
files=['PLAN.txt','sources.json','decision.json','completed_pilot_independent_check.json',
       'paired_exact_and_cost_check.json','verify_completed_pilot.py','build_review.py','review.html','legacy_pure_status.json']
manifest={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files if (HERE/p).exists()}
(HERE/'MANIFEST.json').write_text(json.dumps({'sha256':manifest,'new_training_started':False,'new_model_calls':0},indent=2)+'\n')
print(json.dumps({'report':str(HERE/'review.html'),'files':len(manifest),'new_training_started':False},ensure_ascii=False))
