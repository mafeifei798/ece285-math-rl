"""Build a local evidence report; never generate predictions or modify run inputs."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / 'final_evidence'

def read(rel):
    return json.loads((EV / rel).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def table(heads, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+html.escape(str(x))+'</th>' for x in heads) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'

def link(rel, label):
    assert (ROOT / rel).is_file(), rel
    return '<a href="'+html.escape(rel, quote=True)+'">'+html.escape(label)+'</a>'

def main():
    ev = read('eval_audit/verified.json')
    signal = read('signal_audit/analysis.json')
    training = {a:read('training_audit/'+a+'.json') for a in 'AB'}
    assert ev['passed'] and signal['passed'] and all(v['passed'] for v in training.values())
    part1 = read('case_review/early_box_losses_part1_review.json')
    part2 = read('case_review/early_box_losses_part2_review.json')
    gains = read('case_review/early_box_gains_control_review.json')
    for doc, src in [(part1,'early_box_losses_part1.json'), (gains,'early_box_gains_control.json')]:
        assert doc['source_sha256'] == sha(EV/'case_review'/src)
    assert part2['source_hashes']['case_review/early_box_losses_part2.json'] == sha(EV/'case_review/early_box_losses_part2.json')
    loss_counts = Counter(x['classification'] for x in part1['reviews']+part2['cases'])
    gain_counts = Counter(x['classification'] for x in gains['cases'])
    assert loss_counts == Counter(body_correct_stale_box=31,body_math_error=8,ambiguous=5)
    assert gain_counts == Counter(body_correct_stale_box=6,ambiguous=1)
    assert len({x['problem_id'] for x in part1['reviews']+part2['cases']}) == 44
    now = datetime.now(timezone.utc)
    stamp = now.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S PT')
    paired = ev['paired_B_minus_A']
    results = []
    failure_rows = []
    summary = {'created_utc':now.isoformat(),'run_status':'completed',
        'scope':'Corrected-only A/B, seed17, 6M generated completion tokens per arm; no new reward/loss.',
        'all_three_independent_audits_passed':True, 'datasets':{},
        'qualitative_case_review':{'loss_stratum':dict(loss_counts),'gain_stratum':dict(gain_counts),
            'status':'Assistant qualitative review; post-hoc selected stratum; official rewards unchanged.'},
        'limitations':['One training seed; paired question intervals exclude training-seed variability.',
            'No significant advantage for B established, and no equivalence established.',
            'Checked implementation fidelity is not proof that the verifier or PRM has perfect mathematical validity.',
            'No causal attribution of the accuracy gap to PRM misses, normalization or stale boxes.'],
        'no_new_training_or_model_generation':True}
    for dataset, title in [('math500','MATH500'),('gsm8k','GSM8K')]:
        a,b = (ev['arms'][arm]['datasets'][dataset] for arm in 'AB')
        p=paired[dataset]
        lo,hi=p['paired_question_bootstrap_95ci_pp']
        results.append([title,f"{a['correct']}/{a['count']} · {100*a['accuracy']:.2f}%",f"{b['correct']}/{b['count']} · {100*b['accuracy']:.2f}%",f"{p['difference_pp']:+.2f} pp",f"[{lo:+.2f}, {hi:+.2f}]"])
        summary['datasets'][dataset]={'A_correct':a['correct'],'B_correct':b['correct'],'n':a['count'],
            'A_accuracy':a['accuracy'],'B_accuracy':b['accuracy'],'B_minus_A_pp':p['difference_pp'],
            'paired_question_bootstrap_95ci_pp':[lo,hi],'A_only_correct':p['A_only_correct'],'B_only_correct':p['B_only_correct']}
        for arm,d in [('A',a),('B',b)]:
            f=d['failure_types']
            failure_rows.append([title,arm,f['correct'],f['complete_incorrect'],f['unparsable'],f['truncated'],f"{d['length']['mean']:.1f}"])
    group_rows=[]
    for key,title in [('mixed','混合结果'),('all_correct','全对'),('all_wrong','全错')]:
        g=signal['B']['groups'][key]
        group_rows.append([title,g['counts']['groups'],f"{100*g['group_fraction']:.2f}%",f"{g['metrics']['absolute_actual_advantage']['mean']:.3f}"])
    title='只纠错 A/B：训练、评测与独立核验已完成'
    text = f'''{title}
报告时间：{stamp}。控制器与 GPU 完成状态以 07:17 PT 留存快照为准。

结论
本轮固定 7B PRM 的 B 没有显示出优于纯结果奖励 A 的收益。MATH500 基本持平，GSM8K 点估计低约 2.05 个百分点。两项配对区间均包含 0；不能据此断言二者等效，也不能用单 seed 推广成“PRM 没用”。

正式结果（greedy、同一固定验答器、各自固定最终 checkpoint）
MATH500：A 270/500=54.00%；B 269/500=53.80%；B−A −0.20 pp；95%配对题目 bootstrap 区间 [−4.00,+3.60] pp。
GSM8K：A 884/1319=67.02%；B 857/1319=64.97%；B−A −2.05 pp；区间 [−4.25,+0.08] pp。
10,000 次配对 bootstrap，bootstrap seed285；训练只有 seed17，不包含跨训练 seed 不确定性。

这轮已经实际验收了什么
A 6,022,435 completion tokens / 284 更新；B 6,008,457 / 280 更新。按首个完整更新达到6M停止；两者不同更新数符合相同生成 token 预算。初始 adapter 相同，共同1,120个题组顺序相同。
训练账本审计、奖励/advantage审计、评测审计全部通过。全部3,638条预测重跑冻结验答器，答案/状态/reward均一致。原reward和loss保留，没有二次advantage归一化。7,304条实际PRM评分记录的文本分段/跨度/hash核验通过。历史代码、结果和初始化记录保留。
这说明已查项目符合冻结方案；不是证明验答规则覆盖一切数学等价答案，也不是证明PRM评分正确。

B 的问题现在分清了三层
1. 正式答案与后续正文不一致。GSM8K中A对B错120题、B对A错93题；其余是同对或同错。完整可解析的错答对应89次损失、50次收益。其中错误侧的最后答案框出现在前25%文本且后续至少100字符的有44次损失、7次收益。逐条核对这51条：损失侧31条后文得出并接受正确数值但旧框未更新，8条数学仍错，5条矛盾未解；反方向6条旧框未更新、1条矛盾未解。它是本轮差异集里的明显的输出现象，不是因果证明；不据此重写准确率。31条中的一个边界样例只以正确的模拟代码输出结束，报告单列此限制。
2. PRM确实漏过数学错误。针对训练中分数最高的10条outcome=0回答核对，4条有真正数学错误，4条是boxing/单位/回答对象不符，2条是合法非唯一或等价表示未被固定验答器接受。不能把所有outcome=0都当过程错。这10条是刻意选的异常样本，不估计总体误判率。其中只有1条实际advantage为正。
3. 组内归一化仍会放大很弱的相对差异。全对/全错共242/1,120组（21.61%），贡献19.92%的实际advantage绝对值总和；这不是梯度贡献比例。真实全错组1060中PRM最低步分全部只有0.017–0.054，却有回答获得+1.070的advantage。该回答有无效图索引、虚构执行结果和答案不一致。这是冻结算法的行为，不是新发现的代码归一化bug；也不能单凭该例证明其导致测试下降。

不能漏掉的反证
B的GSM8K不可解析输出反而减少（89→53），所以不能说“B只是没写可提取答案”。B完整但盒内答案错误314→368，截断32→41。训练PRM分没有清晰单调上升趋势，不能写成已证实奖励投机。平均分段数几乎相同（A11.917，B11.899），没有整体少分段塌缩证据。
GSM8K配对净−27题可描述为：完整错答相关−39，解析失败相关+23，截断相关−11；类别互斥，但仍不等于数学能力的因果分解。

预算与局限
B另消耗5,281,779 PRM输入token（含padding7,724,200），评分耗时593.35秒，包含分词/搬运/打分等，不是纯GPU forward。相同actor token预算不等于相同总算力。
本报告的数学逐题审查由助手完成，附原句、位置与可复算算式；不是人工金标准。所有定向抽样和新增结构分层均为事后诊断。测试输出没有用于改训练、调参或选checkpoint。向量多解/等价表示的验答覆盖仍有限，正式结果保持原协议。

明确的下一步（未启动）
保留当前A/B为正式基线；先不换PRM，也不启动完整C/D。下一项最小验证应复用现有A/B checkpoint，在固定、隔离的验证题集上，用原提示与“先推导，末尾只给一个最终答案框”的统一提示做2×2推理对照，检查正文与答案框一致性和准确率。生成预算另行明确；当前授权不包含这批新推理，因此未运行。若一致性改善而B仍差，再针对已证实的PRM数学漏判做适配检查。任何提示或验答协议变化须单列版本，不能冒充本轮只纠错结果。
对于原计划的C/D：仅用outcome=0作失败监督会混入“正文算对但框错/等价答案被拒”的负标签，适配前应把这一风险纳入检查；不能假设C/D必然修好B。

文件
主报告：review.html
独立评测：final_evidence/eval_audit/verified.json
训练审计：final_evidence/training_audit/A.json、B.json
信号审计：final_evidence/signal_audit/analysis.json
逐题说明：final_evidence/case_review/ 下各 review.txt / review.json
所有本次分析为已保存输出上的CPU离线检查，没有新增训练或模型生成。
'''
    # Preserve the last published progress page before replacing the live report.
    for name in ['review.html','REPORT_latest.txt']:
        old=ROOT/name
        preserved=ROOT/(Path(name).stem+'_pre_final_20261022'+Path(name).suffix)
        if old.exists() and not preserved.exists():
            preserved.write_bytes(old.read_bytes())
    (ROOT/'REPORT_latest.txt').write_text(text)
    (EV/'final_report.txt').write_text(text)
    (EV/'results_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    section1='''<p class="lead">这轮已经跑完并逐条核验。<strong>B 没有展示出超过 A 的收益</strong>：MATH500 几乎相同，GSM8K 点估计更低。进一步检查找到了答案框与正文脱节、真实 PRM 漏判，以及全同结果组的归一化效应，三者不能混成一个“实现 bug”。</p>'''
    body=f'''<header><span class="eyebrow">ECE285 · corrected-only · seed17</span><h1>{title}</h1><p class="muted">更新 {stamp} · 已完成：A/B 训练 → 统一评测 → 三套独立 CPU 审计 → 真实回答核对</p>{section1}</header>
    <section><h2>正式结果</h2>{table(['评测集','A：结果奖励','B：结果 + 固定7B PRM','B − A','95%配对区间'],results)}
    <p class="note">pp = 百分点。两个区间都包含0：未建立 B 的优势，也未建立 A/B 等效。10,000次同题 bootstrap；只有1个训练seed，区间不覆盖训练随机性。</p>
    <img class="plot" src="final_evidence/figures/eval_differences.svg" alt="MATH500和GSM8K的B减A准确率差及95%配对区间，均跨过零。"></section>
    <section><h2>实现核验：已通过，但边界清楚</h2><div class="cards"><article><strong>3,638 / 3,638</strong><span>评测答案、状态与reward独立重算一致</span></article><article><strong>7,304</strong><span>实际PRM记录的分段、文本跨度与hash核验通过</span></article><article><strong>原始 GRPO</strong><span>reward/loss未改；advantage没有再次归一化</span></article></div>
    <p>共同初始化与前1,120题组顺序相同；A 6,022,435 token / 284更新，B 6,008,457 token / 280更新，均在首个完整更新达到6M时停止。不同更新数符合生成token预算口径。旧代码和旧结果保留。</p>
    <p class="note">通过的是冻结实现与账本的一致性检查。PRM是否判断正确、验答器能否覆盖所有数学等价答案，仍须看真实案例；没有宣称“所有数学层面的实现问题已排除”。</p></section>
    <section><h2>发现一：模型算到了正确答案，却没更新答案框</h2><p>GSM8K里 A对B错120题、B对A错93题。对其中完整可解析的错答，分别有44条与7条满足“最后答案框很早出现，后面仍继续推导”。我们检查了这51条的完整正文，而不是只搜索正确数字。</p>
    {table(['这一事后筛选子集','后文接受正确数值、旧框未更新','正文仍错','矛盾未解'],[['A对B错中的44条',31,8,5],['B对A错中的7条',6,0,1]])}
    <div class="example"><b>真实例子 · GSM8K / 893 · B</b><p>题目：时薪10美元增加2美元，每天8小时、每周5天。开头答案框是 <code>400</code>，后文正确计算 <code>12 × 40 = 480</code>，并写出“Mark makes $480 per week”。但没更新答案框，固定验答器仍判错。</p></div>
    <p>固定验答器按最后一个答案框取值，在这里遵守了协议。把正文新数字拿来替换答案，会改动评分规则，所以<strong>正式准确率没有重算成“修正版”</strong>。两组都存在这种行为，不能把它一概归因为 PRM。</p>
    <p class="note">结构阈值从前10%改到25%或50%时，这里的44/7计数不变。语义分类仍是助手定性审查，非独立人工标注；31条中有1条仅以正确模拟代码输出结束，未写完整结论句。这个类别不代表全过程正确或只有格式错误；其余未解决的错误/矛盾未被并入此类。这些计数不能推广成总体错误率或因果贡献。</p>
    <p>{link('final_evidence/case_review/early_box_losses_part1_review.txt','44条：前22条证据')} · {link('final_evidence/case_review/early_box_losses_part2_review.txt','44条：后22条证据')} · {link('final_evidence/case_review/early_box_gains_control_review.json','反方向7条')}</p></section>
    <section><h2>发现二：PRM有真漏判，但“高分错误”不一定被强化</h2><p>从B训练记录中取最低步骤分最高的10条 outcome=0 回答：4条有真实数学错误，4条是答案框、单位或回答对象不符，2条是合法多解/等价表示未被固定验答器接受。仅1条的实际 advantage 为正。</p>
    <div class="example"><b>真实漏判 · 训练行937</b><p><code>4 × 6 + 8 × 3 − 28 / 2 = 34</code>，模型伪造执行结果 <code>36</code>，错误步骤仍得 PRM 0.999718。但整条回答实际 advantage 是 −1.072，它并没有因高PRM分而获得正advantage。</p></div>
    <p>因此，“答错但PRM高”必须继续看推导、答案协议与组内实际优势，不能直接当成数学漏判或有害强化。这里的10条是挑选异常用的，不用于估计总体误判率。</p>
    <p>{link('final_evidence/case_review/prm_high_wrong_review.txt','10条高分 outcome=0 的逐步证据')}</p></section>
    <section><h2>发现三：绝对低分也能变成较大的正 advantage</h2>{table(['B训练组类型','组数','占比','平均 |advantage|'],group_rows)}
    <p>全对与全错合计21.61%的组，贡献19.92%的实际advantage绝对值总和。它们的结果奖励没有组内差异，组内奖励产生的advantage由过程分及其评分资格决定；0.2不能保证归一化后的影响很小。</p>
    <div class="example"><b>真实例子 · 全错组1060</b><p>8条回答的PRM最低步骤分只有0.017–0.054，最高的那条仍得到 <strong>+1.070</strong> advantage。这条回答有无效图索引、虚构运行结果和前后答案不一致；独立有限枚举核实答案为54。</p></div>
    <p class="note">这是原算法的相对比较机制，不能在baseline里当bug悄悄改掉。正advantage的错答也可能包含有用进展；这个反例只证实风险，不能单独证明它导致测试分数下降。19.92%是advantage绝对值之和占比，不是梯度贡献占比。</p>
    <p>{link('final_evidence/case_review/low_score_allwrong_group.json','真实8条回答及优势')} · {link('final_evidence/case_review/graph_group_independent_check.json','独立图计数验证')} · {link('final_evidence/signal_audit/FINDINGS.txt','完整信号审计')}</p></section>
    <section><h2>其余检查与成本</h2>{table(['评测集','组','答对','完整但框内答错','无法解析','截断','平均生成token'],failure_rows)}
    <p>GSM8K里B的无法解析输出减少（89→53），但完整错答和截断增加。配对净少27题，对应完整错答−39、解析问题+23、截断−11。这是输出类别分解，不能直接称为数学能力分解。</p>
    <p>训练期平均文本分段数 A11.917、B11.899；未见整体少分段塌缩。分箱PRM均分没有清晰单调上升，不能宣称已证明奖励投机。</p>
    <p>B额外使用5,281,779 PRM输入token（padding后7,724,200），评分耗时593.35秒，包含分词、搬运和提取等。<strong>相同actor生成预算不等于相同总算力</strong>；当前顺序单次运行的墙钟差异不构成速度优势结论。</p></section>
    <section class="next"><h2>下一步：先验证输出一致性，不再盲换PRM</h2><p><strong>保留这轮正式A/B结果，先不跑完整C/D。</strong>最小后续是复用这两个checkpoint，在固定独立验证集上做2×2推理对照：A/B × 原提示/统一的“先推导、末尾一个最终答案框”提示。两边采用相同greedy配置，分别报告准确率、答案框与正文一致性及截断。</p>
    <p>它只验证一个具体疑点：减少答案与正文脱节后，B相对A的差距是否仍在。它不能证明训练期PRM的因果效应；也不保证提示词能修好。先固定验证集与预算，且不拿本次测试题调提示。若差距仍在，再针对真实PRM漏判推进适配。</p>
    <p>原C/D的失败监督也要承认：outcome=0里混有数学正确但答案框错误、合法等价答案被拒的样本，不能把它们自动当成所有步骤都错。</p>
    <p class="note">此项需要额外模型生成，超出本次CPU离线检查范围，<strong>尚未启动</strong>。没有新模型、C/D、额外seed或新训练。提示/验答变化须独立版本，不能替换本轮baseline。</p></section>
    <footer><h2>可追溯证据</h2><p>{link('REPORT_latest.txt','中文完整报告')} · {link('final_evidence/results_summary.json','汇总JSON')} · {link('final_evidence/eval_audit/verified.json','评测逐条验收')} · {link('final_evidence/training_audit/A.json','A训练验收')} · {link('final_evidence/training_audit/B.json','B训练验收')} · {link('final_evidence/signal_audit/analysis.json','信号审计JSON')}</p>
    <p>{link('final_evidence/case_review/math500_review.txt','MATH500定向10对')} · {link('final_evidence/case_review/gsm8k_review.txt','GSM8K定向10对')} · {link('final_evidence/transfer_verified.json','原始文件传输核验')} · {link('snapshots/20261022T141700664265Z.json','07:17控制器完成快照')} · {link('review_launch.html','原始启动记录')}</p>
    <p class="muted">案例审查为助手定性分析，附原句、位置与独立计算；测试预测、官方reward与固定checkpoint均未改动。所有增量分析在CPU上读取现有输出完成。</p></footer>'''
    css='''*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#1d2e40;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1080px;margin:32px auto;padding:42px 48px;background:#fff;border:1px solid #dce4ed;border-radius:18px}header{border-bottom:1px solid #dce4ed;padding-bottom:24px}.eyebrow{font-size:12px;letter-spacing:.14em;color:#47718e;text-transform:uppercase}h1{font-size:30px;line-height:1.4;letter-spacing:-.03em;margin:12px 0}h2{font-size:21px;margin:0 0 14px;line-height:1.5}section{padding:28px 0;border-bottom:1px solid #e4eaf0}p{margin:12px 0}.lead{font-size:18px}.muted,.note{color:#647489;font-size:14px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #e1e7ef;white-space:nowrap}th{background:#f0f4f8;font-weight:600}.plot{width:100%;height:auto;margin:14px 0 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.cards article{background:#f1f6fa;border-radius:10px;padding:16px}.cards strong{display:block;font-size:23px;color:#215b82}.cards span{font-size:14px;line-height:1.6;display:block;margin-top:6px}.example{border-left:3px solid #437da3;background:#f5f8fb;padding:14px 18px;margin:18px 0}.example p{margin:7px 0 0}code{font-family:ui-monospace,SFMono-Regular,monospace;background:#eaf0f5;padding:2px 5px;border-radius:4px;font-size:14px}a{color:#1e648e;text-underline-offset:3px}.next{background:#f4f8f5;margin:24px -20px 0;padding:24px 20px;border:1px solid #d7e5dc;border-radius:12px}footer{padding-top:26px}footer a{display:inline-block;margin-right:7px}strong{font-weight:650}@media(max-width:720px){main{margin:0;padding:24px 18px;border:0;border-radius:0}h1{font-size:25px}.cards{grid-template-columns:1fr}.next{margin:20px 0 0}}@media print{body{background:white}main{border:0;margin:0;padding:0}section{break-inside:avoid}.plot{max-height:230px}a{color:inherit}}'''
    (ROOT/'review.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
    state=json.loads((ROOT/'follow_through_state.json').read_text())
    state.update({'postprocessing_completed':True,'postprocessing_completed_utc':now.isoformat(),
        'health':'All authorized training/evaluation and substantive CPU verification completed. No experiment process remains in last live completion snapshot.',
        'independent_audits':{'training_A':'passed','training_B':'passed','evaluation':'passed','signals':'passed'},
        'final_report':str(ROOT/'review.html'),'results_summary':str(EV/'results_summary.json'),
        'next_action':'Only final report QA remains; then stay quiet unless material change. At09:00PT give bounded final summary. No new training or model generation authorized.',
        'posthoc_semantic_reviews':{'B_loss_stratum':dict(loss_counts),'A_loss_stratum':dict(gain_counts)},
        'official_evaluation_unchanged':True})
    (ROOT/'follow_through_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(ROOT/'review.html'),'counts':dict(loss_counts),'created':stamp},ensure_ascii=False))

if __name__=='__main__':main()
