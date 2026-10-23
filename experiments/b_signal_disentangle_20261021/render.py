"""Write the ranking-versus-amplification report without changing older audits."""
import csv
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'execution/b_signal_disentangle_20261021'
read=lambda name:json.loads((OUT/name).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table(headers,data):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table></div>'


def main():
    summary=read('summary.json');review=read('reviewed_groups.json');qa=read('independent_QA.json')
    assert qa['passed']
    byrow={r['source_row']:r for r in review}
    csvrows={int(r['source_row']):r for r in csv.DictReader((OUT/'per_response.csv').open())}
    candidate={'model':'sail/ActPRM-X','official_source':'https://huggingface.co/sail/ActPRM-X',
        'official_author_reported_ProcessBench':76.0,'base':'Qwen/Qwen2.5-Math-PRM-7B',
        'benchmark_scope':'Authors reported benchmark, not measured on our rollouts or in RL.',
        'weights_downloaded':False,'loaded_or_scored_in_this_audit':False,
        'decision':'A reasonable one-candidate offline comparison if remaining ranking errors justify it; no automatic model replacement or RL launch.',
        'integration':'Official native inference output semantics require inspection; do not assume it is a drop-in logits replacement.'}
    (OUT/'candidate_prm.json').write_text(json.dumps(candidate,ensure_ascii=False,indent=2)+'\n')
    metric=[]
    for key,name in [('repaired_original_B','修复后的原B目标'),('lambda_0p02_diagnostic_only','仅把权重降到0.02'),
        ('homogeneous_only_bounded','仅限制同结果组'),('bounded_residual_recommended','推荐：结果优势+有界过程项')]:
        x=summary['signal_stats'][key]
        metric.append([name,f"{x['all_correct']['mean_abs_advantage']:.3f}",f"{x['all_wrong']['mean_abs_advantage']:.3f}",
            x['mixed']['correct_negative_rows'],x['mixed']['wrong_positive_rows']])
    rank=[]
    for k,name in [('896','转盘概率（新增随机组）'),('4984','公斤换算（新增随机组）'),
        ('6344','奇数平方差（已知问题组）'),('8744','周末降雨（已知问题组）')]:
        v=summary['within_question_rank_reviews'][k]
        rank.append([name,v['valid'],v['explicit_error'],v['proof_gap_or_terminology_excluded'],
            f"{v['inversions']}/{v['pairs']}" if v['pairs'] else '无可比较的正误对'])
    cases=[]
    for i,name in [(4987,'完整正确的250g换算'),(899,'把1/3≠4/12写成真命题，随后纠正'),
        (6350,'答案对但证明有错'),(8745,'完整正确的独立事件乘法')]:
        r=byrow[i];c=csvrows[i]
        cases.append([i,name,f"{r['min_score']:.4f}",f"{r['original_advantage']:+.4f}",
            f"{float(c['lambda_0p02_diagnostic_only']):+.4f}",f"{r['bounded_advantage']:+.4f}"])
    text='''B不如A：排序问题与优势放大的离线拆分

结论
两类问题同时存在，但不是同一个机制。PRM在若干回答上有误罚/错排；组内中心化决定相对正负，除以正标准差不会颠倒排序，却会把不可靠的分数差放大。现有离线数据不能把B-A的准确率差按这两类原因做因果分摊。

完整八回答组复核
补看32条完整回答：两个从其余31个修复后全对组中用random.Random(286)均匀选择的新组（896、4984），再加两个已有问题组（6344、8744）。单个Codex助手审阅，部分旧案例分数已知，不是独立人工金标准。
转盘组：7条全程正确，1条明确写出错误不等式1/3≠4/12后恢复。该错误回答min0.947，高过6条正确回答。它可能是符号笔误，不能据此推断PRM整体数学能力差。
公斤换算组：8条都正确，250g=0.25kg，费用5500。一条完整正确回答min0.454，实际advantage=-2.450；有界修复后=-0.090。数学正确性标签不支持这么强的区分，但单条负优势本身也不证明参数更新有害。
降雨组：4条全程正确、3条含明确错误后恢复、1条措辞歧义。排除歧义后12个正误对全部排对。有用的同结果组过程排序确实存在，不能把全部同结果组简单清零当作默认修法。
奇数平方差组：6条明确有错，2条证明不完整，没有可确认完整正确的对照。因此之前用B14的正advantage直接证明PRM错排是不严谨的；这个组能展示低分仍被相对强化，但不能估计正误排序准确率。

全量幅度诊断
9056条回答、1132个原八回答组。33个全对组、206个全错组、893个混合组。只在非零信号组计算增益：全对33组中位增益16.29倍；全错204组中位77.53倍，81组超过100倍。这里是相对未归一化过程残差的优势幅度倍数，不是参数梯度贡献，也不是“放大了多少已确认错误”的比例。
把PRM系数0.2改为0.02：全对平均|A|0.796→0.781，全错0.734→0.669。全同结果组的标准差随系数一起缩小，所以单独降权重不能可靠控制影响。

具体修法（尚未接入训练）
A_i = (y_i-mean(y))/(sample_std(y)+1e-4) + 0.2*(m_i-mean(m))。
沿用已有PRM、min聚合、固定八回答组和单位/CRLF修复；m为最低步骤概率，无效未评分回答沿用0占位。只替换传入GRPO clipped loss的sequence advantage，ratio clipping和KL保持原设置，之后不能再对这个优势做组内标准差归一化。
过程项绝对值在group8下最多0.175。混合二值组中，结果GRPO优势绝对值至少约0.35345，因此过程附加项不会逆转结果奖励的正负方向。全量回放中，混合组6条答对负优势、4条答错正优势都消失；这些计数描述结果标签方向，不是完整数学质量判定。
全对平均|A|0.796→0.046；全错0.734→0.014。公斤换算的过强惩罚减轻，但仍为负；转盘的错误回答仍被排高，只是正优势由0.731降到0.028。这个修法控制放大，不修正PRM排名。降雨组正确的正误排序仍被保留。
附加机制对照分别只限制同结果组、保持旧结果项分母而限制过程项、或清零同结果组，用于拆分机制；它们不是多轮训练试错。本轮没有新训练、模型加载、PRM评分或参数更新。

是否再换PRM
当前更值得先控制优势放大。H4换成Qwen7B后，旧B固定随机审阅中的首错定位由0/18到12/18，说明评分质量已经改善，但这不是当前策略上的总体组内排序指标。
如要再比一个，候选是sail/ActPRM-X。官方说明它基于当前Qwen PRM继续训练，报告ProcessBench76.0%。这是作者基准结果，不保证我们轨迹上的排序或RL收益。本轮只核实资料，未下载或调用；不能因为公开榜单更高就直接替换并开训。先用同一固定轨迹、原生输入、已有过程标签比较误罚和同题排序。

边界
现有rollout可验证具体误判、排序、优势符号/幅度和理论保护；不能证明修复后准确率一定超过A，也不能给B-A下降原因分配因果百分比。原始A/B与v3结果全部保留。本报告既不推进C/D，也不授权新训练。
'''
    (OUT/'REPORT.txt').write_text(text)
    css="body{margin:0;background:#f4f6f8;color:#26384e;font:15px/1.75 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1180px;margin:auto;padding:30px 22px 70px}h1{font-size:30px;line-height:1.3}h2{font-size:21px;margin-top:30px}.card,details{background:white;border:1px solid #dce2e9;border-radius:12px;padding:20px;margin:16px 0}.lead{background:#eaf4f0;border-left:5px solid #258875}.warn{background:#fff5df;border-left:5px solid #b98838}.muted{color:#6c7788;font-size:13px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:9px 11px;border-bottom:1px solid #e5e9ef;text-align:left;vertical-align:top}th{background:#f0f3f7;white-space:nowrap}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:13px;border-radius:8px;font-size:13px}summary{cursor:pointer;font-weight:650}a{color:#276aa1}@media(max-width:700px){main{padding:18px 12px}.card,details{padding:13px}h1{font-size:25px}}"
    parts=["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>B诊断：排序与优势放大</title><style>"+css+"</style><body><main>",
        "<h1>B不如A：先分清排序与信号放大</h1><p class='muted'>9056条现有回答 · 1132个原题组 · 32条完整组数学复核 · 纯CPU离线分析</p>",
        "<div class='card lead'><b>推荐：先修advantage构造，保留7B PRM做受控比较</b><p>已发现PRM误罚/局部错排，也确认全同结果组的标准差归一化会显著放大分数差。现有证据支持先限制不可靠过程项的幅度。这个修复无法修正原始排序，也没有证明新策略准确率超过A。</p></div>",
        "<div class='card warn'><b>对之前判断的纠正</b><p>B14“证明有错但优势为正”不能单独证明PRM错排。补看全部伙伴后，本组没有可确认完整正确的对照。中心化本来就会相对强化组内较高分的回答；归一化不会颠倒顺序。需要把评分误判、相对正负和幅度放大分开。</p></div>",
        "<h2>1 · 完整题组中的数学排序</h2><div class='card'>",table(['题组','全程正确','明确错误','证明缺口/措辞歧义','错排对/可比较对'],rank),
        "<p>转盘组的错误是符号不等式，随后推导恢复，不能以此估计普遍数学能力。降雨组正误排序全部正确，说明同结果组也有有用信号；默认清零全部同结果组会丢失它。</p><p class='muted'>两组新增随机样本、两组已知问题样本；仅32条，单助手且部分案例分数已知。配对不是独立样本，不能把6/7等比例外推到总体。</p></div>",
        "<h2>2 · 同一评分怎样变成过强的信号</h2><div class='card'>",
        "<pre>旧B：A = center(y + 0.2 m) / (std(y + 0.2 m) + 1e-4)\n全同结果组：结果项中心化为0，只剩PRM过程差异\n放大倍数：1 / (std(R) + 1e-4)</pre>",
        "<p>非零信号全对组的放大中位数<b>16.29倍</b>；全错组<b>77.53倍</b>，其中81组超过100倍。零信号组排除，避免把零乘很大的系数误称为强信号。</p><p>权重0.2降到0.02，全对平均|A|仍为0.781，原来是0.796。标准差同步变小，单独降权重没有把过程影响降十倍。</p><p class='muted'>这是优势幅度的数学分解，不是实际参数梯度占比，也不意味着所有被放大的过程排序都是错的。</p></div>",
        "<h2>3 · 固定PRM分数的离线修复对照</h2><div class='card'>",table(['处理','全对平均|A|','全错平均|A|','混合组：答对负A','混合组：答错正A'],metric),
        "<p class='muted'>混合组两列只描述结果奖励方向；答错仍可能有有用步骤，答对仍可能过程有错。全部原伙伴保留，没有删除无效输出或挑选有利子集。</p></div>",
        "<h2>4 · 明确推荐的修法</h2><div class='card lead'><pre>A_i = (y_i − mean(y)) / (sample_std(y) + 1e-4)\n      + 0.2 × (m_i − mean(m))</pre>",
        "<p>保留纯结果GRPO的优势，把PRM作为不按自身标准差放大的附加项。group8下过程项绝对值≤0.175，不能逆转混合组的结果优势符号。有用的同结果组排序继续保留。</p><p>实现位于bounded_signal.py，仅构造sequence advantage，未安装到trainer。使用时保留原ratio clipping和KL，不能再对返回优势做一次组内标准差归一化。没有改PRM、min聚合或步骤划分。</p></div>",
        "<h2>5 · 具体案例：控制幅度不等于纠正排名</h2><div class='card'>",table(['源行','数学复核','PRM min','原B优势','仅权重0.02','有界过程项'],cases),
        "<p>正确的公斤换算仍有负优势，但从−2.45减轻至−0.09；转盘错误符号回答仍排高，但正优势从0.731降至0.028。因此不能宣称数学误判已消失。</p></div>",
        "<h2>6 · 要不要再换PRM</h2><div class='card'><p>目前优先修优势构造。一个可后续离线比较的候选是<a href='https://huggingface.co/sail/ActPRM-X'>sail/ActPRM-X</a>：官方说明基于当前Qwen PRM继续训练，报告ProcessBench76.0%。这是作者基准，不是我们的轨迹或RL结果。</p><p>本轮未下载或调用新模型。更高公开分数不能消除现有归一化放大机制；换模型应看同一固定轨迹上的数学误罚和同题排序。现有助手审阅仅作诊断，不能当独立模型选择金标准。</p></div>",
        "<h2>7 · 32条完整组复核原文</h2><p class='muted'>V：全程数学正确；E：含明确错误（含后来恢复）；G：证明缺口；T：措辞歧义。G/T不计入主要正误配对。</p>"
    ]
    for r in review:
        parts += [f"<details id='row{r['source_row']}'><summary>row {r['source_row']} · {r['label']} · min {r['min_score']:.4f} · 原B优势 {r['original_advantage']:+.4f}</summary>",
            '<pre>'+html.escape(r['question'])+'</pre><p>'+html.escape(r['reason'])+'</p>',
            table(['阶段','原文','PRM分'],[[i+1,s['text'],f"{s['score']:.6f}"] for i,s in enumerate(r['steps'])]),'</details>']
    parts += ["<div class='card'><b>证据与复现</b><p><a href='REPORT.txt'>文字报告</a> · <a href='summary.json'>全量诊断统计</a> · <a href='per_response.csv'>逐回答对照</a> · <a href='reviewed_groups.json'>完整组数学复核</a> · <a href='group_details.json'>每组放大倍数</a> · <a href='independent_QA.json'>独立验证</a> · <a href='candidate_prm.json'>替代评分器核实</a></p><p class='muted'>独立验证覆盖65536组二值结果/极端概率组合，以及全部9056条实际回答；混合组结果符号无翻转，原伙伴完整保留。跨平台FP32与独立FP64最大差已记录，不把数值末位差异视为训练错误。无模型加载、无新推理采样、无参数更新。旧A/B与v3文件哈希未变。</p></div></main></body></html>"]
    (OUT/'review.html').write_text('\n'.join(parts))
    receipt={'complete':True,'no_training':True,'new_PRM_loaded':False,'PRM_inference_calls':0,
        'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'artifact_sha256':{n:sha(OUT/n) for n in ['REPORT.txt','review.html','summary.json','per_response.csv','reviewed_groups.json','group_details.json','independent_QA.json','candidate_prm.json','input_manifest.json']}}
    (OUT/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(OUT/'review.html')


if __name__=='__main__':main()
