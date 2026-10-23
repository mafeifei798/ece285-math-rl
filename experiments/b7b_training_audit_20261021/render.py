"""Produce the local audit report, scientific plots, and all20 reviewed cases."""
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import OUT, INPUTS, read, rows, save, sha

NAMES={"A":"A · outcome only", "B_old":"Old B · H4 PRM", "B7":"B′ · Qwen7B PRM"}
COLORS={"A":"#263b60","B_old":"#a76b51","B7":"#168477"}


def pct(x):
    return "—" if x is None else f"{100*x:.2f}%"


def table(headers, body):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in body)+'</tbody></table></div>'


def main():
    stat, native, unit = [read(OUT/name) for name in ["statistics.json","native_replay.json","unit_replay.json"]]
    selected, labels = [read(OUT/name) for name in ["selected_records.json","mathematical_annotations.json"]]
    csv_rows=[r for r in csv.DictReader((OUT/"training_rows.csv").open()) if r["arm"]=="B7"]
    lookup={int(r["source_row"]):r for r in csv_rows}
    all_B=rows(INPUTS/"runs/B_Qwen7B_seed17_6M/rollouts.jsonl")
    base_probe={}
    for arm,name in [("A","A_seed17"),("B7","B_Qwen7B_seed17_6M")]:
        initial=rows(INPUTS/"runs"/name/"rollouts.jsonl")[:32]
        assert all(r["optimizer_step_before_update"]==0 for r in initial)
        base_probe[arm]={"n":len(initial),"optimizer_steps_before_update":[0],
            "with_internal_double_LF":sum("\n\n" in r["completion"].strip() for r in initial),
            "mean_raw_steps":sum(len([s for s in r["completion"].split("\n\n") if s.strip()]) for r in initial)/len(initial)}
    save(OUT/"base_separator_check.json",base_probe)
    group_labels=read(OUT/"group_statistics.json")["B7"]
    reviewed=[]
    for data,ann in zip(selected,labels):
        assert data["sample_id"]==ann["sample_id"]
        meta=lookup[data["source_row"]];r=data["record"]
        assert len(ann["steps"])==len(data["steps"])
        joined=[]
        for i,(text,decision) in enumerate(zip(data["steps"],ann["steps"]),1):
            joined.append({**decision,"text":text,"prm_score":r["step_scores"][i-1] if r["step_scores"] else None})
        base=(data["source_row"]//8)*8
        partners=[]
        for index in range(base,base+8):
            q=lookup[index]
            partners.append({"source_row":index,"outcome":float(q["y"]),"min_score":float(q["min_score"]) if q["scored"]=="True" else None,
                "advantage":float(q["actual_or_replayed_advantage"]),"selected":index==data["source_row"]})
        reviewed.append({**data,"annotation":ann,"steps_with_review":joined,"training_signal":meta,"same_eight_response_group":partners})
    save(OUT/"reviewed_cases.json",reviewed)
    valid=[c for c in reviewed if c["annotation"]["math_status"].startswith("valid")]
    scored_valid=[c for c in valid if c["record"]["valid_for_process_reward"]]
    scored_error=[c for c in reviewed if c["annotation"]["first_error_step"] is not None and c["record"]["valid_for_process_reward"]]
    review_summary={"n":20,"uniform_seed":285,"reviewer":"Codex single-reviewer, not independent human labels",
        "logged_status_counts":dict(Counter(c["record"]["status"] for c in reviewed)),
        "math_status_counts":dict(Counter(c["annotation"]["math_status"] for c in reviewed)),
        "fully_valid_target_reasoning_n":len(valid),"scored_valid_reasoning_n":len(scored_valid),
        "valid_reasoning_any_step_below_0p5":sum(c["record"]["min_score"]<.5 for c in scored_valid),
        "scored_with_known_earlier_errors_n":len(scored_error),
        "all_step_scores_ge_0p5_despite_earlier_error":[c["sample_id"] for c in scored_error if c["record"]["min_score"]>=.5],
        "unresolved_wrong_reasoning_all_step_scores_ge_0p5":[c["sample_id"] for c in scored_error if c["record"]["min_score"]>=.5 and c["annotation"]["math_status"]=="error"],
        "scored_rows":sum(c["record"]["valid_for_process_reward"] for c in reviewed),
        "unscored_rows":sum(not c["record"]["valid_for_process_reward"] for c in reviewed),
        "known_unit_false_negative":"B20",
        "caution":"0.5 is a descriptive threshold, not tuned. Recovery after an early error is separated from a permanently wrong derivation. Rates from3 valid scored examples cannot establish population reliability."}
    save(OUT/"review_summary.json",review_summary)
    alternative_groups={}
    for arm,name in [("A","A_seed17"),("B_old","B_seed17_6M"),("B7","B_Qwen7B_seed17_6M")]:
        data=rows(INPUTS/"runs"/name/"rollouts.jsonl")
        changes={c["source_row"]:c["new"] for c in unit["changes"] if c["arm"]==arm and c["dataset"]=="train"}
        kinds=lambda values:"all_correct" if sum(values)==8 else "all_wrong" if sum(values)==0 else "mixed"
        transition=Counter()
        new_counts=Counter()
        for start in range(0,len(data),8):
            y=[r.get("outcome_reward",r["reward"]) for r in data[start:start+8]]
            corrected=[changes.get(start+i,v) for i,v in enumerate(y)]
            old,new=kinds(y),kinds(corrected)
            transition[old+" -> "+new]+=1;new_counts[new]+=1
        alternative_groups[arm]={"typed_percent_corrected_group_counts":dict(new_counts),"transitions":dict(transition)}
    save(OUT/"unit_group_sensitivity.json",alternative_groups)
    assert native["native_all_rows"]["scored_rows"]==stat["train"]["B7"]["scored_n"]
    assert native["max_score_difference_from_training"]==0 and native["model_frozen"]
    assert len(reviewed)==20 and len({c["source_row"] for c in reviewed})==20
    for rel,receipt in read(OUT/"input_manifest.json")["files"].items():
        assert sha(INPUTS/rel)==receipt["sha256"],rel

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.titlesize":12,"axes.titleweight":"bold","axes.grid":True,"grid.alpha":.16})
    fig,axes=plt.subplots(2,3,figsize=(15,8.1),layout="constrained")
    x=np.arange(6)+.5
    for arm in NAMES:
        p=stat["phases"][arm]
        axes[0,0].plot(x,[100*q["outcome_accuracy"] for q in p],marker="o",color=COLORS[arm],label=NAMES[arm])
        axes[0,2].plot(x,[100*q["missing_separator_rate"] for q in p],marker="o",color=COLORS[arm])
        axes[1,0].plot(x,[q["completion_tokens_mean"] for q in p],marker="o",color=COLORS[arm])
        axes[1,1].plot(x,[q["raw_steps_mean"] for q in p],marker="o",color=COLORS[arm])
    p=stat["phases"]["B7"]
    for field,title,color in [("min_score_mean_scored","All scored",COLORS["B7"]),("min_score_mean_correct","Answer correct","#3e67ae"),("min_score_mean_wrong","Answer wrong","#b3704e")]:
        axes[0,1].plot(x,[q[field] for q in p],marker="o",color=color,label=title)
    axes[0,0].set(title="Logged training answer accuracy",ylabel="Percent",ylim=(30,50));axes[0,0].legend(fontsize=8,loc="lower left")
    axes[0,1].set(title="B′: mean minimum PRM score",ylabel="Native probability",ylim=(0,.75));axes[0,1].legend(fontsize=8)
    axes[0,2].set(title="No internal double-LF separator",ylabel="Percent of all outputs",ylim=(0,12))
    axes[1,0].set(title="Mean generated length",ylabel="Completion tokens",ylim=(500,800))
    axes[1,1].set(title="Mean raw split count",ylabel="Nonempty double-LF segments",ylim=(8,15))
    values=[100*stat["groups"][a]["all"]["process_component_abs_share"] for a in NAMES]
    axes[1,2].bar(["A","Old B","B′"],values,color=[COLORS[a] for a in NAMES],width=.55)
    for i,v in enumerate(values):axes[1,2].text(i,v+.9,f"{v:.1f}%",ha="center")
    axes[1,2].set(title="Process term amplitude share",ylabel="Pooled absolute component share (%)",ylim=(0,36))
    for ax in [*axes[0],axes[1,0],axes[1,1]]:ax.set_xlabel("Generated completion tokens (millions)")
    fig.suptitle("Seed17 · 6M math RL audit · stored rollouts only",fontsize=17,weight="bold")
    fig.savefig(OUT/"training_trends.png",dpi=170)
    plt.close(fig)

    train_table=[]
    metrics=[("无效输出合计",lambda t:pct(sum(t["failure_rates"].get(k,0) for k in ["truncated","missing_or_invalid_box","unparseable","verifier_error"]))),
             ("截断",lambda t:pct(t["failure_rates"].get("truncated",0))),
             ("缺少/无效答案框",lambda t:pct(t["failure_rates"].get("missing_or_invalid_box",0))),
             ("数学表达式无法解析",lambda t:pct(t["failure_rates"].get("unparseable",0))),
             ("平均 completion token",lambda t:f'{t["completion_tokens_mean"]:.1f}'),
             ("平均原始分段数",lambda t:f'{t["raw_steps_mean"]:.2f}'),
             ("无内部双换行",lambda t:pct(t["missing_separator_rate"])),
             ("CRLF 使原始切分合并",lambda t:pct(t["crlf_merges_steps_rate"]))]
    for title,fn in metrics:train_table.append([title,*[fn(stat["train"][a]) for a in NAMES]])
    group_table=[]
    for kind,title in [("mixed","混合"),("all_correct","全对"),("all_wrong","全错")]:
        g=stat["groups"]["B7"][kind]
        group_table.append([title,str(g["groups"]),pct(g["fraction"]),str(g["has_invalid_groups"]),f'{g["mean_abs_advantage"]:.3f}',pct(g["process_component_abs_share"])])
    phase_table=[]
    for p in stat["phases"]["B7"]:
        phase_table.append([f'{p["bin_million_completion_tokens"][0]}–{p["bin_million_completion_tokens"][1]}M',p["n"],pct(p["outcome_accuracy"]),f'{p["min_score_mean_scored"]:.3f}',f'{p["min_score_mean_correct"]:.3f}',f'{p["min_score_mean_wrong"]:.3f}',pct(p["missing_separator_rate"])])
    unit_table=[]
    for a in NAMES:
        v=unit["sources"][a+"/gsm8k"]
        unit_table.append([NAMES[a],int(unit["sources"][a+"/train"]["delta"]),f'{int(v["old_correct"])}/1319 · {pct(v["old_accuracy"])}',f'{int(v["new_correct"])}/1319 · {pct(v["new_accuracy"])}'])
    review_table=[]
    for c in reviewed:
        r=c["record"];signal=c["training_signal"]
        review_table.append([c["sample_id"],c["source_row"],c["annotation"]["math_status"],r["status"],int(r["outcome_reward"]),f'{r["min_score"]:.3f}' if r["step_scores"] else "未评分",f'{float(signal["actual_or_replayed_advantage"]):+.3f}'])

    text="""7B PRM B′训练审计（固定输入，未启动新训练）

结论：原生 PRM 分段位置、评分批次、冻结权重、混合奖励及实际 GRPO advantage 的检查通过。没有发现能复现的索引/评分/优势实现错误。但原结果验答器的百分数单位错误仍在：它在 A/旧B/新B 训练记录中分别造成9/11/18个假阴性。修复该单位口径后，GSM8K的A和新B各增加10题，差距仍是31题；MATH500不变。这里只重判已生成回答，不会修复已执行训练的梯度。

1. 分隔符与长度
训练前尚未更新参数的首批32条回答，A和新B均为32/32含内部双LF，平均13.94段；这是首批训练题的检查，不是整个基座输出分布的独立评测。
A/旧B/新B 无内部双LF比例为5.74%/7.48%/5.89%；平均长度671.9/659.3/664.2 token；平均原始段数11.64/11.51/11.53。新B无分隔符比例从首1M的7.55%降到末1M的4.90%，段数从11.32升到12.44。未见整体越来越少写分隔符、越来越短的证据。
新B有934/9056条在CRLF规范化后段数会改变（10.31%），属于现行切分器的格式敏感问题；A也有10.76%，不能仅凭此解释B落后。原生marker位置和日志分数数量对应是正确的。非空无分隔符响应是一段；空串是零段且不评分。

2. 无效输出
A/旧B/新B的训练无效比例17.40%/18.07%/17.45%，截断6.04%/6.03%/5.72%。新B未出现相对A的大规模格式或截断崩溃。上述指标以所有生成输出为分母；原始文本段数与仅评分回答的step_count分开统计。

3. 分数趋势
新B首1M到末1M：已评分回答平均min分数0.350→0.379，训练答对率42.25%→40.63%。但答对回答均分0.574→0.645，答错回答0.112→0.116接近不变。未见持续拉高错误答案过程分的清晰趋势。训练题目随阶段变化，趋势不是固定题目复测，因此不能凭两条曲线认定投机。未做中间checkpoint的固定测试集准确率评测。

4. 下降来自哪类失败
按原验答器口径，在GSM8K上，A正确而新B错误98题：72题有可解析但错误的答案、15题缺框、10题截断、1题解析失败。反向恢复67题后，净少31题可按正确性翻转核算为28题可解析错答、3题截断；缺框和解析失败净为0。按明确百分数单位修正后，净差仍为31题，细分为27题可解析错答、3题截断、1题不可解析。两种口径都不支持把差距主要归因于没写出答案。
MATH500双向翻转36失/31得，净少5题；对应净分类为3个可解析错答、1个截断、1个缺框。全测试集失败类型总数的变化还含两组都答错但失败类型互换的题，故与这个翻转核算不同。这只是输出类别核算，不证明训练机制因果。

5. 组类型和优势
新B：混合893组(78.89%)、全对32组(2.83%)、全错207组(18.29%)。全同结果组的结果项中心化后为0，非零奖励优势由过程项产生；其平均|advantage|约0.74–0.79，混合组约0.82。按相同混合奖励分母进行加法分解，过程项的绝对幅度份额为28.68%，混合组内为12.03%。这不是梯度占比，也不是PRM的因果贡献率。
207个全错组中173组含无效输出，因此这部分信号还包含对“完整且可解析”输出的偏好，不全是有效数学进展。分组依据旧验答标签；百分数假阴性的敏感性另见unit_group_sensitivity.json。

6. 随机20条逐条数学审阅
Python random.Random(285)从全部9056条均匀不放回抽样，包含无效输出；题目、gold和全文先看，PRM分数/结果标签后揭示。由Codex单审阅，不是独立人工金标准。
13条实际被PRM评分，7条未评分。4条目标数学推导无误，其中3条被评分且均无<0.5步骤；另1条只有空答案框而未评分。这不能由3条样本推出整体没有错罚。
10条已评分回答含早期数学错误，其中B04/B07/B08全过程分均>=0.5；后两条后续纠正了目标结论，不能与始终未纠正的B04混为一类。B04漏掉绝对值方程的x=60解，却获min=0.533（错误分支0.698）；其总advantage仍为负，不能称策略整体在鼓励这条错答。
B14最终8正确，但“80不能被16整除”等推导明确错误；min=0.241，在全对组实际advantage仍为+0.401。低绝对分不等于负学习信号。
B15开头错误79得0.980，后来正确504所在步骤得0.234，显示PRM对错误位置的判断不可靠；这不是token索引对齐错误，且整条含早期错误，不能算纯正确推导被错罚。
B09先正确求出披萨31/32，随后生成另一题并box210，覆盖原答案。B18质因数计数6正确，却没写有效答案框。B19正确算出sqrt59后又box8，最终正文再回到sqrt59，仍被最后答案框协议判错。B20的15%全程正确，却受单位验答bug误罚，实际advantage=-0.579。

范围：无新解题采样、无策略训练、无PRM适配或权重更新。只对20个原评分批次共512条已存回答做冻结PRM重放，逐步分数与日志完全一致；全部7476已评分回答的81182个步骤数量/位置检查通过；20条旧验答标签复算与原代码一致，说明单位错误可由原规则稳定复现。331个固定输入文件哈希未变。

材料：review.html含20条完整题目、生成原文、每步判断/分数和组内信号；statistics.json/CSV为全量统计；native_replay.json为原生评分重放；unit_replay.json为百分数敏感性；mathematical_verification.json为独立算术/代数核验。
"""
    (OUT/"REPORT.txt").write_text(text)

    css="""body{margin:0;background:#f4f5f7;color:#253147;font:15px/1.7 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1180px;margin:auto;padding:38px 24px 70px}h1{font-size:32px;line-height:1.3;margin-bottom:8px}h2{font-size:22px;margin-top:32px}h3{font-size:18px}.sub{color:#647086}.card,details.case{background:white;border:1px solid #dde2ea;border-radius:13px;padding:22px;margin:17px 0}.lead{background:#eaf3f0;border-left:5px solid #168477}.warn{background:#fff4df;border-left:5px solid #bc8534}.tag{font-size:12px;padding:4px 9px;border-radius:20px;background:#e9edf4;margin-right:7px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;border-bottom:1px solid #e5e8ef;text-align:left;vertical-align:top}th{background:#f2f5f8;white-space:nowrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;border-radius:8px;padding:13px;font:13px/1.65 ui-monospace,SFMono-Regular,monospace}.muted{color:#6b7383;font-size:13px}img{width:100%;border-radius:9px}.steps td:nth-child(2){width:65%}.bad{color:#a34538}.good{color:#16715a}summary{cursor:pointer;font-size:17px;font-weight:650}.score{font-variant-numeric:tabular-nums}.blind .score{visibility:hidden}.toggle{background:#263b60;color:white;border:0;border-radius:7px;padding:9px 14px;cursor:pointer}.reason{background:#f0f4f8;padding:12px;border-radius:7px}a{color:#2863a1}@media(max-width:700px){main{padding:22px 12px}h1{font-size:25px}.card,details.case{padding:14px}th,td{padding:7px}}"""
    parts=["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>7B PRM B′ 训练审计</title><style>"+css+"</style><body><main>",
        "<h1>7B PRM B′：实现与训练信号审计</h1><p class='sub'>seed17 · 三组约6M生成token · 9056条新B回答 · 随机20条逐条数学审阅</p>",
        "<div class='card lead'><b>核心结论</b><p>评分位置、冻结权重、混合奖励和实际优势核验通过；512条原批次重放的逐步分数与训练日志完全一致。没有看到整体少写分隔符或缩短输出的证据。</p><p>仍有明确的百分数单位验答bug、答案被后续文本覆盖，以及PRM漏判/相对优势误导案例。GSM8K差距主要来自可解析错答，无法归结为单纯格式失败。</p></div>",
        "<div class='card warn'><b>统计边界</b><p>在线训练各阶段题目不同，分数上涨与正确率下降的相关性不能直接证明投机。随机20条是单个助手的数学审阅，不是独立人工金标准；0.5仅作描述性阈值。已知单位修复只重判现有输出，无法追溯修复训练参数。</p></div>",
        "<h2>1 · 分隔符、无效输出与长度</h2><div class='card'>",table(["训练指标","A","旧B","新B′"],train_table),
        "<p class='muted'>无效=截断、缺框、不可解析或验证异常；互斥计数。段数由全部原始输出按同一个双LF切分器重算，不把未评分step_count=0当成真实文本无步骤。</p><p>训练前尚未更新参数的首批32条，A和新B均32/32含内部双LF，平均13.94段；这只是首批题目检查。新B′无分隔符从首1M的7.55%降至末1M的4.90%；平均分段数从11.32升至12.44。CRLF依然会合并步骤，但不是新B独有的问题。7476条原生编码全部对齐，最大输入3062token，没有静默截断。</p></div>",
        "<h2>2 · 分数涨了，是否在投机？</h2><div class='card'>",table(["阶段","输出数","结果正确率","平均min分","答对min分","答错min分","无分隔符"],phase_table),
        "<p>平均min分0.350→0.379，正确率42.25%→40.63%；但答错回答均分0.112→0.116基本不变，答对均分0.574→0.645。缺少持续抬高错误过程分、缩短步骤等一致证据，不能只凭总均分和正确率两条曲线下结论。</p>",
        "<img src='training_trends.png' alt='Training trends and process advantage component amplitude'><p class='muted'>同题顺序核对：三组共有前1120个八回答题组，题目顺序完全一致；token窗口中的具体题目仍不完全相同。以上不是固定评测集的中间checkpoint学习曲线。</p></div>",
        "<h2>3 · 准确率损失来自哪类输出</h2><div class='card'>",
        table(["新B′相对A","A对B错","B对A错","净少答对","净可解析错答","净截断","净缺框"],[["MATH500",36,31,5,3,1,1],["GSM8K",98,67,31,28,3,0]]),
        "<p>上表采用原验答器口径。GSM8K的98个丢失答案中，72个可解析但错误，15个缺框、10个截断、1个不可解析；上表扣除了反向恢复的67题。明确百分数单位修正后，净差仍为31题，细分为27个可解析错答、3个截断、1个不可解析。</p><p class='muted'>此分类针对正确性翻转；与全体错误类型总数变化不同，后者还包含两组都错误而类型发生互换的题。两种口径都不支持将差距主要归因于没写出答案。</p></div>",
        "<h2>4 · 组类型与PRM驱动的优势</h2><div class='card'>",table(["新B′组类型","组数","比例","含无效输出","平均|A|","过程项幅度份额"],group_table),
        "<p>固定混合奖励的同一个标准差分母，将优势拆为结果中心项+过程中心项，过程项全体绝对幅度份额为<b>28.68%</b>。这不是梯度占比，也不涵盖PRM改变分母的全部影响。全同结果组的奖励优势全部来自过程项；173个全错组含无效输出，还混有对完整/可解析格式的偏好。</p></div>",
        "<h2>5 · 已确认的单位验答问题</h2><div class='card warn'><p>B20明确要求百分数，数学上正确的15%被旧验答器与无单位gold15比较而判错。其PRM min=0.992，但实际advantage=-0.579。这不是PRM打低分。</p>",
        table(["组别","训练假阴性修正数","GSM8K原口径","GSM8K单位修正后"],unit_table),
        "<p>MATH500三组不变；GSM8K的A和新B均增加10题，差距仍为31题。修复不会解释掉现有评测差距，但训练期间的错误监督影响不能通过事后重判消除。只对题面明确要求的百分数单位做已有修复；未全局剥除百分号，明确要求不带百分号的math500/24保持原规则。</p></div>",
        "<h2>6 · 随机20条完整审阅</h2><div class='card'>",table(["样本","源行","数学审阅","日志状态","结果奖励","min分","实际优势"],review_table),
        "<p>13条实际评分、7条未评分；3条完整可评分且推导无误的回答均无&lt;0.5步骤。10条已评分且含早期错误的回答中，B04/B07/B08全程分≥0.5，其中后两条后来纠正了目标答案。B04漏掉解60是最明确的未纠正漏判。B14的错推导在全对组仍获正优势。B15低分落在正确的后续步骤，但它的开头确有错误，不能计为整条纯正确证明被错罚；这是PRM语义判断问题，不是token索引错位。</p></div>"
    ]
    for c in reviewed:
        r,a,m=c["record"],c["annotation"],c["training_signal"]
        parts += [f"<details class='case' id='{c['sample_id']}'><summary>{c['sample_id']} · row {c['source_row']} · {html.escape(a['math_status'])}</summary>",
            f"<p class='muted'>{html.escape(c['problem_id'])} · optimizer step before update {r['optimizer_step_before_update']} · {html.escape(m['outcome_group'])}</p>",
            "<h3>题目与标准答案</h3><pre>"+html.escape(c["question"])+"\n\nGOLD: "+html.escape(str(c["gold"]))+"</pre>",
            "<p class='reason'>"+html.escape(a["mathematical_reason"])+"</p>",
            f"<p>原日志：{html.escape(r['status'])}；提取答案 <code>{html.escape(str(r['answer']))}</code>；结果奖励 {r['outcome_reward']:.0f}；实际advantage {float(m['actual_or_replayed_advantage']):+.4f}；过程中心项 {float(m['process_component']):+.4f}。</p>",
            "<button class='toggle' onclick=\"this.nextElementSibling.classList.toggle('blind')\">显示 / 隐藏逐步PRM分数</button><div class='blind'>",
            "<table class='steps'><thead><tr><th>步</th><th>完整原文</th><th>数学审阅</th><th>PRM</th></tr></thead><tbody>"]
        for s in c["steps_with_review"]:
            value="未评分" if s["prm_score"] is None else f'{s["prm_score"]:.6f}'
            parts.append(f"<tr><td>{s['step']}</td><td><pre>{html.escape(s['text'])}</pre></td><td>{html.escape(s['judgment'])}</td><td class='score'>{value}</td></tr>")
        parts += ["</tbody></table></div><details><summary>同一八回答组的实际信号</summary>",
            table(["源行","结果奖励","min分","实际优势"],[[p["source_row"],p["outcome"],"未评分" if p["min_score"] is None else f'{p["min_score"]:.5f}',f'{p["advantage"]:+.4f}'] for p in c["same_eight_response_group"]]),"</details></details>"]
    parts += ["<div class='card'><b>复核材料</b><p><a href='REPORT.txt'>完整文字报告</a> · <a href='statistics.json'>全量统计</a> · <a href='training_rows.csv'>逐回答CSV</a> · <a href='native_replay.json'>原生评分重放</a> · <a href='unit_replay.json'>单位验答敏感性</a> · <a href='reviewed_cases.json'>20条完整审阅JSON</a> · <a href='mathematical_verification.json'>算术/代数核验</a></p><p class='muted'>源数据331文件SHA-256核验通过。全部原始A/B与B′文件保留；无新训练、无新解题采样。一次20个评分批次的冻结PRM推理重放，不是策略训练。</p></div></main></body></html>"]
    (OUT/"review.html").write_text("\n".join(parts))
    save(OUT/"audit_receipt.json",{"complete":True,"frozen_input_files":331,"input_hashes_verified":True,
        "new_training":False,"new_actor_rollouts":0,"PRM_weight_updates":0,"frozen_inference_replayed_rows":512,
        "reviewed_records":20,"native_scored_rows_checked":7476,"native_steps_checked":81182,
        "actual_advantage_audits":283,"originals_preserved":True,
        "source_sha256":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
        "report_sha256":{n:sha(OUT/n) for n in ["REPORT.txt","review.html","statistics.json","native_replay.json","unit_replay.json","mathematical_annotations.json","review_summary.json","unit_corrected_flip_attribution.json","reviewed_cases.json","training_trends.png","sample_manifest.json","base_separator_check.json"]}})
    print(json.dumps({"review_summary":review_summary,"unit_group_sensitivity":alternative_groups,"report":str(OUT/"review.html")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
