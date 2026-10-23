"""Readable fixed-rollout repair report and final provenance receipt."""
from pathlib import Path
import argparse
import html
import json
from repairs import atomic_json, sha256, VERSION

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"execution/reward_repair_v3_20261021"
read=lambda n:json.loads((OUT/n).read_text())


def table(headers,rows):
    return "<div class='scroll'><table><thead><tr>"+"".join("<th>"+html.escape(str(v))+"</th>" for v in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+html.escape(str(v))+"</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table></div>"


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=OUT)
    OUT=parser.parse_args().out
    result,score,units,cases=[read(n) for n in ["replay_complete.json","scoring_complete.json","unit_summary.json","case_impacts.json"]]
    assert result["complete"] and score["complete"] and score["model_frozen"]
    c=result["comparisons"]
    unit_rows=[]
    for arm,label in [("A","A"),("B_old","旧H4 B"),("B7","7B B′")]:
        train=units[arm+"_train"];test=units[arm+"_gsm8k"]
        unit_rows.append([label,len(train["changes"]),f'{int(test["old_correct"])}/{test["n"]}',
            f'{int(test["new_correct"])}/{test["n"]}',f'{100*test["new_correct"]/test["n"]:.2f}%'])
    variants=[]
    for key,label in [("backend_padding_control","数值/批次控制"),("units_only","仅单位修复"),
        ("crlf_only","仅CRLF修复"),("combined_matched","两项联合修复")]:
        x=c[key]
        variants.append([label,x["outcome_label_changed_rows"],x["step_count_changed_rows"],
            x["reward_changed_rows_gt_1e_minus_6"],x["reward_order_changed_groups"],
            x["reward_pairwise_order_changes"],x["advantage_changed_rows_gt_0p01"],x["advantage_nonzero_sign_flip_rows"]])
    guard=result["guardrail"];guard_rows=[]
    for k,label in [("mixed","混合"),("all_correct","全对"),("all_wrong","全错")]:
        v=guard["stats"][k]
        guard_rows.append([label,v["groups"],f'{v["current_mean_abs_advantage"]:.4f}',f'{v["candidate_mean_abs_advantage"]:.4f}',f'{v["candidate_max_abs_advantage"]:.4f}'])
    case_table=[]
    selected={v["sample_id"]:v for v in cases}
    for name,reason in [("B04","漏掉方程的第三个解"),("B14","最终答案对但推导有错"),("B15","先错后对，步骤评分定位不可靠"),("B20","15%遭结果验答器误判")]:
        r=selected[name];old=r["variants"]["matched_legacy"];new=r["variants"]["combined"]
        case_table.append([name,reason,f'{old["outcome_reward"]:.0f} → {new["outcome_reward"]:.0f}',
            f'{old["min_score"]:.4f} → {new["min_score"]:.4f}',f'{old["reward"]:.4f} → {new["reward"]:.4f}',
            f'{old["advantage"]:+.4f} → {new["advantage"]:+.4f}',f'{r["proposed_bounded_advantage"]:+.4f}'])
    v=c["combined_matched"];s=c["crlf_only"];u=c["units_only"]
    text=f"""百分数验答与CRLF分段修复：独立v3离线验证

完成范围
独立版本：experiments/reward_repair_v3_20261021。
使用原来7B B′的9056条完整训练记录及原八回答组；A和旧B只复核单位标签。
不训练策略，不适配PRM，不生成新回答，不修改原A/B代码、数据或结果。
七项回归检查通过。源331个受保护输入文件的哈希前后相同。

实际修复
百分数：只依据固定题目/标准答案哈希绑定的单位元数据，对明确要求百分数的答案使用同一单位比较；不是全局删百分号，概率题语义不变。修复后的unit_exact/unit_symbolic同时纳入过程评分有效状态，避免修好结果标签后又被旧状态白名单屏蔽。
CRLF：只在PRM输入副本把CRLF转换为LF，仍按原双换行切分。策略原文、答案提取和actor token不改；保留逐段到原文的字符位置映射。
混合奖励仍是R=y+0.2*min(p)，原GRPO标准差归一化不变。以下不包含新目标的训练效果。

单位重判
A/旧B/7B B训练假阴性分别纠正9/11/18条。GSM8K重新判分A935/1319=70.89%，7B B904/1319=68.54%，仍差31题；MATH500不变。事后纠正无法撤销过去训练的错误梯度。

7B B分段重放
867条实际评分输入受CRLF影响。仅重放包含这些输入的668个原生微批次，共2580条伙伴的旧/新输入；其余复用原冻结分数。
两侧相同BF16模型、伙伴、顺序和padding形状。历史缓存与旧输入重放的差异单列为数值控制，不当作修复收益。
仅CRLF修复：{s['step_count_changed_rows']}条评分回答的步数改变，{s['reward_changed_rows_gt_1e_minus_6']}条奖励改变，{s['reward_order_changed_groups']}个题组排序改变，{s['advantage_nonzero_sign_flip_rows']}条非零优势翻转符号。
仅单位修复：{u['outcome_label_changed_rows']}条结果标签改变，{u['reward_order_changed_groups']}个题组排序改变，{u['advantage_nonzero_sign_flip_rows']}条非零优势翻转符号。
联合修复：{v['reward_changed_rows_gt_1e_minus_6']}条奖励改变，{v['reward_order_changed_groups']}个题组排序改变，{v['advantage_changed_rows_gt_0p01']}条优势变化超过0.01，{v['advantage_nonzero_sign_flip_rows']}条非零优势翻转符号。
比较使用完整原题组、FP32奖励和安装版TRL0.23.1的奖励张量运算，未构造trainer。排序改变不等于数学判断更正确；增加步骤本身就可能降低min。

全同结果组的独立方案（未启用训练）
A_i=GRPO_outcome_advantage(y_i)+0.2*(m_i-mean_group(m))。
保持结果奖励的GRPO优势，把过程项作为不再按自身标准差放大的有界残差。八回答组中过程项绝对值<=0.175；混合组的结果优势绝对值至少约0.35345，因此不能被这个过程残差翻转正确/错误的优势符号。
这是统一工程保护，不是本项目新贡献，不会让错误PRM自动变正确；全对组内的错误推导仍可能得到小的正残差。数据上的幅度变化见guardrail_diagnostic.json，不能解释成新策略准确率提升。

PRM漏判的方案
代码修复不能补上缺失的数学判断能力，也不能只靠整体降分或调阈值解决排序。
继续保留原C/D适配定义和公共有标签步骤锚点：C=L_fail+L_success+L_step，D=L_fail+0*L_success+L_step。失败解答只表示至少有一步错，不给整条所有步骤强行贴负标签。
如果选7B，B/C/D必须都从同一个7B PRM初始化。旧H4 C/D小试验不能当作7B B的适配对照；那轮结果没有修复已知错答的相对排序，不能承诺D一定有效。
后续任何适配先修正弱结果标签、统一CRLF输入，在固定且与训练隔离的有标签过程验证集上检查有效推导接受率、首错识别、答对内部AUROC及区间，再决定是否值得投入RL。已审阅20例只作回归诊断，不用作新增训练标签或独立测试集。

不破坏四组
A：统一修复验答器的纯结果GRPO。
B：共同PRM初始化，不适配。
C：共同PRM初始化，c=1适配。
D：同C的初始化、批次、锚点、优化器和步数，只把c变为0。
单位修复影响所有组；CRLF和过程奖励规则统一用于B/C/D。若以后采用有界残差，也必须同时用于B/C/D，A过程项为0，D-C仍只比较c。
旧A/B作为历史结果保存。新规则下训练过的B/C/D不能与旧规则A冒充严格匹配对照；共同重判无法修正旧训练梯度。此次没有启动任何新训练。
"""
    (OUT/"REPORT.txt").write_text(text)
    css="body{margin:0;background:#f4f6f8;color:#26354b;font:15px/1.75 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1180px;margin:auto;padding:32px 22px 65px}h1{font-size:30px;line-height:1.3}h2{font-size:21px;margin-top:30px}.card{background:white;border:1px solid #dce2e9;padding:21px;border-radius:12px;margin:17px 0}.lead{background:#e9f4ef;border-left:5px solid #258571}.warn{background:#fff5df;border-left:5px solid #b98635}.muted{color:#6b7788;font-size:13px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;vertical-align:top;border-bottom:1px solid #e6e9ed;padding:9px 11px}th{background:#f0f4f7;white-space:nowrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f7;padding:15px;border-radius:8px}a{color:#286a9f}code{background:#eef1f5;padding:2px 4px;border-radius:4px}@media(max-width:700px){main{padding:18px 12px}.card{padding:14px}h1{font-size:25px}}"
    parts=["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Reward repair v3 · 离线验证</title><style>"+css+"</style><body><main>",
        "<h1>百分数验答与CRLF分段：独立v3修复</h1><p class='muted'>原7B B′ · 9056条固定回答 · 原八回答组 · 无新训练或新解题采样</p>",
        "<div class='card lead'><b>已经完成的代码修复</b><p>按题目要求的百分数单位验答；只在PRM输入副本规范化CRLF；接通新的单位判断状态与过程奖励。保持原奖励公式和GRPO目标。原始A/B代码、数据和结果保留。</p></div>",
        "<div class='card warn'><b>解释边界</b><p>这里测的是同一批回答会收到什么奖励和学习信号，不能称为新模型准确率提升。PRM数学漏判不会因换行修复而自动消失。下方有界过程项仅为单独的离线方案，没有启用训练。</p></div>",
        "<h2>1 · 单位修复：正确的15%不再与15按不同单位比较</h2><div class='card'>",
        table(["组别","训练假阴性纠正数","GSM8K原结果","单位修正后","修正后准确率"],unit_rows),
        "<p>只处理固定元数据明确要求的百分数/百分点题，题目与gold均绑定哈希。概率、数量题及明确禁止百分号的题保留原语义。MATH500不变；A与新B仍差31题。旧训练中的错误梯度无法由事后重判消除。</p></div>",
        "<h2>2 · 成对回放：隔离单位、CRLF与数值波动</h2><div class='card'>",
        table(["比较","标签改变","步数改变","reward改变条数","排序改变组数","改变样本对","|ΔA|>0.01","非零A符号翻转"],variants),
        "<p class='muted'>所有修复效应均相对matched_legacy比较；第一行仅是历史缓存到相同padding重放的数值控制。reward和排序采用实际FP32量化，变化死区1e-6。原题组不删伙伴。步数变化仅对两侧实际评分的回答计数。</p>",
        f"<p>867条输入受CRLF影响；保留668个原生微批次的全部伙伴，共2580条，旧/新两侧同模型、顺序和padding。原文和actor token不改，规范化步骤映射回原字符位置。冻结推理共{score['wall_seconds']:.1f}秒，峰值allocated显存{score['peak_allocated_GiB']:.2f}GiB；该测量是PRM单独推理，不是联合训练显存。</p>",
        f"<p>重复输入最大差{score['maximum_repeat_difference']:.3g}，未改变输入的配对伙伴最大差{score['maximum_unchanged_input_difference']:.3g}。更多步骤可机械降低min分，不能把下降或排序变化直接当作更懂数学。</p></div>",
        "<h2>3 · 已审阅案例：修代码与修数学判断分开看</h2><div class='card'>",
        table(["样本","原问题","结果奖励","min分","混合reward","原GRPO优势","有界方案优势*"],case_table),
        "<p class='muted'>前后使用同一已生成回答、相同原伙伴；*最后一列只是单独目标方案的离线计算。此前已看过的20例不是未见测试证据。<a href='../b7b_training_audit_20261021/review.html'>查看原20条完整数学审阅</a>。</p></div>",
        "<h2>4 · 全同结果组：把过程信号变成有界附加项（未启用）</h2><div class='card warn'>",
        "<pre>A_i = (y_i − mean(y)) / (sample_std(y) + 1e-4)\n      + 0.2 × (m_i − mean(m))\n\nm_i = 最低步骤概率；未评分无效输出沿用0占位</pre>",
        "<p>保留结果GRPO的优势，过程残差不再除以自己的微小标准差。group8下过程项绝对值最多0.175；混合组的结果优势至少约0.35345，所以这个附加项不会逆转答对/答错的优势符号。全同结果组仍可接收小的过程信号。</p>",
        table(["修复后组类型","组数","原目标平均|A|","有界方案平均|A|","有界方案最大|A|"],guard_rows),
        "<p>这是共有工程保护，不是D独有的新贡献，也不保证PRM排序正确。错误推导仍可能得到小的正残差。方案未接入训练；无需为这些数字启动RL。</p></div>",
        "<h2>5 · PRM漏判与四组对照</h2><div class='card'>",
        table(["组","统一验答/分段规则后的定义"],[["A","结果奖励GRPO"],["B","共同公共PRM初始化，保持冻结"],["C","同B的PRM初始化，L_fail + L_success + L_step适配，随后冻结"],["D","同C所有配置，只令成功弱监督项c=0，随后冻结"]]),
        "<p>数学漏判要靠评分器的过程标签和泛化能力改善，不能仅整体压低分数。保留公共步骤标签锚点，失败解答只作为“至少有一步错误”的弱监督。旧H4小适配没有修复已知错答排序；不能承诺D一定能修复7B PRM。</p><p>若选7B，B/C/D必须同为7B初始化。适配先统一单位弱标签和CRLF视图，用隔离的有标签过程验证集检查有效推导接受、首错识别、答对内部AUROC及区间。已看过20例只作回归，不当新增训练标注。</p><p>单位修复影响所有四组；CRLF与过程规则共同应用于B/C/D。若以后采用有界过程项，也同时给B/C/D，不能仅给D。旧A/B留作历史结果；新规则训练组与旧规则A不是严格匹配对照。此次未启动训练。</p></div>",
        "<div class='card'><b>复现与原始证据</b><p><a href='REPORT.txt'>文字报告</a> · <a href='replay_complete.json'>完整回放统计</a> · <a href='case_impacts.json'>20例影响</a> · <a href='guardrail_diagnostic.json'>有界方案诊断</a> · <a href='scoring_complete.json'>冻结推理收据</a> · <a href='unit_summary.json'>单位纠正清单</a> · <a href='trl_tensor_kernel.json'>TRL运算来源</a></p><p class='muted'>331个受保护输入SHA-256前后相同。代码位于experiments/reward_repair_v3_20261021。7项回归检查通过；无新策略样本、无任何参数更新。</p></div></main></body></html>"
    ]
    (OUT/"review.html").write_text("\n".join(parts))
    files=["REPORT.txt","review.html","replay_complete.json","guardrail_diagnostic.json","scoring_complete.json","unit_summary.json","case_impacts.json","preparation.json"]
    atomic_json(OUT/"final_receipt.json",{"complete":True,"version":VERSION,"original_inputs_unchanged":331,
        "new_training":False,"new_actor_samples":0,"PRM_updates":0,"guardrail_training_enabled":False,
        "source_sha256":{p.name:sha256(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
        "dependency_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/"experiments/reward_repair_v2_20261021/v2core.py",ROOT/"experiments/reward_repair_v2_20261021/trl_kernel.py",ROOT/"experiments/reward_repair_v2_20261021/replay_offline.py",ROOT/"experiments/qwen7b_offline_20261021/core.py",ROOT/"experiments/b_qwen7b_6m_20261021/frozen_prm.py"]},
        "artifact_sha256":{n:sha256(OUT/n) for n in files}})
    print(OUT/"review.html")


if __name__=="__main__":main()
