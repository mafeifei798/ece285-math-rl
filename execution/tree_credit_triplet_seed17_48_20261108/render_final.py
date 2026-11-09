"""Render actual accepted results. Never uses simulated/forecast accuracies."""
from pathlib import Path
import json,hashlib,html,re
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent;OUT=HERE/'completed_evidence'
def read(p):return json.loads(p.read_text())


def main():
    verified=read(HERE/'completion_verification.json');assert verified['passed']
    c=read(OUT/'comparison.json');stats=c['paired_comparisons'];cost=c['training_costs'];values=c['validation_summary']
    evaluation_record=read(OUT/'evaluation/summary.json')
    mechanism=read(OUT/'admission/mechanism_diagnostics.json')['arms']
    independent=read(OUT/'admission/independent_tensor_acceptance.json')
    interpretation=read(HERE/'interpretation.json');runtime=read(HERE/'runtime_closeout.json')
    assert runtime['passed'] and not runtime['our_live_controller_or_children']
    primary=stats['tree_cb_minus_chain_cb'];increment=stats['tree_cb_minus_tree_vr']
    tree_only=stats['tree_vr_minus_chain_cb']
    if primary['paired_bootstrap_95CI_pp'][0]>0:
        decision='树状＋PRM 在本次单 seed 短续训验证集上高于链式控制，值得进一步确认；本轮不自动扩预算。'
    elif tree_only['paired_bootstrap_95CI_pp'][0]>0:
        decision='无PRM的树状续训在本次验证集上高于链式控制；树状＋PRM未确认优于控制。优先保留tree_vr候选，本轮不自动扩预算。'
    elif primary['paired_bootstrap_95CI_pp'][1]<0:
        decision='树状＋PRM 在本次短续训验证集上较低，不建议按当前配置直接扩长训。'
    else:
        decision='本轮没有确认树状＋PRM 优于链式 C-bounded。结合较高回退率和额外耗时，不建议把当前树状配置扩到3M/6M；继续保留链式 C-bounded 作为主线。'
    text=['树状信用分配三组短试 · 最终结果','2026-11-08｜真实模型验收、三组训练、共同验证与独立复核完成',
          '',decision,'','实验范围：同一 C-bounded checkpoint-288 各续跑48次更新至336，seed17，固定题序，每次4题、每题8叶；只评测此前已使用的 validation500 验证集，不是 MATH500 测试集。source 指续训前的 C-bounded，不是初始基座模型。',
          'chain_cb＝原链式 C-bounded；tree_vr＝树状 rollout＋结果树信用，不加载 PRM；tree_cb＝同一树信用＋固定有界 PRM 项。',
          '树结构只用于训练；共同验证仍是单条贪心 CoT，不使用树搜索、投票或 PRM 重排。',
          '这是 TreeRL-paper 风格在既有 TRL clipped loss 内的适配，不是完整官方 OpenRLHF/RLOO 复现。树信用用原始节点成功率，区别于链式 GRPO 的结果标准差缩放；比较同时改变采样和信用构造，不能据此单独归因。','',
          '统一验证集（500题）：']
    for arm in ('source','chain_cb','tree_vr','tree_cb'):
        v=values[arm];text.append(f"{arm}: {int(v['correct'])}/500 = {v['accuracy']*100:.1f}%；截断 {v['truncated']}，缺框 {v['missing_or_invalid_box']}，空文本 {v['empty']}，32token内错答 {v['short_wrong']}。")
    text+=['','配对比较（百分点；题目 bootstrap95%区间）：',
           '预先指定 tree_cb−chain_cb 为主比较；另两项为探索性次比较，未做多重比较校正。']
    for name in ('tree_cb_minus_chain_cb','tree_vr_minus_chain_cb','tree_cb_minus_tree_vr'):
        v=stats[name];lo,hi=v['paired_bootstrap_95CI_pp']
        text.append(f"{name}: {v['difference_percentage_points']:+.1f} [{lo:+.1f}, {hi:+.1f}]；{v['wins']}胜/{v['losses']}负；McNemar p={v['mcnemar_exact_two_sided_p']:.4f}。")
    text.append('统计边界：tree_vr−chain_cb 的原 NumPy 20,000次bootstrap区间上端为−0.2点，独立标准库20,000次重算为0.0点；恰差一题，符合预设0.2点Monte Carlo容差。其精确McNemar p=0.065，且属于未校正的次比较，因此不宣称它稳健地显著更差。主比较的区间两次重算一致，均包含0。')
    text+=['','资源成本（同更新次数，实际token不强行相等）：']
    for arm,v in cost.items():
        prefill=independent['arms'][arm]['rollout_requested_prefill_tokens']
        text.append(f"{arm}: 唯一生成 {v['unique_generation_tokens']:,}；展开训练 {v['expanded_leaf_training_tokens']:,}；逻辑prefill输入 {prefill:,}；PRM输入 {v['PRM_input_tokens']:,}；{v['seconds']/3600:.2f}小时；采样峰值 {v['peak_sampled_GPU_GiB']:.2f}GiB。")
    text.append(f"三组训练合计 {sum(v['seconds'] for v in cost.values())/3600:.2f}小时；共同验证另耗 {evaluation_record['seconds']/60:.1f}分钟。逻辑prefill是请求输入长度之和，不代表缓存命中后的实际计算量；显存为轮询采样峰值。")
    text+=['','为什么当前树状扩展不值得直接加预算（事后描述性分析，不改原始主比较）：']
    for arm,v in interpretation['arms'].items():
        text.append(f"{arm}: {v['fallback_roots']}/{v['roots']}个根回答回退为独立采样（{v['fallback_rate']*100:.1f}%）；前缀至少32token的分叉仅{v['forks_at_least32_tokens']}/{v['roots']}（{v['forks_at_least32_tokens_fraction_of_all_roots']*100:.1f}%）。相对展开叶子的生成token只节省{v['unique_vs_expanded_token_saving_fraction']*100:.1f}%，墙钟为链式的{v['walltime_ratio_to_chain']:.2f}倍。32token仅用于描述覆盖面，不代表数学正确性或新调的阈值。")
    text+=['回退根的前一两个token已出现答案标记。真实例子以“ \\boxed{(6,1)}”开头，之后仍有数百token的推导；也有先写空框再推理。固定规则在首次答案标记前选分叉，因而无法利用其后推理。它遵守了本轮事先固定的过滤规则，但与当前checkpoint的答案先行格式不相容，不能称为充分实现了中间推理分叉。',
           '全错组仍为链式27/192、tree_vr31/192、tree_cb26/192，且其结果与正确限定的PRM优势均为0；本轮没有显示树结构解决了全错组无学习信号的问题。KL项仍然存在。',
           '上述覆盖面、采样依赖、结果信用尺度和两轮生成成本都是当前适配的限制；这些描述性检查不能单独确定准确率下降的因果来源。它们连同未支持提升的主比较，支持停止扩大当前树状配置。']
    text+=['','机制检查：']
    for arm,m in mechanism.items():
        counts=m['fork_root_counts'];raw=independent['arms'][arm]
        text.append(f"{arm}: 分叉/回退 {counts.get('nonempty_fork',0)}/{counts.get('no_eligible_prefix',0)}；错答共享前缀的正结果信用token {raw['wrong_positive_shared_prefix_tokens']:,}；错答正PRM与独有后缀正优势均为0；最大独立重算误差 {raw['max_credit_float64_error']:.3g}。")
        if arm=='tree_cb':
            text.append('终答正确的叶子，其整条PRM最低分的位置：'+json.dumps(m['correct_leaf_PRM_minimum_location'],ensure_ascii=False))
            reversed_tokens=m['by_group_type']['mixed']['correct_tokens_reversed_from_positive_outcome_to_negative_total']
            text.append(f'混合组中，终答正确回答里原正结果信用被PRM压成负总优势的token：{reversed_tokens}。这不是对token数学正确性的标注；固定0.2上限并不保证过程项永不盖过较小的树信用。')
            reversed_rows=interpretation['arms']['tree_cb']['counts'].get('answer_correct_leaves_positive_outcome_reversed_by_process',0)
            text.append(f'这{reversed_tokens}个token来自{reversed_rows}条终答正确回答，不能据此把总体差距归因于PRM反向惩罚。最低分位置统计中“独有后缀”包含空前缀回退时的整条回答，也不等于成功定位了数学错误。')
    text+=['','验收：CPU与真实模型零更新门槛通过；各组48次真实优化器更新；原始token/EOS、前缀log-prob来源和优势fingerprint核对；全部144批/4608回答独立float64重算；2000条验证预测使用原验答器重放一致；原始保护清单与源checkpoint保持原样。',
           '验收覆盖边界：三组都沿用vLLM采样校正关闭、num_iterations=1的设置；144批均未缓存old log-prob或sampling ratio。原独立检查器的可选采样比率重建分支不适用，误差0不能理解为已重建比率。补充CPU核查已确认这一点；当前loss在该分支使用当前log-prob的detach值。训练代码与设置未改。',
           '限制：单seed、48更新短续训，实际token另列；验证集已使用；配对题目区间不包含训练seed与跨运行数值波动。此前相同源checkpoint的贪心重评出现过0.4点净差，原因未定位。短试无效不证明从初始模型训练的TreeRL无效；过程奖励优劣不能只由原始分数升降判断。',
           '共享前缀的正信用表示此后更可能答对，不是数学步骤正确性的证明。整条最低PRM分投影到独有后缀仍是启发式，尤其需结合最低分所在区域解释。',
           '没有启动新的PRM适配、额外seed、3M/6M长训或测试集评测。所有远端新作业位于 /path/to/ece285_math_rl_project/runs/Tree_credit_triplet_seed17_48_20261108。']
    (HERE/'REPORT.txt').write_text('\n'.join(text)+'\n')
    result={'decision':decision,'primary_comparison':primary,'PRM_incremental_comparison':increment,
            'validation':values,'costs':cost,'evaluation_wall_seconds':evaluation_record['seconds'],
            'independent_statistics':verified['independent_local_standard_library_statistics'],
            'post_hoc_interpretation':interpretation,'runtime_closeout':runtime,
            'independent_acceptance_passed':True,'test_sets_used':False}
    (HERE/'result_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    labels={'source':'续训前 C-bounded（288步）','chain_cb':'原 C-bounded，再训48步','tree_vr':'树状结果信用','tree_cb':'树状结果信用＋PRM'}
    val_rows=''.join(f'<tr><td>{labels[a]}</td><td>{int(v["correct"])}/500 ({v["accuracy"]*100:.1f}%)</td><td>{v["truncated"]}</td><td>{v["missing_or_invalid_box"]}</td></tr>' for a,v in values.items())
    cost_rows=''.join(f'<tr><td>{a}</td><td>{v["unique_generation_tokens"]:,}</td><td>{v["expanded_leaf_training_tokens"]:,}</td><td>{v["seconds"]/3600:.2f}h</td><td>{v["peak_sampled_GPU_GiB"]:.2f}GiB</td></tr>' for a,v in cost.items())
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>树状信用三组短试 · 完成报告</title><style>body{font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f3f5f8;color:#21344c;margin:0}main{max-width:1080px;margin:35px auto;padding:38px 44px;background:white;border-radius:12px}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.callout{background:#eef5ff;border-left:4px solid #4276b4;padding:18px}table{width:100%;border-collapse:collapse;font-size:15px}th,td{border-bottom:1px solid #dce5ee;text-align:left;padding:11px}th{background:#f4f7fa}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:15px;line-height:1.85;background:#f6f8fa;padding:18px}.muted{font-size:14px;color:#63748a}a{color:#2764a2}@media(max-width:720px){main{padding:22px;margin:0}table{font-size:12px}td,th{padding:7px}}</style><main><h1>树状信用分配 · 三组短试完成</h1>'''
    page+='<p class="muted">seed17 · 同源checkpoint288→336 · 每组48次更新 · validation500验证集（非MATH500测试）· 原始结果保留</p><div class="callout">'+html.escape(decision)+'</div>'
    page+='<h2>共同验证结果</h2><table><tr><th>模型</th><th>准确率</th><th>截断</th><th>缺框</th></tr>'+val_rows+'</table>'
    page+='<h2>真实成本</h2><table><tr><th>组别</th><th>唯一生成token</th><th>展开训练token</th><th>墙钟</th><th>峰值显存</th></tr>'+cost_rows+'</table>'
    page+='<h2>解释与独立证据</h2><pre>'+html.escape('\n'.join(text[7:]))+'</pre>'
    page+='<p><a href="REPORT.txt">完整文本</a> · <a href="result_summary.json">结果JSON</a> · <a href="completion_verification.json">独立验收</a> · <a href="completed_manifest.json">文件哈希</a> · <a href="interpretation.json">覆盖面与样例</a> · <a href="runtime_closeout.json">释放状态及检查覆盖边界</a> · <a href="analysis_plan.json">分析范围</a> · <a href="../tree_extension_audit_20261108/review.html">文献与代码查重</a> · <a href="../../experiments/tree_credit_triplet_seed17_48_20261108/README.txt">独立实现说明</a></p></main></html>'
    (HERE/'review.html').write_text(page)
    deliver=['REPORT.txt','result_summary.json','review.html','completion_verification.json','analysis_plan.json','completed_manifest.json','interpretation.json','runtime_closeout.json']
    scripts=['independent_cpu_verify.py','mechanism_diagnostics.py','verifier_replay.py','verify_local_completed.py','interpret_completed.py','runtime_closeout.py','render_final.py']
    (HERE/'delivery_receipt.json').write_text(json.dumps({'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'files':{n:{'bytes':(HERE/n).stat().st_size,'sha256':hashlib.sha256((HERE/n).read_bytes()).hexdigest()} for n in deliver},
        'analysis_code_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in scripts},
        'training_release_sha256':read(HERE/'launch_receipt.json')['release_sha256'],
        'training_complete':True,'independent_acceptance_complete':True},indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
