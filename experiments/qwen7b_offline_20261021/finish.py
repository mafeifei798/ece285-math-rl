"""Seal the completed offline audit and produce a concise evidence-led interpretation."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from core import atomic_json, read_jsonl, sha256


def main():
    root,out=map(lambda p:Path(p).resolve(),sys.argv[1:3])
    summary=json.loads((out/'summary.json').read_text())
    independent=json.loads((out/'independent_QA.json').read_text())
    assert summary['complete'] and independent['passed']
    assert json.loads((out/'identity_control_QA.json').read_text())['passed']
    source_root=root/'experiments/qwen7b_offline_20261021'
    sealed=json.loads((out/'source_release_after_loading_compat.json').read_text())
    assert all(sha256(source_root/name)==value for name,value in sealed.items())
    local_sources={p.name:sha256(p) for p in sorted(source_root.iterdir()) if p.is_file()}
    dependencies={rel:sha256(root/rel) for rel in [
        'experiments/reward_repair_v2_20261021/v2core.py',
        'experiments/reward_repair_v2_20261021/trl_kernel.py']}
    atomic_json(out/'final_source_manifest.json',{'production_sources_unchanged_since_qwen_start':True,
                'source_files':local_sources,'imported_project_dependencies':dependencies,
                'additions_after_production_source_seal':sorted(set(local_sources)-set(sealed))})

    # Descriptive signal magnitude: no change to reward, normalization or model.
    strength={}
    for name in ['h4_raw','qwen7b_raw','h4_repaired','qwen7b_repaired','outcome_only_raw','outcome_only_repaired']:
        rows=read_jsonl(out/'variants'/f'{name}.jsonl');strength[name]={}
        for kind in ['mixed','all_correct','all_wrong']:
            groups=[rows[i:i+8] for i in range(0,len(rows),8)
                    if rows[i]['group_type']==kind and not rows[i]['quarantined'] and rows[i]['group_usable']]
            values=[abs(r['advantage']) for g in groups for r in g]
            strength[name][kind]={'groups':len(groups),'mean_abs_advantage':statistics.mean(values),
                'RMS_advantage':math.sqrt(statistics.mean(v*v for v in values)),
                'groups_with_max_abs_advantage_below_0p01':sum(max(abs(r['advantage']) for r in g)<.01 for g in groups),
                'groups_with_max_abs_advantage_below_0p1':sum(max(abs(r['advantage']) for r in g)<.1 for g in groups)}
    atomic_json(out/'signal_strength.json',strength)
    random=summary['review']['uniform_random'];target=summary['review']['targeted_combined']
    comparisons=summary['comparisons']['model_only_raw']
    cases={c['review_id']:c for c in json.loads((out/'review_cases.json').read_text())}
    h,q=summary['resources']['h4'],summary['resources']['qwen7b']
    text=[
        '离线结论：Qwen 7B PRM 值得作为更有依据的 B′ 评分器候选；不是所有 PRM 问题都已解决。', '',
        '执行范围：B 前 6M 保存的 9120 条回答，7472 条完整可评分；每个模型共评分 8339 个原始/CRLF 变体视图。',
        '没有生成新回答，没有策略/PRM 训练，没有恢复 B。原 A/B 源码、数据和结果的保护检查通过。', '',
        '1. 随机 30 条的固定审阅中，7 条过程有效、20 条有明确错误；另外 2 条不确定、1 条题面矛盾不计入二分类。',
        f"   有效推导被拒绝：{random['h4']['clean_solutions_falsely_rejected_at_0p5']}/7 → {random['qwen7b']['clean_solutions_falsely_rejected_at_0p5']}/7。",
        f"   错误推导全程被放过：{random['h4']['erroneous_solutions_fully_passed_at_0p5']}/20 → {random['qwen7b']['erroneous_solutions_fully_passed_at_0p5']}/20。",
        f"   首个确定数学错误定位命中：{random['h4']['first_error_location_exact_at_0p5']}/18 → {random['qwen7b']['first_error_location_exact_at_0p5']}/18。",
        '   这些是旧的 Codex 单审阅小样本，非独立人工金标准；不外推成总体准确率。', '',
        '2. 改善确实进入了原八回答组的优势，不只是把分数整体挪动。',
        f"   虚构 Python 乘积输出 R01：min {cases['R01']['h4_min']:.3f} → {cases['R01']['qwen7b_min']:.3f}，advantage {cases['R01']['h4_advantage']:+.3f} → {cases['R01']['qwen7b_advantage']:+.3f}。",
        f"   错误几何论证 S11：min {cases['S11']['h4_min']:.3f} → {cases['S11']['qwen7b_min']:.3f}，advantage {cases['S11']['h4_advantage']:+.3f} → {cases['S11']['qwen7b_advantage']:+.3f}。",
        f"   有效代码推导 S03：min {cases['S03']['h4_min']:.3f} → {cases['S03']['qwen7b_min']:.3f}，advantage {cases['S03']['h4_advantage']:+.3f} → {cases['S03']['qwen7b_advantage']:+.3f}。", '',
        '3. 仍有明确反例，不能把换模型当作全部修复。',
        f"   正确 gcd(5!,6!)=120 推导 R02：min 从 {cases['R02']['h4_min']:.3f} 升到 {cases['R02']['qwen7b_min']:.3f}，但组内 rank 从 5 降到 8，advantage 从 {cases['R02']['h4_advantage']:+.3f} 变为 {cases['R02']['qwen7b_advantage']:+.3f}。",
        '   这说明绝对分提高不等于组内学习信号变好；其他伙伴的分数也变了。',
        '   另一个已知错答案例行 78 仍排第一，Qwen min=0.056、advantage=+1.240。',
        '   正确解答的负优势本身不自动构成错误：整组没有逐条过程真值，案例需要结合明确数学错误与步骤判分解读。', '',
        '4. 全量分组比较（预先排除整组矛盾题，双方共 1139 组/9112 条）。',
        f"   {comparisons['reward_order_changed_groups']} 组 reward 顺序变化；{comparisons['advantage_sign_flips']} 条优势正负/零状态变化；{comparisons['advantage_changed_rows_gt_0p01']} 条 |Δadvantage|>0.01。",
        '   变化本身不是准确率收益；混合/全对/全错分层在 summary.json，实际幅度在 signal_strength.json。',
        '   答对解答内部 AUROC 0.714→0.833，但差值 bootstrap 95% 区间跨 0；不能据此声称可靠识别“答对但推导错”已被证明。',
        '   单独筛出的 20 条可疑样本修复更明显，但其选择依赖旧 H4 分数，不能作为无偏总体比较。', '',
        '5. 单张 A100 80GB、BF16、batch=8 实测，以下是独立评分器成本。',
        f"   H4：{h['forward_seconds']:.1f} 秒，{h['production_views_per_second']:.2f} 视图/秒；Qwen：{q['forward_seconds']:.1f} 秒，{q['production_views_per_second']:.2f} 视图/秒。",
        f"   评分阶段耗时比 {q['forward_seconds']/h['forward_seconds']:.2f} 倍；不是整个 RL 训练的耗时比。",
        f"   PyTorch 活跃分配峰值：{h['allocated_peak_GiB']:.2f} → {q['allocated_peak_GiB']:.2f} GiB；缓存预留峰值：{h['reserved_peak_GiB']:.2f} → {q['reserved_peak_GiB']:.2f} GiB。",
        '   缓存预留与正在参与计算的分配量是两种口径；不能只用较小数字承诺整套训练能装进某张卡。',
        '   本轮未测 actor、生成 KV cache、反向和优化器的组合显存。', '',
        '6. 对项目的建议：保留旧 B，后续若采用 Qwen 7B，B′/C′/D′ 共用这一底座和一致的奖励/判分协议。',
        '   本轮支持评分器选择，不代表新 B 的数学答题准确率已提高，也不证明 C/D 适配已有效；当前不启动任何训练。', '',
        '逐条对照：review_comparison.html；完整统计：REPORT.txt、summary.json；验收：independent_QA.json。',
    ]
    (out/'FINDINGS.txt').write_text('\n'.join(text)+'\n')
    atomic_json(out/'pipeline_status.json',{'complete':True,'stage':'offline_scoring_replay_and_QA_complete',
        'old_A_B_preserved':True,'new_training':False,'new_rollouts':0,'model_updates':0})
    print('Sealed source manifest, signal strength and FINDINGS.txt',flush=True)


if __name__=='__main__':main()
