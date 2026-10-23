"""Render concise Chinese findings from validated receipts, no inference."""
import json
from pathlib import Path
import sys

from v2core import atomic_json, read_jsonl


def main():
    out = Path(sys.argv[1])
    load = lambda name: json.loads((out / name).read_text())
    summary = load("replay_summary.json")
    outcomes = load("outcome_summary.json")
    scoring = load("paired_scoring_receipt.json")
    tests = load("validation_fixed_fixture/test_results.json")
    protection = load("originals_after.json")
    qa = load("independent_QA.json")
    numerical = load("numerical_audit_summary.json")
    failure = load("actual_failure_batch_replay.json")
    admission = load("training_admission_manifest.json")
    assert tests["passed"] and qa["passed"] and protection["passed"]
    comparisons = summary["common_retained_groups"]
    lines = ["独立奖励修复 v2：现有 B rollout 离线回放完成", "",
        "本轮没有生成新回答，没有策略或 PRM 参数更新，没有恢复 B，也没有修改原 A/B 代码、数据和运行结果。",
        f"原始 B-6M：{summary['rows']} 条回答、{summary['groups']} 个八回答组。原验证器的标签、状态、PRM 可用性全部重现。",
        f"独立版本：{tests['tests']} 项回归测试通过；对 {scoring['rows']} 条可评分回答做旧/新输入成对 BF16 推理，其余不变组使用原冻结 PRM 分数缓存。",
        "受影响组保留全部八个伙伴，包括原来无效/截断的输出。未评分的 min 显示 null，避免把占位零当 PRM 判错。", "",
        "一、可确认的代码/数据修复", 
        f"百分数：纠正 {outcomes['unit_label_changes']} 条结果奖励，见 outcomes.jsonl；固定单位元数据来自题目/标准答案，不看当前模型分数。",
        f"矛盾题：独立清单隔离 {admission['excluded_ids']}；未来训练题数 {admission['original_training_questions']}→{admission['prospective_eligible_questions']}。原数据未删改。",
        f"回放仍保留全部原始组；下表在双方共同保留的 {summary['retained_groups']} 组/{8*summary['retained_groups']} 条上比较，排除的是整组矛盾题。",
        "换行：仅规范化 PRM 视图中的 CRLF；actor 原文、提示、PRM 权重、双换行原生序列化及评分终点规则不变。",
        f"重复相同输入分数最大差 {scoring['repeat_max_difference']:.3g}；旧/新视图输入相同的回答分数最大差 {scoring['unchanged_input_max_difference']:.3g}。", "",
        "二、reward、排序与 advantage 的实际变化（变化不等于改善）",
        "修复项 | 结果标签改变 | reward 改变 | reward 排序改变组 | 优势正负/零状态改变 | |Δ优势|>0.01",
    ]
    for key, label in [("units_only", "仅单位"), ("split_only", "仅换行"), ("combined_matched", "联合修复")]:
        r = comparisons[key]
        lines.append(f"{label} | {r['outcome_label_changed_rows']} | {r['reward_changed_rows_gt_1e_minus_6']} | {r['reward_order_changed_groups']} | {r['advantage_sign_flip_rows']} | {r['advantage_changed_rows_gt_0p01']}")
    lines.extend(["reward/优势变化使用 1e-6 死区；另报 |Δ优势|>0.01。排序按原八回答组计算，并单独保留 process 排序与全对/全错/混合分层。",
        "主比较以同一设备、精度和填充形状的旧输入回放为参照；不是直接把历史日志与新推理的差异全部算作修复。",
        "单位与换行的 reward 变化可分别定位，标准化优势的变化不能简单相加。", "",
        "历史重推的数值对照（不能算作修复收益）："])
    backend = comparisons["backend_replay_control"]
    lines.append(f"仅重新推理旧输入：{backend['process_min_changed_rows_gt_1e_minus_6']} 条最低分变化，{backend['reward_order_changed_groups']} 组 reward 排序变化，{backend['advantage_sign_flip_rows']} 条优势符号状态变化。")
    lines.extend(["", "三、已审阅数学案例（同一原文，未作为本轮训练数据）",
                  "案例 | 视图 | 结果奖励 | 最低过程分 | 段数 | 混合 reward | reward 排名 | 优势"])
    examples = {r["review_id"]: r for r in load("case_comparisons.json")}
    chosen = ["R02", "R04", "R01", "S03", "S08", "S11", "S15", "S19", "S20", "prior_row_2095"]
    for name in chosen:
        for variant, label in [("logged", "原日志"), ("legacy_replayed", "匹配旧输入"), ("combined", "联合修复")]:
            r = examples[name]["variants"][variant]
            score = "未评分" if r["min_score"] is None else f"{r['min_score']:.6f}"
            lines.append(f"{name} | {label} | {r['outcome_reward']:g} | {score} | {r['step_count']} | {r['reward']:.6f} | {r['reward_rank']:g} | {r['advantage']:+.6f}")
    lines.extend(["R02 是全程正确的最大公因数推导；R04 是 20% 的单位误判；R01 是错误乘积及虚构计算输出；S11 是错误几何论证/舍入。",
        "S03/S08 是原审阅中有效推导，S19 是正确但单段的反函数推导；S20 是单段多处配方错误。",
        "需按上述实际数值判断每个案例有无变化。修换行不能自动修好一般数学误判；增加步骤还会改变 min 聚合的机会数。",
        "这些案例和先前 50 条审阅均是诊断材料，不是独立人工金标准，不能据此估计总体误判率。", "",
        "四、B 退出断言的定向回放",
        f"读取 B 最后保存批次（optimizer_step_before_update={failure['optimizer_step_before_update']}）的 32 个 reward，重建安装版 TRL 的张量计算。",
        f"旧 CPU/GPU 硬比较是否失败：{failure['old_guard_failed']}；不匹配元素 {failure['old_guard_mismatched_elements']}；全 32 项最大差 {failure['max_old_cross_device_difference']:.9g}。",
        f"重建的完整断言错误信息与原日志一致：{failure['reconstructed_old_guard_error'] == failure['recorded_failure_message']}。原报错的最大差仅统计未通过比较的项，口径不同于上面的全 32 项最大差。",
        f"新同设备审计通过：{failure['new_guard_passed']}；实际优势张量未改动：{failure['actual_advantage_tensor_unchanged']}。",
        "原 GPU 优势完整张量未保存；这里是从保存 reward 重建，而非对缺失原张量逐位比较。B 本身没有恢复。",
        f"B-6M 数值对照：旧跨设备容差失败批次 {numerical['old_cross_device_fail_batches']}；最大跨设备优势差 {numerical['max_old_cross_device_error']:.9g}。",
        f"六个回放版本共 {numerical['batches']} 批实际优势均通过新审计，核对没有替换训练优势；独立 Python 算术另核对 {qa['checked_variant_groups']} 个版本-组。", "",
        "五、对四组实验的含义",
        "A 纯结果；B 原始冻结 PRM；C 对称适配；D 去掉成功弱监督。C/D 的成功项系数仍是唯一的方法差异。",
        "数值审计补丁是目标不变的修复。单位验证器、训练题清单或 PRM 输入更改则需按影响范围建立匹配版本；没有把补丁装入现有训练。",
        "仅改 PRM 输入时 A 可在兼容核对后复用，B/C/D 要共享该规则；训练期单位判分/数据更改影响四组，不能直接拼旧 A 与新 B/C/D。",
        "本轮说明代码口径确有影响，不能证明策略数学准确率改善，也不能证明 C/D 已修复 PRM。后续是否训练仍按用户指令。", "",
        f"保护验收：源码/日志/适配器哈希未变={protection['hashed_files_unchanged']}；A/B 全结果文件大小与修改时间清单未变={protection['result_inventory_unchanged']}。",
        "详情：replay_summary.json；case_comparisons.csv；group_changes.csv；paired_scores.jsonl；variants/；validation_fixed_fixture/；independent_QA.json。",
    ])
    (out / "REPORT.txt").write_text("\n".join(lines) + "\n")
    atomic_json(out / "delivery_status.json", {"complete": True, "tests_passed": tests["tests"],
                "offline_rows": summary["rows"], "paired_scored_rows": scoring["rows"],
                "new_training": False, "new_rollouts": 0, "model_updates": 0, "B_resumed": False,
                "originals_unchanged": protection["passed"], "independent_QA_passed": qa["passed"]})
    print(json.dumps({"report": str(out / "REPORT.txt"), "complete": True}))


if __name__ == "__main__":
    main()
