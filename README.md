# Mathematical Reasoning RL: Outcome and Process Rewards

ECE285 三人项目：用 **Qwen2.5-Math-1.5B、GRPO 和 LoRA** 比较结果奖励与过程奖励，检查 PRM 打分如何变成实际训练信号。成员按报告顺序为 **Hongyi Ding、Yang Zeng、Jasmine Jiang**。主要硬件是一张共享 A100 80GB。

这是截至 **2026-11-09** 的代码与结果审阅快照。训练和推理按用户要求停止；本次同步没有使用 GPU，也不会自动恢复旧任务。

## 阅读入口

- **期中报告**：[PDF](midterm_neurips_20261109/ece285_midterm_report_20261109.pdf) · [LaTeX](midterm_neurips_20261109/main.tex) · [引用核对](midterm_neurips_20261109/citation_verification.json)
- **当前进度**：[方案与边界](docs/CURRENT_PLAN.txt) · [机器可读状态](results/current_status.json)
- **主 A/B 对照**：[结果](results/corrected_ab_seed17_6m.json) · [训练实现](experiments/ab_corrected_6m_seed17_20261022/)
- **后续 A/C-bounded**：[seed17](execution/abc_seed17_resumed_v1_20261104/completed_evidence/comparison.json) · [seed42](execution/ac_seed42_v1_20261106/completed_evidence/comparison.json) · [综合分析](results/ac_6m_summary_20261109.json)
- **逐题紧凑结果**：[CSV 与校验信息](results/compact_predictions/)；不含题目正文或完整 CoT。

## 1. GRPO 与直接 PRM 混合

两组使用同一个初始模型、提示词和纠错后的冻结验答器。A 的 reward 是终答是否正确的 `y`；B 的 reward 是 `y + 0.2*m`，其中 `m` 是未适配的 Qwen2.5-Math-PRM-7B 给出的最低步骤分。两者都按普通 GRPO 组内标准化。

这组历史实验使用 **V0 评测路径、seed17、约 6M 生成 token**。A 为 284 次更新，B 为 280 次；前 1,120 组题序相同，按生成 token 匹配，而非相同更新次数或总计算量。

| 数据集 | A：结果奖励 | B：直接混合 | B−A，95% 配对题目区间 |
|---|---:|---:|---:|
| MATH500 | 54.00% | 53.80% | −0.20 pp [−4.00, +3.60] |
| GSM8K | 67.02% | 64.97% | −2.05 pp [−4.25, +0.08] |

两个区间均含零。这轮没有观察到 B 的收益，也不能据此断言 PRM 普遍有害或两种方法等效。

失败分析区分两件事：**PRM 是否漏掉数学错误，以及回答实际拿到了什么 advantage**。B 有 577 条错答得到正优势，其中 571 条来自全错组。全同结果组内，0.2 大体被标准差归一化抵消；系数小并不保证过程信号小。高 PRM 分也不一定产生正优势。具体案例与实现检查见期中报告第 5 节。

## 2. C-bounded 后续实验

`A_out` 表示对 `y` 做普通组内标准化。C-bounded 用

```text
A_out + 0.2 * y * (m - mean_correct(m))
```

合成后不再归一化。没有正确回答时过程项为零。过程修正绝对值不超过 0.2，错答的 reward advantage 不会为正；KL 项仍可能产生梯度。这是信号约束，不是数学推导有效性的证明。

后续实验使用 **V1 路径及 C 适配后的冻结 7B PRM**。每个 seed 内匹配题序和 288 次更新，约 6.1–6.3M 生成 token；因此不能把本表和上面的 V0 A/B 表直接拼成一个多臂比较。

| Seed | 方法 | MATH500 | GSM8K | 生成 token |
|---|---|---:|---:|---:|
| 17 | A | 53.00% | 66.87% | 6,155,438 |
| 17 | C-mix | 54.00% | 67.70% | 6,114,465 |
| 17 | C-bounded | 57.80% | 69.37% | 6,167,374 |
| 42 | A | 52.80% | 67.48% | 6,308,962 |
| 42 | C-bounded | 54.80% | 67.93% | 6,229,129 |

C-bounded−A 的配对 95% 区间：

| Seed | MATH500 | GSM8K |
|---|---|---|
| 17 | +4.80 pp [+1.40, +8.40] | +2.50 pp [+0.45, +4.55] |
| 42 | +2.00 pp [−1.00, +5.00] | +0.45 pp [−1.52, +2.43] |

方向一致，但 seed42 的两个区间都含零。两个 seed 的均值只是描述性平均，不能把重复使用的测试题当成独立翻倍样本。C-bounded 同时改变门控、幅度和归一化；当前结果不能单独归因于某一项，也未证明推导更有效。门控本身不是我们的首创。

## 3. 短试与暂停状态

以下是从同一已有 C-bounded checkpoint 续跑 48 次更新的**验证集探索**，不是新的完整训练结果，也不能与测试集准确率直接比较。

| 比较 | 验证集差值 | 95% 配对区间 |
|---|---:|---|
| 均匀减罚 − 同轮 C-bounded | +1.60 pp | [−1.00, +4.20] |
| 前缀保护 − 均匀减罚 | +0.20 pp | [−2.00, +2.60] |
| 只改长度归一化 − 原 loss | −0.60 pp | [−3.20, +2.00] |
| 树状 rollout − 同轮链式 rollout | −1.80 pp | [−4.40, +0.80] |

尚无支持替换主方法的清晰证据。实现分别在 `experiments/prefix_protection_20261107/`、`length_norm_pair_seed17_48_20261107/`、`tree_credit_triplet_seed17_48_20261108/`。

较大预算的续训已经暂停：A 完整 checkpoint 为 step432（9,321,966 token），C-bounded 为 step365（7,845,984 token；日志到 step368）。**没有共同 9M/12M 评测结果**，不能推断续训收益。来源见 [CPU 状态核对](execution/ac_status_cpu_20261109/summary.json)。不直接执行历史 controller、dispatch 或 CONTINUATION 文件。

## 代码和复核

| 目录 | 内容 |
|---|---|
| `training/`、`training_b/` | 初始 GRPO、PRM runner、数据准备、验答与评测 |
| `experiments/` | 独立实验版本、真实依赖和运行控制代码 |
| `execution/` | 审计、离线重算、配置、聚合证据及报告生成代码 |
| `midterm_neurips_20261109/` | 当前报告、NeurIPS 样式、引用和公式核对 |
| `results/` | 紧凑结果与当前状态；旧记录保留作历史 |
| `proposal_neurips/` | 原 proposal 和历史修订 |
| `operations/` | 历史运维脚本，仅按记录阅读 |

依赖见 [training/requirements.txt](training/requirements.txt)，历史环境锁见 `execution/baseline_seed17/requirements.lock.txt`。主要版本为 torch 2.8.0、transformers 4.56.2、TRL 0.23.1、PEFT 0.17.1、vLLM 0.10.2。

CPU 核对紧凑结果：

```bash
python tools/check_compact_results.py
```

这只重算保存标签的正确数、配对胜负和文件 hash，不重新验答完整推导。完整重算需要原环境的数据、权重和 rollout。代码同步工具是 `tools/sync_project_snapshot.py`，默认仅预览；只有显式 `--apply` 才写分享副本。

## 分享与复现边界

代码、报告、图表、聚合审计和紧凑逐题结果进入 Git；权重、优化器状态、完整 rollout、数据集、缓存和重复压缩包留在原位置。分享副本中的个人路径、服务器地址和 SSH 目标已替换为示例值；原始训练代码与结果未改动。

[SHARE_MANIFEST.json](SHARE_MANIFEST.json) 记录来源与分享 hash；历史 release/receipt 对应原始实验文件，不能直接认证脱敏后的源码。[SHARE_CHECKS.json](SHARE_CHECKS.json) 记录本次检查。仓库没有自动 GPU 工作流，同步代码不授权恢复实验。

## 上游方法

[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)、[Qwen PRM](https://arxiv.org/abs/2501.07301)、[PURE](https://arxiv.org/abs/2504.15275)、[ProcessBench](https://arxiv.org/abs/2412.06559)。第三方来源和许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
