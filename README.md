# ECE285 — Mathematical Reasoning RL and Process Rewards

三人课程研究项目：**小模型数学推理 GRPO 中的过程奖励、失败诊断与结果门控**。

策略模型：Qwen2.5-Math-1.5B base；过程评分器：冻结的 Qwen2.5-Math-PRM-7B；主要硬件：A100 80GB。

## 先看这三个地方

1. [当前完整方案](docs/CURRENT_PLAN.txt)：研究问题、四组实验、预算、评测、分工与时间线。
2. [纠错后的正式 A/B 结果](results/corrected_ab_seed17_6m.json)：已完成、单 seed、匹配生成 token 预算。
3. [已纠错 A/B 的训练实现](experiments/ab_corrected_6m_seed17_20261022/train_arm.py) 与 [原 GRPO runner](training/train_baseline.py)。

## 当前完成到哪里

| 部分 | 状态 |
|---|---|
| 结果奖励 A / 结果＋PRM 的 B | 各约 6M completion tokens、seed17 已完成；训练/信号/评测审计通过 |
| 验答与分段修复 | 已实现并保留离线回放与版本记录 |
| A/B 提示词 2×2 对照 | 已完成 3,992 次推理，未改善 B 的表现 |
| 原 C/D 的 PRM 适配 | 仅小规模探索；没有支持扩大训练的正面证据 |
| PURE OpenRLHF | 独立单卡 LoRA pilot 与官方信号源码核对；不是完整论文配置复现，进度快照不能当实时状态 |
| **结果门控 G / 分数置换 S** | **方案与 CPU 算术可行性检查；尚未接入正式训练** |

正式 A/B：greedy、同一冻结验答器、最多 2,048 tokens，无投票或 PRM 重排。

| 数据集 | A | B | B−A（95% 配对题目区间） |
|---|---:|---:|---|
| MATH500 | 54.00% | 53.80% | −0.20 pp [−4.00, +3.60] |
| GSM8K | 67.02% | 64.97% | −2.05 pp [−4.25, +0.08] |

只有一个训练 seed，两个区间均含零：尚未观察到 B 的收益，不能据此宣称 PRM 普遍有害、等效或非劣。

## 当前四组设计（计划）

令 `y` 是固定验答器的 0/1 结果，`m` 是原生步骤 PRM 正类概率的最小值。

| 组 | reward | 比较作用 |
|---|---|---|
| A | `y` | 纯结果奖励 |
| B | `y + 0.2*m` | 现有混合奖励 |
| G | `y*(1 + 0.2*m)` | 相对 B 只移除错误回答的过程项 |
| S | 同 G，正确回答内部随机交换 `m` | 检验回答与评分的对应信息 |

GRPO 标准化、loss、KL、clipping 与原设置一致。G−A 是主要实际收益比较，G−B 看门控，G−S 看分数归属。门控思想已有先行工作，本项目不声称新的 RL 范式。

## 代码地图

目录保持原有层级，避免破坏跨版本依赖。

| 目录 | 内容 |
|---|---|
| `training/` | 初始 outcome-only GRPO、模型/数据准备、评测及 verifier 测试 |
| `training_b/` | 历史 PRM 混合奖励基线 |
| `experiments/ab_corrected_6m_seed17_20261022/` | 正式纠错 A/B 的独立版本与控制器 |
| `experiments/reward_repair_v2_20261021/`、`reward_repair_v3_20261021/` | 验答/CRLF 分段修复；v2 的 `legacy_snapshot/` 是真实依赖 |
| `experiments/pure_openrlhf_integration_20261023/` | 原始 token/步骤/EOS 对齐与官方 PURE 信号函数接入 |
| `experiments/pure_openrlhf_pilot_20261023/` | 独立 P-VR/P-PRM pilot 的训练与阶段控制代码 |
| `experiments/*` | 其余离线评分、历史方法试验与配对评测 |
| `execution/*` | CPU 审计、回放、统计、报告构建代码及小型证据摘要 |
| `operations/` | 历史续训、恢复和运行脚本；仅作为运行记录阅读 |
| `proposal_neurips/` | 原 proposal 源文件；当前方向见 `docs/CURRENT_PLAN.txt` |
| `results/` | 便于组员阅读的正式摘要入口 |

`b_fix_integration_20261021` 曾探索改变 advantage 的方法，**不等于**正式“只纠错 B”。对照结论以 `ab_corrected_6m_seed17_20261022` 为准。

## 环境与代码检查

训练环境见 [training/requirements.txt](training/requirements.txt)；完整历史环境锁在 `execution/baseline_seed17/requirements.lock.txt`。主要版本为 torch2.8.0、transformers4.56.2、TRL0.23.1、PEFT0.17.1、vLLM0.10.2。

本分享包已做 Python 语法检查、文件哈希/体积检查和凭据模式扫描；不代表重新运行了所有训练或端到端复现。

在具备依赖的环境可按需检查：

```bash
python -m pytest training/test_reward.py training_b/test_b.py
python experiments/pure_openrlhf_integration_20261023/cpu_checks.py
```

这些测试是否在当前机器通过，应以实际运行记录为准。仓库没有开自动训练或 GPU 工作流。

## 分享范围与可复现性边界

- 这是公开的**脱敏代码审阅快照**。服务器地址替换为 `server.example.invalid`，本机路径替换为示例路径；修改只发生在分享副本。
- 原始实验目录、代码和结果未修改。`SHARE_MANIFEST.json` 记录原始/分享文件哈希及是否脱敏；旧 release、receipt 中的哈希对应原实验环境，不能拿它们验证脱敏后的源码。
- 权重、checkpoint、完整训练/评测 rollout、第三方原始数据和重复压缩包不进入 Git。读取这些文件的审计/绘图代码仍保留，重算需先取得对应输入。
- 脚本保留历史阶段的运行入口，**clone 后不应直接执行 dispatch/controller/continuation**。复现实验需要准备数据与模型、配置本机路径，并重新冻结新环境的 release。
- 不含 SSH 私钥、API token 或真实连接凭据。训练中所有失败和负结果仍属于项目证据，不以新方案覆盖。

## 相关工作

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [PAPO](https://arxiv.org/abs/2603.26535)：正确子集过程优势与门控先行工作。
- [PROF-GRPO](https://arxiv.org/abs/2509.03403)：过程分筛选与混合奖励对照。
- [PURE](https://arxiv.org/abs/2504.15275) / [官方代码](https://github.com/CJReinforce/PURE)：逐步骤信用分配；本仓库含带来源记录的核查快照与函数提取。
- [ProcessBench](https://arxiv.org/abs/2412.06559)：过程错误定位评测。

第三方代码来源及许可边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本快照没有为第三方代码追加统一开源许可。

成员：Hongyi Ding、Jasmine Jiang、Yang Zeng。个人贡献以具体代码与实验记录为准。
