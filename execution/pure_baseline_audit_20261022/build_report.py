"""Build a standalone source-audited plan, not a training configuration/launcher."""
from pathlib import Path
import datetime
import hashlib
import html
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
manifest = json.loads((HERE / "source_manifest.json").read_text())
checks = json.loads((HERE / "numerical_checks.json").read_text())
remote = json.loads((HERE / "remote_readonly_evidence.json").read_text())
old_b = json.loads((ROOT / "execution/ab_corrected_6m_seed17_20261022/final_evidence/training_audit/B.json").read_text())
commit = manifest["commit"]
url = f"https://github.com/CJReinforce/PURE/blob/{commit}/"

plan = {
    "status": "source_audited_plan_only",
    "ready_for_training": False,
    "training_started": False,
    "model_forward_or_backward_performed_this_audit": False,
    "official_repository": manifest["repository"],
    "official_branch": "verl",
    "official_commit": commit,
    "paper": "https://arxiv.org/html/2504.15275v3",
    "recommended_name": "PURE RLOO core, Qwen-PRM, single-A100 LoRA adaptation",
    "recommended_execution": "Reuse audited current single-process generation/LoRA infrastructure; port pinned PURE math into a separate trainer. Do not replace the old environment with the old official distributed stack.",
    "comparisons": {
        "primary": "P_PRM minus P_VR: matched new implementation, models, data, optimizer and budget; reward source differs",
        "historical": "Keep A_GRPO and B_MIX as original references; differences versus P arms are whole-recipe comparisons, not isolated reward/credit-assignment effects",
        "deferred": "Only if a min-form mechanism claim is needed: a matched signed sum-form arm. Do not label P_PRM minus B_MIX an isolated min-form ablation."
    },
    "proposed_shared_settings": {
        "actor": "Qwen/Qwen2.5-Math-1.5B",
        "actor_revision": "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
        "initialization": "same original base and seed17 LoRA initialization as corrected A/B; never continue A/B final checkpoints for this comparison",
        "seed": 17,
        "raw_vllm_engine_seed": 0,
        "evaluation_seed": 285,
        "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0,
        "bf16": True, "gradient_checkpointing": True,
        "learning_rate": 1e-6, "learning_rate_reason": "fixed official verl config default, shared by P_VR/P_PRM; differs from old LoRA A/B 5e-6 and paper PRM-only 5e-7",
        "scheduler": "constant", "warmup_steps": 0,
        "group_size": 8, "prompts_per_update": 4, "responses_per_update": 32,
        "actor_microbatch": 1, "gradient_accumulation_steps": 32,
        "dynamic_microbatching": False,
        "prm_microbatch": 2,
        "max_prompt_tokens": 2048, "max_completion_tokens": 2048,
        "fixed_return_padding_width_N": 2048,
        "native_prm_max_total_tokens": 4096,
        "sampling_temperature": 1.0, "top_p": 1.0, "top_k": -1,
        "prompt": "unchanged original corrected A/B prompt; no final_only or official separator-instruction prompt substitution",
        "data_manifest_sha256": remote["data_manifest_sha256"],
        "train_rows": remote["data_manifest"]["files"]["train.jsonl"]["count"],
        "adv_estimator": "official compute_rloo_outcome_advantage",
        "adv_norm": True,
        "whitening_scope": "once across valid tokens of the complete 32-response generation batch, before microbatching",
        "gamma": 1.0,
        "return_aggregate_method": "sum after approximate-min reward transform",
        "ppo_clip": 0.2, "ppo_epochs": 1,
        "kl": {"where": "token reward before return", "type": "official low_var_kl", "coefficient": 0.001, "separate_KL_loss": False},
        "loss_reduction": "official compute_policy_loss per static one-response microbatch; accumulate 1/32. Shared by P_VR and P_PRM; not a full global token-mean substitution.",
        "vllm_gpu_memory_utilization": 0.3,
        "proposed_completion_tokens_per_arm": 6000000,
        "budget_status": "proposal only; no new launch authorized or dispatched",
        "evaluation": "fixed final budget checkpoint; common original greedy MATH500 (500) and GSM8K (1319), max new tokens 2048, existing frozen verifier, no PRM reranking",
    },
    "arms": {
        "P_VR": {"verifiable_reward": "existing repaired outcome at last valid response token", "prm": False},
        "P_PRM": {"verifiable_reward_used_in_training": False,
                  "verifiable_reward_logged_separately": True,
                  "prm": "Qwen/Qwen2.5-Math-PRM-7B",
                  "prm_revision": "0610740060112df12585d00a1c5f4624d2f59051",
                  "score": "r=2*p_positive-1",
                  "softmin_temperature": 0.1, "modeling_reward_coefficient": 0.5,
                  "coverage": "all generated responses, independent of final-answer parsing/correctness; no inherited B valid_for_prm gate",
                  "native_interface": "AutoModel trust_remote_code with <extra_0> and Qwen chat template; PRM-only canonical CRLF view, raw actor output unchanged",
                  "step_to_actor_mapping": "explicit raw-character span to original actor completion-token end index; last step goes to EOS when present or final generated token when truncated",
                  "empty_response": "one terminal step, native Qwen assistant <extra_0> input; score and report separately, do not substitute a hand-designed penalty",
                  "context_overflow_or_alignment_failure": "fail the batch and preserve evidence; never silently truncate, replace the score with zero, or drop/reweight that sample"}
    },
    "necessary_adaptations": [
        "Official PRM checkpoint replaced with existing Qwen PRM and its native scoring API",
        "Single GPU LoRA and current validated inference engine replace official full-parameter distributed workers",
        "32 rather than 512 generated responses per full batch; static microbatch one",
        "2048 completion tokens instead of paper 8192 / public default YAML 7168",
        "Existing data, prompt and frozen verifier retained instead of repository data/forced prompt/grader",
        "Official default curriculum filter explicitly disabled for both new arms",
        "FP32 score processing based on existing Qwen scorer; not a bit-exact BF16 logits reproduction"
    ],
    "must_not_claim": ["full reproduction of published PURE accuracy", "PURE training already smoke-tested on GPU", "PRM-only is guaranteed better", "old A vs new PURE isolates only reward", "one seed establishes general superiority"],
    "acceptance_before_training": [
        "Pin and preserve existing A/B and new source/config hashes",
        "CPU tensor parity against pinned official signed transform, return, RLOO, whitening and PPO functions",
        "Replay exact token ids with CRLF, multiple newline forms, whitespace-only, EOS and truncated cases; assert each native PRM score maps once to a valid actor position",
        "Real-model no-update forward plus backward without optimizer.step, after separate authorization/resource availability; record total-process CUDA peak, throughput, finite gradients and unchanged parameter hashes",
        "Assert P_PRM advantages do not change if gold answers are permuted; P_VR must respond to outcome labels",
        "Assert fixed N=2048 even for all-short batches and no additional TRL group normalization or per-microbatch whitening",
        "Compare chosen static microbatch gradient accumulation to official reference on a tiny deterministic batch",
        "Verify native Qwen PRM has no new/missing scoring-head parameters and remains frozen",
        "Only after user launch authorization, register distinct runs and resumable receipts; no automatic launch from this plan"
    ]
}
(HERE / "baseline_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))


def link(path, label, lines=""):
    return f'<a href="{url}{path}{lines}">{html.escape(label)}</a>'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


sections = []
sections.append('''<section><p class="eyebrow">SOURCE AUDIT · 2026-10-22 · NO TRAINING</p><h1>PURE：适合当前设备的标准基线方案</h1>
<p class="lead">建议采用<strong>官方 RLOO 数学核心的单卡 LoRA 适配版</strong>：沿用 1.5B actor、Qwen 7B PRM 和现有可靠的采样基础设施，成对比较 PURE-VR 与 PURE-PRM。原 A/B 保留为历史对照。</p>
<div class="callout"><strong>可行性判断：</strong>一张 <strong>A100 80GB</strong> 有充分的资源依据承载该适配版；官方默认 8 卡命令不能直接使用。当前已完成源码与 CPU 数值核对，尚未完成 PURE 真模型无更新联调，因此不把“预计可运行”说成“已经跑通”。没有启动训练、没有加载新模型、没有修改原 A/B。</div></section>''')

sections.append('<section><h2>1. 已经核实了哪些事实</h2>' + table(
    ['核实项', '证据与结论'], [
        ['官方版本', f'<code>verl</code> 分支，固定 commit <code>{commit}</code>。下载 19 个关键文件，逐一核对 Git blob 与 SHA-256。'],
        ['硬件', '只读确认 NVIDIA A100-SXM4-80GB。现有 B 的相同 actor＋PRM 组合已完成训练；记录的 PyTorch peak allocated 为 <strong>44.88 GiB</strong>。它不是完整进程峰值，不能直接当全部 GPU 用量。'],
        ['当前资源状态', f'服务器快照 UTC {html.escape(remote["utc"])}：两卡算力利用率均为 100%。GPU1 虽有较多空闲显存，也不能称为空闲卡。本次没有调度作业。'],
        ['当前环境', 'torch 2.8.0 / transformers 4.56.2 / TRL 0.23.1 / PEFT 0.17.1 / vLLM 0.10.2。PURE requirements 限制 vLLM ≤0.9.2；其 worker 走 FSDP、全参数 actor 路径，没有现成的当前 LoRA 接口。'],
        ['模型接口', 'PURE 默认 jinachris/PURE-PRM-7B 是 Qwen2ForTokenClassification；现有 Qwen PRM 是自定义 Qwen2ForProcessRewardModel，通过 AutoModel 和 <extra_0> 评分。两个 checkpoint 不是同一个 PRM，不能只换 model.path。'],
        ['CPU 检查', f'32 个随机 batch，RLOO 开／关 whitening 两种模式，另测 PPO clipping、全零回报、KL 和 padding 例子。全部通过，advantage 最大绝对误差 {checks["max_advantage_absolute_error"]:.3g}。这是合成张量测试，不是模型训练或真实 token 对齐验收。']]) +
    '<p class="sources">证据：<a href="source_manifest.json">源码清单</a> · <a href="remote_readonly_evidence.json">服务器只读快照</a> · <a href="prm_compatibility_sources.json">两个 PRM 的固定配置</a> · <a href="numerical_checks.json">CPU 数值检查</a></p></section>')

sections.append('''<section><h2>2. PURE 实际怎样把奖励变成训练信号</h2>
<p>应以以下整条路径作为基线，不能只把当前 B 的最低分换个名字。</p>
<ol><li><strong>原生步骤打分。</strong>Qwen PRM 在每个 <code>&lt;extra_0&gt;</code> 给出正类概率 p；按 PURE 源码转换为有符号分数 r = p − (1−p) = 2p−1。</li>
<li><strong>近似 min-form。</strong>对一条回答的各步骤计算 w<sub>s</sub> = softmax(−r<sub>s</sub>/0.1)，并取 q<sub>s</sub> = w<sub>s</sub>r<sub>s</sub>。低分步骤权重大；有限温度时不是硬取最小值。</li>
<li><strong>放回 token 位置。</strong>将 0.5q<sub>s</sub> 放在该步骤的最后一个 actor token，其他位置过程奖励为零。0.5 来自固定官方 YAML，而不是继续沿用旧 B 的 0.2。</li>
<li><strong>先加入 KL 约束，再累计回报。</strong>使用官方 low_var_kl，系数 0.001；从后往前累计，γ=1。P_PRM 没有答案正确奖励；P_VR 则将原验答奖励放在最后一个有效 token。</li>
<li><strong>官方 RLOO。</strong>每个 token 的累计回报减去同题其他 7 条回答的 token 回报均值。分母使用固定填充宽度 N=2048，不改成每条真实长度或 batch 动态最长长度。</li>
<li><strong>一次 masked whitening 和 PPO loss。</strong>在整个 32 条回答生成批次的有效 token 上做一次 whitening，再分成 microbatch。使用官方 clipped PPO 函数，ε=0.2；不再经过 TRL 原有的标量奖励组标准化，也不再添加第二份 KL loss。</li></ol>
<pre>原 B：步骤概率 → 整篇最低分 → 加答案奖励 → 整篇一个 advantage
PURE：有符号步骤分 → soft-min → 步骤末 token 奖励
      → 各 token 的后续回报 → RLOO / whitening → 逐 token advantage</pre>
<p>例：三个步骤的 p=[0.9,0.1,0.8]，对应 r=[0.8,−0.8,0.6]。soft-min 几乎集中在第二步，最差步骤之前与之后的回报因此不同。<strong>“最差步骤之后回报接近零”不等于其最终 advantage 为零</strong>，还要考虑 RLOO baseline 和 KL。</p>''' +
    '<p class="sources">直接依据：' + link('verl/workers/fsdp_workers.py','步骤分及 soft-min','#L1524-L1539') + '；' + link('verl/trainer/ppo/core_algos.py','return 与 RLOO','#L71-L228') + '；' + link('verl/trainer/ppo/ray_trainer.py','奖励、KL、优势的执行顺序','#L1193-L1227') + '；' + link('verl/utils/torch_functional.py','masked whitening','#L118-L148') + '。</p></section>')

sections.append('<section><h2>3. 和现在 B 的实质区别</h2>' + table(
    ['维度', '现有 B', '建议的 PURE-PRM 适配版'], [
        ['外部任务奖励', '答案 0/1 ＋ 0.2 × 最低正类概率', '只使用 PRM；答案验证只记录与评测，不进入训练'],
        ['PRM 分数', 'p ∈ [0,1]', '2p−1 ∈ [−1,1]'],
        ['聚合', '整条轨迹的一个硬最小值', 'soft-min 加权的步骤奖励＋后续累积回报'],
        ['优势', '同题组内标准化，一个回答一个标量', '官方 RLOO，每个 token 的数值可不同；一次全生成批次 whitening'],
        ['评分覆盖', '只有完整、可解析输出调用 PRM；其余为零', '不依赖答案是否可解析或正确；包含截断输出，单列统计'],
        ['KL', 'TRL loss 中 β=0.04', '在 token reward 中 β=0.001，遵循公开默认路径'],
        ['模型和参数', '1.5B actor＋rank16 LoRA，Qwen 7B PRM 冻结', '同一模型与 LoRA；不加 critic，不训练 PRM'],
        ['学习率', '5e−6', '两条新臂统一 1e−6，采用公开 YAML 默认值'],
        ['结论范围', '简单混合奖励是否有效', '在同一官方 RLOO 路径下，PRM 与答案奖励的对比']]) +
    '<p>这里包含真实的训练算法变化，不是“修复旧 B 的 bug”。原 B 仍是有效的简单混合对照；新结果不能覆盖原结果，也不能只用新 PURE 对旧 B 的差值声称 min-form 的独立贡献。</p></section>')

sections.append('''<section><h2>4. 最小而公平的实验组合</h2>''' + table(
    ['组别', '训练目标', '用途 / 状态'], [
        ['A_GRPO', '原结果奖励 GRPO', '已经完成，原样保留'],
        ['B_MIX', '原结果＋最低过程分 GRPO', '已经完成，原样保留'],
        ['P_VR', '官方 PURE/RLOO 核心，只使用原验答结果', '建议新增的配套控制；尚未启动'],
        ['P_PRM', '同一核心，只使用现有 Qwen PRM', '建议新增的过程奖励基线；尚未启动']]) +
    '''<p><strong>主比较固定为 P_PRM − P_VR。</strong>两组共享初始化、数据、提示词、RLOO、KL、学习率、loss、长度与生成预算，仅外部任务奖励来源不同。这样避免把 GRPO→RLOO、学习率或 KL 的变化误认为 PRM 的收益。</p>
<p>原 A/B 与新 P 两组仍可并排报告端到端准确率和成本，但属于整套训练配方比较。若以后要单独证明 min-form 的贡献，才增加同一 P 管线的 signed sum-form 消融；这不是本轮必须增加的训练。</p>
<p>不把“PURE-PRM+VR 的 10%”理解成给结果项乘 0.1：论文对应的是仅约 10% 题目提供可验证答案。首轮暂不引入该混合设置，避免额外的数据／奖励覆盖差异。</p></section>''')

cfg = plan['proposed_shared_settings']
sections.append('<section><h2>5. 具体运行规格与单卡可行性</h2>' + table(
    ['设置', '固定方案'], [
        ['起点', '原始 Qwen2.5-Math-1.5B base＋seed17 LoRA 初始化；不是从已训练 A/B checkpoint 继续训练'],
        ['数据与提示词', '原 6,993 道 MATH 训练题、同一冻结清单与原始 prompt。GSM8K 为迁移评测，不加入训练。不启用官方默认 curriculum 筛选。'],
        ['批量', '每题 8 条；每次生成 4 题＝32 条；actor microbatch 1、累积 32；PRM microbatch 2。'],
        ['长度', 'completion 2048；return 固定 padding 到 2048；actor 总上下文上限 4096；Qwen PRM 原生上限 4096，需按加 marker 后实际输入检查。'],
        ['精度与优化', 'BF16、LoRA rank16 / alpha32、checkpointing、AdamW、恒定 LR=1e−6、clip=0.2、一次 PPO 更新遍历。'],
        ['奖励与优势', 'T=0.1、PRM coefficient=0.5、γ=1、官方 RLOO、adv_norm=true、KL reward coefficient=0.001。'],
        ['预算', '建议两条新臂各 6M actor completion tokens、seed17。只是方案，不代表已获准新预算或已排队。完整 optimizer update 达到阈值后停，披露超额。'],
        ['评测', '固定最终 checkpoint，原 greedy MATH500 / GSM8K、同一 2048 上限与冻结验答器；不使用 final_only prompt，不用 PRM 重排，不挑测试集最佳 checkpoint。']]) +
    '''<p>无需再加载 critic 或训练一个 7B PRM；与已经跑通的 B 相比，主要新增的是很小的 token 奖励／回报／优势张量及一次显式 old-policy logprob 前向。模型规模没有上升，PRM microbatch 还从 4 减至 2，因此预计可在独占的 80GB A100 上运行。</p>
<p><strong>尚未测量：</strong>真实 PURE 接入后的 GPU 峰值、吞吐、并发下的运行时长。已有 B 的 44.88 GiB 是参考，不是新版本的实测。显存可容纳不代表当前 GPU 没有其他计算负载；不承诺当前共享状态下的 ETA。</p>
<p>若换成 A100 40GB，不能套用本结论。两组新实验的 6M 也是比论文更小的验证预算，不能据此期待或宣称复现论文分数。</p></section>''')

sections.append('''<section><h2>6. 需要改哪些代码</h2>
<p>以下是<strong>待实现的独立模块</strong>，本次没有接入训练入口。原 A/B 源码、运行目录和环境保留。</p>''' + table(
    ['独立模块', '工作内容', '保留／避免'], [
        ['vendor / pure_core.py', '按固定 commit 保留官方 compute_return、compute_rloo_outcome_advantage、masked_whiten、kl_penalty、compute_policy_loss，并保留来源与许可说明。', '使用默认 approximate min-form；不走源码标有 TODO 的 strict min 分支。'],
        ['pure_prm_adapter.py', '复用现有 FrozenQwenPRM 的权重校验与原生分数。构造 2p−1 和 soft-min；返回全部步骤分数、奖励及 raw span。', '保留 <extra_0>、AutoModel、Qwen chat template；不强制转换为 PURE 的 token-classification head 或换成单换行评分位置。'],
        ['step_token_alignment.py', '将规范化评分视图映射回未修改 actor 原文，再映射到实际 completion_ids 的步骤末 token；末步定位 EOS／截断末 token。', '不把 PRM token 索引当 actor token 索引；不靠重分词长度猜位置；CRLF、BPE 边界跨字符等必须实测。'],
        ['pure_trainer.py', '生成完整 batch 后缓存 old/ref logprob，构建固定 N 的 [32,2048] token rewards，算 RLOO 和一次 whitening，再同步切分 ids/mask/advantages，调用官方 PPO loss。', '不再继承原 B “标量优势必须不变”的 mixin；不让 TRL 再归一化；不重复加 KL；microbatch 静态设为 1。'],
        ['pure_reward_ledger.py', '区分外部任务奖励、KL、变换奖励、returns 和最终 advantage；记录所有输出的 PRM 覆盖、步数、终止、完整性、评分开销。', 'P_PRM 禁用原 B 的 valid_for_prm 筛选。无框、错答案、截断不自动置零；上下文溢出／对齐错误直接阻止该批更新并保留证据。'],
        ['plan / receipts / evaluator', '创建独立 P_VR / P_PRM 配置、恢复点和计数；沿用现有冻结评测器与题目清单。', '运行前查 PID、锁、receipts，避免重复启动；不覆盖 A/B 结果，不复用 A/B 末尾 checkpoint 当新起点。']]) +
    '''<p>空输出只有 EOS 时，将其作为一个终止步骤，按原生 Qwen 空 assistant＋marker 接口评分并单列记录；不能临时发明“空输出额外负奖励”。如果该映射不能通过无更新联调，就不能放行训练，也不能悄悄用零分替代。</p></section>''')

sections.append('''<section><h2>7. 源码与论文之间需要明确记录的细节</h2>
<ul><li><strong>公开默认不是一张卡。</strong>YAML 是 8 GPU、prompt batch64、8 rollouts、completion7168；论文写最大生成8192。单卡方案缩短长度、缩小生成 batch，并保留 LoRA，这是明确的规模适配。</li>
<li><strong>默认主路径是 RLOO。</strong>论文附录确有 GRPO 版本，但不能在当前 TRL 上把步骤分数塞进普通标量 reward callback 就称作该版本。</li>
<li><strong>代码将任务奖励与 KL 先合并，再统一计算 return/RLOO。</strong>论文式(7)把结果项与过程项的 baseline 分开写；公开 verl 路径不完全照此拆分。本方案以固定源码执行顺序为准，不把两套写法拼成第三种实现。</li>
<li><strong>固定 padding 的 N 有数值作用。</strong>官方 token RLOO 也作用在 VR 上。例如 K=2、N=6、长度2/4、结果1/0，错误回答在 whitening 前的优势是 −1/3，而普通整句 RLOO 是 −1。CPU 已验证这个差别，因此必须有 P_VR 配套控制。这里记录行为，不擅自把它“修正”为自己的算法。</li>
<li><strong>官方入口会强制设置提示词。</strong>main_ppo.py 强制加入“步骤用双换行分隔”的提示；本方案保留原 A/B 提示词，并明确标为协议适配。</li>
<li><strong>默认开启课程筛选。</strong>为了固定数据与预算，本方案两臂均关闭；不能让新方法悄悄多一个选题机制。</li>
<li><strong>学习率并非一份唯一官方数字。</strong>公开 YAML 默认1e−6，论文 PRM-only 主实验为5e−7。新两臂先统一公开默认1e−6，不根据测试结果调参；旧 LoRA A/B 是5e−6，因此新旧只是整套配方比较。</li></ul>
<p class="sources">来源：''' + link('verl/trainer/config/ppo_trainer.yaml','官方默认配置') + '；' + link('verl/trainer/main_ppo.py','强制 prompt','#L62-L73') + '；' + link('verl/trainer/ppo/ray_trainer.py','统一奖励路径','#L1193-L1227') + '；<a href="https://arxiv.org/html/2504.15275v3">论文 v3 §3.4 / §4.2 / App. F.2</a>。</p></section>')

sections.append('''<section><h2>8. 放行条件与公平比较</h2>
<ol><li><strong>CPU 数学核对已完成。</strong>其余步骤尚未执行：真实 tokenizer 对齐、模型无更新前向／反向、显存与吞吐测试。</li>
<li><strong>先无更新联调。</strong>同样 4 题×8 回答，长／短回答、CRLF、无分隔符、EOS、截断、空输出都覆盖。允许反向计算但不调用 optimizer.step；前后参数 hash 必须相同。正式训练必须等用户另行授权。</li>
<li><strong>检查是否混入答案监督。</strong>P_PRM 中打乱 gold answer 不应改变训练 reward/advantage；P_VR 则应跟随 outcome 改变。不得通过验答状态决定 P_PRM 是否打分。</li>
<li><strong>统计粒度固定。</strong>whitening 在完整生成 batch 上做，不能在 microbatch 上做；N固定2048；优势、mask、token ID 同步打乱和切片，检查 loss 入口张量原样保留。</li>
<li><strong>生成预算和成本分开。</strong>各6M completion tokens，另外报告 prompt tokens、PRM原生／padding输入tokens、评分时间、actor/reference前向次数、更新次数、实际 GPU 时间。等生成tokens不是等总算力。共享GPU的墙钟耗时不作效率证据。</li>
<li><strong>评测不追着结果改。</strong>最终固定checkpoint，原验答与prompt。报告正确／可解析但错／缺框／截断，以及长度、步数、重复、答案框与正文不一致的实例。配对bootstrap按题目抽样；seed17只能作先导结果。</li>
<li><strong>数据边界写清。</strong>保留当前训练与评测去重清单。原 PRM800K inventory 审计不能证明 Qwen PRM 的全部训练数据无污染；既有验证／测试已用于诊断，不宣称完全未见或盲测。不拿它们挑最佳超参数或 checkpoint。</li></ol>
<div class="callout"><strong>最终建议：</strong>不用继续换 PRM，也不把旧 B 改名覆盖。先实现并无更新验收一套来源明确的 PURE RLOO 核心适配版，之后若获准训练，仅成对运行 P_VR 与 P_PRM。这样才有一个能够回答“可靠过程训练机制下，现有 PRM 是否有帮助”的基线。</div>
<p>本次交付是源代码核对、CPU 数值证据和可执行规格；不是完整 trainer 已经写完或训练已成功。</p></section>''')

sections.append('''<section><h2>文件与来源</h2><ul>
<li><a href="baseline_plan.json">完整固定方案 JSON（仅规格，不是自动启动配置）</a></li>
<li><a href="source_manifest.json">官方 commit、下载文件、Git blob／SHA-256</a></li>
<li><a href="numerical_checks.json">CPU 数值验证结果</a> · <a href="check_official_math.py">验证脚本</a></li>
<li><a href="remote_readonly_evidence.json">服务器与数据清单证据</a> · <a href="current_evidence_pins.json">现有文件 SHA-256</a></li>
<li><a href="https://arxiv.org/html/2504.15275v3">PURE 原论文</a> · <a href="https://github.com/CJReinforce/PURE">PURE 官方仓库</a> · <a href="https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B">Qwen PRM 原生接口</a></li>
</ul><p class="muted">没有修改 reward/loss 或旧结果，没有安装／降级远端环境，没有训练或调度新作业。源码中的默认配置仅作为研究证据，不会自动执行。</p></section>''')

css = '''*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:#172333;font:16px/1.8 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}main{max-width:1140px;margin:auto;padding:40px 24px 70px}section{background:#fff;border:1px solid #dfe6ec;border-radius:14px;padding:28px 34px;margin:0 0 22px}h1{font-size:32px;line-height:1.4;margin:8px 0 18px}h2{font-size:23px;line-height:1.5;margin:0 0 18px}p{margin:12px 0}.eyebrow{color:#466780;font-size:13px;letter-spacing:.1em}.lead{font-size:19px}.callout{background:#edf7f4;border-left:4px solid #23806a;padding:16px 20px;margin:18px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:12px 15px;text-align:left;border-bottom:1px solid #dfe6ec;vertical-align:top}th{background:#eff4f8;white-space:nowrap}td:first-child{font-weight:600;min-width:125px}a{color:#1769a1;text-underline-offset:3px}code{background:#f0f3f6;padding:2px 4px;font-size:.9em;overflow-wrap:anywhere}pre{background:#162837;color:#e6f3f9;padding:18px;overflow:auto;border-radius:8px;font:14px/1.8 monospace}li{margin:9px 0}.sources,.muted{font-size:14px;color:#536675}strong{font-weight:650}@media(max-width:700px){main{padding:15px 10px}section{padding:20px 17px}h1{font-size:26px}h2{font-size:21px}td,th{padding:9px;font-size:14px}}@media print{body{background:white}main{max-width:none;padding:0}section{border:0;padding:10px 0;break-inside:auto}table{font-size:12px}.callout{background:#f4f7f7}}'''
page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PURE 官方实现核对与单卡 baseline 方案</title><style>' + css + '</style></head><body><main>' + ''.join(sections) + '</main></body></html>'
(HERE / "review.html").write_text(page)

before = json.loads((HERE / "current_evidence_pins.json").read_text())
assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in before.items())
receipt = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "source_commit": commit, "source_file_count": len(manifest["files"]),
           "numerical_checks_passed": checks["passed"], "old_evidence_files_unchanged": len(before),
           "training_started": False, "new_model_inference": False,
           "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir()
                     if p.is_file() and p.name != "audit_receipt.json"}}
(HERE / "audit_receipt.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps({"report": str(HERE / "review.html"), "plan": str(HERE / "baseline_plan.json"),
                  "old_files_unchanged": len(before), "training_started": False}, indent=2))
