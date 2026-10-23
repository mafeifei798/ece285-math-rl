"""Build a local evidence report; never touches original A/B or starts a job."""
from pathlib import Path
import hashlib
import html
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=HERE/'remote_evidence/runs/PURE_no_update_20261022_v3'
def read(p):return json.loads(p.read_text())
def esc(s):return html.escape(str(s))

result=read(RUN/'result.json')
accepted=read(HERE/'real_acceptance.json')
history=read(HERE/'remote_evidence/audits/PURE_integration_20261022_cpu_v3/offline_replay.json')
preserved=read(RUN/'preservation.json')
edge=read(RUN/'native_edge_forwards.json')
scored=[json.loads(s) for s in (RUN/'scored_alignment.jsonl').read_text().splitlines()]
generated=[json.loads(s) for s in (RUN/'generated_exact_ids.jsonl').read_text().splitlines()]
assert accepted['passed'] and result['passed']

gate=dict(implementation_checks_passed=True,real_no_update_passed=True,formal_training_started=False,
    optimizer_steps=0,ready_for_formal_training=False,
    reason='Observed empty-response high process score under the approved terminal-step adapter; reward protocol decision needed before training. No optimizer/resume lifecycle tested.',
    actor='Qwen2.5-Math-1.5B',prm='Qwen2.5-Math-PRM-7B',gpu='A100-SXM4-80GB',
    official_commit='b174e508a4469663543cfcb366be78cf199d20bf',
    historical_replay_exact_original_token_ids_available=False,
    measured=accepted,
    source_note='Final GPU run used final_gpu_tested_source.tar, reused the exact generated IDs from attempt 2, and retested separator-token positions plus standalone P-VR interface.',
    preservation=dict(local_files=7,remote_hashed=preserved['hashed_files'],remote_inventory=preserved['inventoried_files']))
(HERE/'acceptance_receipt.json').write_text(json.dumps(gate,indent=2,ensure_ascii=False)+'\n')
edge_rows=[]
for raw,score in zip(edge['texts'],edge['scores']):
    r=[2*p-1 for p in score['step_scores']];z=[-v/.1 for v in r]
    w=[math.exp(v-max(z)) for v in z];q=.5*sum(a*b for a,b in zip(r,w))/sum(w)
    edge_rows.append(f'<tr><td>{esc(repr(raw))}</td><td>{esc(", ".join(f"{p:.6f}" for p in score["step_scores"]))}</td><td>{q:+.6f}</td></tr>')
stage_rows=''.join(f'<tr><td>{esc(k)}</td><td>{v["seconds"]:.2f} s</td><td>{v["peak_allocated_bytes"]/1024**3:.2f} GiB</td></tr>' for k,v in result['stages'].items())
max_actor=max(len(r['prompt_ids'])+len(r['completion_ids']) for r in generated)
max_prm=max(r['native']['prm_input_tokens'] for r in scored)
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PURE P-VR / P-PRM · 独立接入与无更新验收</title>
<style>body{{margin:0;background:#f3f5f8;color:#1d2c3b;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1040px;margin:40px auto;padding:0 26px 60px}}h1{{font-size:31px;line-height:1.35}}h2{{font-size:22px;margin-top:34px}}small,.muted{{color:#5d6978}}.card{{background:white;padding:24px 28px;border-radius:14px;margin:20px 0;border:1px solid #dfe5eb}}.ok{{border-left:5px solid #2b7964}}.risk{{border-left:5px solid #b87516;background:#fffaf0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #dfe5eb;padding:10px 12px;text-align:left;vertical-align:top}}th{{background:#f0f4f8}}code{{font-size:13px;background:#eef1f5;padding:2px 5px;border-radius:4px}}a{{color:#2462a0;word-break:break-word}}.formula{{overflow:auto;padding:16px;background:#f0f4f8;font-family:ui-monospace,monospace;font-size:14px}}.pill{{font-size:13px;border-radius:20px;padding:4px 12px;background:#deeee8;color:#235b49}}li{{margin:6px 0}}</style>
<main><small>ECE285 · 2026-10-22 PT · 独立版本，旧 A/B 保留</small>
<h1>PURE P-VR / P-PRM<br>离线回放与真实模型无更新验收</h1>
<div class="card ok"><span class="pill">代码与无更新联调通过</span><p>1.5B 策略 + 7B PRM 在同一张 A100 80GB 上完成了生成、原生评分、old/reference log-prob、两组 loss 的反向传播。<b>optimizer 更新数为 0；策略与 PRM 的全部参数哈希前后相同。</b></p>
<p>这验证了计算链和这批样本的资源可行性。它不证明 PURE 提高了数学准确率，也不代表正式训练已获准或已启动。</p></div>
<div class="card risk"><b>正式训练前的实质问题：空回答得到了接近最高的 PRM 分数。</b>
<p>在固定边界题 <code>Solve x+1=3.</code> 上，空字符串的原生 PRM 分数为 <b>{accepted['empty_response_positive_probability'][0]:.6f}</b>。按照既定的单终端步骤接口和 PURE 转换，任务奖励约为 <b>+0.499918</b>。它比同题正常推导的奖励和更高。此处“空回答”是人工构造的边界输入，不是本次模型已经出现训练坍塌。</p>
<p>数值与接口均可复核；风险是未来策略可能利用这个入口。本次没有加惩罚、mask、结果门控或更换 PRM。<b>正式训练就绪状态保持 false</b>，先确定无有效步骤输出的奖励协议，并将该协议明确记录为适配选择。</p></div>
<h2>两组具体接了什么</h2><div class="card"><table><tr><th>环节</th><th>P-VR</th><th>P-PRM</th></tr>
<tr><td>任务奖励</td><td>原有纠错版验答结果，落在最后一个有效 token</td><td>Qwen PRM 每步 p，转换为 signed soft-min 稀疏奖励</td></tr>
<tr><td>是否需要 PRM</td><td>不需要；独立入口不受 PRM 分段、评分或上下文限制</td><td>原生 Qwen chat + &lt;extra_0&gt;，所有输出评分，含截断/无答案框</td></tr>
<tr><td>共同目标</td><td colspan="2">KL 进入 token reward（low_var_kl × 0.001）→ reverse cumulative sum → 官方 RLOO → 32 条回答有效 token 上白化一次 → 每条回答 PPO loss，再平均 32 条</td></tr>
<tr><td>隔离</td><td colspan="2">原 TRL 的 reward、advantage、loss、train 与 optimizer 创建入口均阻断。仅复用已固定版本的模型/LoRA/vLLM 初始化和权重同步设施。</td></tr></table>
<div class="formula">r_j = 2 p_j − 1<br>q_j = 0.5 · softmax(−r / 0.1)_j · r_j<br>R_t = sparse(q)_t − 0.001 · KL_t<br>N = 2048 固定；group = 8；batch = 4 × 8；actor microbatch = 1</div>
<p class="muted">官方函数固定于 <a href="https://github.com/CJReinforce/PURE/blob/b174e508a4469663543cfcb366be78cf199d20bf/verl/trainer/ppo/core_algos.py">PURE b174e508</a>，选取函数原文保留；LoRA、Qwen PRM 原生接口、短上下文及当前数据属于先前明确记录的适配，不称论文完整复现。</p></div>
<h2>验收结果</h2><div class="card"><table><tr><th>检查</th><th>真实结果</th><th>边界</th></tr>
<tr><td>实际 token 对齐</td><td>32/32 条；446 个步骤分数；26,557 tokens；含 5 条截断</td><td>直接使用 vLLM 返回的原始 ID；每个分数唯一映射；非末步落在分隔符 token，末步在 EOS 或最后生成 token</td></tr>
<tr><td>CRLF / Unicode / 空输出 / 多换行</td><td>12 个真实 tokenizer 用例通过；4 类异常拒绝；另做 4 个 PRM 真实边界前向</td><td>自然生成的 32 条里没有 CRLF；CRLF 覆盖来自固定边界输入</td></tr>
<tr><td>重复归一化</td><td>每组完整 batch 白化 1 次；32 次 loss 切片逐一核对未改变</td><td>没有再套 GRPO 的组标准差或微批次白化</td></tr>
<tr><td>独立 CPU 复算</td><td>P-VR 最大 advantage 误差 {accepted['checks']['P_VR']['max_advantage_error']:.2e}；P-PRM {accepted['checks']['P_PRM']['max_advantage_error']:.2e}</td><td>复算真实 GPU 保存的 rewards / returns / advantages</td></tr>
<tr><td>实际梯度</td><td>P-VR norm {accepted['checks']['P_VR']['gradient_norm']:.5f}；P-PRM {accepted['checks']['P_PRM']['gradient_norm']:.5f}；全部有限</td><td>初始 LoRA B=0，所以部分 LoRA A 梯度为 0 是预期；两组都未执行更新</td></tr>
<tr><td>标签隔离</td><td>翻转结果标签不改变 P-PRM；P-VR 会变化</td><td>只说明代码未泄漏 outcome；不证明过程分数正确</td></tr>
<tr><td>原始结果保护</td><td>远端 {preserved['hashed_files']} 个文件哈希、{preserved['inventoried_files']} 个文件清单一致；本地 7 个核心引用一致</td><td>旧 A/B 未覆盖，模型与训练配置未改动</td></tr></table></div>
<h2>显存与吞吐</h2><div class="card"><p>本进程 NVML 采样峰值 <b>{accepted['measured_peak_own_process_GiB']:.2f} GiB</b>，PyTorch allocated 峰值 <b>{accepted['measured_peak_torch_allocated_GiB']:.2f} GiB</b>。整卡（含其他用户作业）峰值 {accepted['measured_peak_device_total_GiB']:.2f} GiB。NVML 每 0.5 秒采样，短峰值可能未被捕获。</p>
<p>生成 <b>{result['generation_tokens_per_second']:.1f} completion tokens/s</b>；PRM <b>{result['prm_input_tokens_per_second']:.1f} input tokens/s</b>。32 条的 PRM 评分约 {result['prm_cost']['seconds']:.2f} 秒；P-VR / P-PRM 反向分别约 {result['stages']['P_VR_backward']['seconds']:.2f} / {result['stages']['P_PRM_backward']['seconds']:.2f} 秒。生成吞吐来自第二次尝试的真实生成；最终第三次尝试复用完全相同的 token ID，没有新采样。最终联调 {result['total_seconds']:.1f} 秒，包含两次全部参数哈希，不包含之前的生成时间。</p>
<table><tr><th>阶段</th><th>实测时间</th><th>PyTorch allocated 峰值</th></tr>{stage_rows}</table>
<p class="muted">这是共享 GPU、单 batch、初始参数的测量，不用于承诺 6M 训练工期。最长实际 actor 序列 {max_actor}，PRM 输入 {max_prm}；尚未覆盖总长 4096 的最坏组合。未分配 optimizer 状态，未测试恢复/多步训练。P-VR 联调时仍让 PRM 驻留显存，独立 P-VR 可以不加载它。不能由此声称 A100 40GB 可按原配置运行。</p></div>
<h2>历史 rollout 回放的限制</h2><div class="card"><p>旧 B 全部 8,960 条均纳入扫描，928 条含 CRLF。8,674 条重新分词后的长度与日志相同；其余 286 条不同。8,949 条可对重建 token 序列做映射，11 条拒绝重建（10 条超过 2048，1 条文本与解码不一致）。<b>长度相同也不能证明 token ID 与历史采样逐个相同。</b></p>
<p>从首批满足“完整组且已有 PRM 分数”的记录中固定取 32 组/256 条，回放两种信号；最大数值误差 {max(v['max_abs_reference_difference'] for b in history['batches'] for v in b.values()):.2e}。未给旧日志中的缺失评分补造分数；历史 KL=0 回放只验证公式，不能当作当时训练信号的精确重放。历史完整重建检查因此明确记录为 partial/failed；新 32 条原始 ID 联调另行通过。</p></div>
<h2>边界分数原始证据</h2><div class="card"><p>固定题：<code>Solve x+1=3.</code>。表中是 Qwen PRM 分数，不是经校准的数学正确率。奖励列为转换后各步骤任务奖励的和，尚未经过 RLOO。</p><table><tr><th>回答</th><th>每步 p</th><th>任务奖励和</th></tr>{''.join(edge_rows)}</table>
<p>核对官方 <a href="https://github.com/CJReinforce/PURE/blob/b174e508a4469663543cfcb366be78cf199d20bf/verl/workers/fsdp_workers.py#L1367">终端步骤逻辑</a>，其代码也保留最后一步评分；不存在可直接照搬的“空回答自动负奖”。不能把额外处罚包装成恢复官方公式。Qwen PRM 的空输入高分是本次适配的实测风险，不能推断原论文 PRM 一定同样失效。</p></div>
<h2>可复核文件</h2><div class="card"><ul>
<li><a href="../../experiments/pure_integration_20261022/trainer.py">独立模型接入与无更新反向</a> · <a href="../../experiments/pure_integration_20261022/signals.py">奖励/RLOO/loss</a> · <a href="../../experiments/pure_integration_20261022/alignment.py">步骤与 token 映射</a></li>
<li><a href="acceptance_receipt.json">总验收记录与停止边界</a> · <a href="real_acceptance.json">真实张量独立复算</a> · <a href="remote_evidence/runs/PURE_no_update_20261022_v3/result.json">GPU 实测结果</a></li>
<li><a href="remote_evidence/runs/PURE_no_update_20261022_v3/scored_alignment.jsonl">逐条 PRM 分数与映射</a> · <a href="remote_evidence/runs/PURE_no_update_20261022_v3/generated_exact_ids.jsonl">原始生成 ID</a></li>
<li><a href="remote_evidence/audits/PURE_integration_20261022_cpu_v3/offline_replay.json">旧 rollout 回放</a> · <a href="remote_evidence/runs/PURE_no_update_20261022_v3/preservation.json">参数及 A/B 保护证明</a></li></ul>
<p class="muted">第一次尝试在模型加载前因 pynvml 包缺失退出；改用已有 vLLM 内置接口后完成第二次尝试，未安装新依赖。第一次与第二次尝试的日志和 source tar 均保留。最终核对发现 67 个非末步奖励需从内容末 token 移到官方分隔符 token；第三次尝试使用完全相同的原始生成 ID 重做评分、两组反向和哈希验收，未新增采样。最终版也已移除 P-VR 对 PRM 元数据的不必要依赖。</p></div></main></html>'''
(HERE/'review.html').write_text(page)
paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(HERE/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'report':str(HERE/'review.html'),'files':len(manifest),'formal_training':False,'technical_checks':True,'reward_protocol_gate':False},indent=2))
