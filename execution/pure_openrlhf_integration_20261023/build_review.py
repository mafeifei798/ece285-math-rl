"""Render only completed evidence; never infer GPU success from static tests."""
from pathlib import Path
import json,html,hashlib
O=Path(__file__).resolve().parent
R=O/'remote/runs/PURE_OpenRLHF_no_update_20261023_v1'
A=O/'remote/audits/PURE_OpenRLHF_integration_20261023'
def read(p):return json.loads(p.read_text())
def esc(v):return html.escape(str(v))
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
r=read(R/'result.json');assert r['passed']
p=read(R/'preservation.json');mem=read(R/'nvml_peaks.json');edge=read(R/'edge_forwards.json')
cpu=read(A/'cpu_checks.json');tok=read(R/'tokenizer_checks.json')
independent=read(O/'independent_runtime_acceptance.json');assert independent['passed']
checks=[('固定源码',r['signal_receipts']['P_PRM']['pin']),('真实生成',f"32 条 / 4 题 × 8；{r['generation']['emitted_tokens']:,} 原始 token"),('token 与步骤边界',f"{r['exact_token_alignment_rows']} 条独立 oracle 通过，{r['mapped_boundaries']} 个评分边界"),('EOS',f"{r['original_truncated']} 条原始截断，{r['actor_processed_rows_changed']} 条由官方 process_sequences 改写末端"),('归一化','每臂一次 replay-buffer normalization；PolicyLoss 内零次'),('真实反向传播','两臂均产生有限非零 LoRA 梯度；optimizer 未创建，更新次数 0'),('原结果保全',f"{p['hashed_files']} 文件 hash / {p['inventoried_files']} 文件清单未变"),('参数保全','actor 和 PRM 的前后全参数 hash 相等'),('CPU 与 tokenizer',f"{cpu['test_count']} 项 signal 检查，{tok['cases']} 项真实 tokenizer 检查")]
measure=[]
for name,s in r['stages'].items():
 m=mem.get(name,{})
 measure.append((name,f"{s['seconds']:.2f}",f"{s['peak_allocated_bytes']/2**30:.2f}",f"{s['peak_reserved_bytes']/2**30:.2f}",f"{m.get('own_process_bytes',0)/2**30:.2f}"))
edge_rows=[]
for text,align,s in zip(edge['texts'],edge['alignments'],edge['scores']):
 edge_rows.append((repr(text),len(align['emitted_completion_ids']),align['actor_boundary_positions'],s['native_step_scores'],s['native_signed_scores']))
body='<p class="tag">ECE285 · 独立验证版本 · 0 次参数更新</p><h1>PURE OpenRLHF：真实模型联调验收</h1><p class="lead">完整接通官方 OpenRLHF 的奖励、baseline、KL、return、边界 mask、单次归一化与 PolicyLoss；本页记录真实 forward/backward 的结果，不代表训练收敛或准确率提升。</p>'
body+='<h2>验收结果</h2>'+table(['检查项','实际证据'],checks)
body+='<p>独立 CPU 重算也通过：P-VR / P-PRM 的 advantage 与 FP64 公式最大差分别为 2.45×10⁻⁷ / 2.07×10⁻⁷；两组各 32 次 loss 调用都没有再次归一化。<a href="independent_runtime_acceptance.json">独立验收收据</a></p>'
body+='<h2>两组到底是什么</h2><p><b>P-VR：</b>原验答器二值结果 + 官方序列 leave-one-out。<b>P-PRM：</b>官方 PURE 7B PRM 的原生 BF16 两分类 logits + signed softmin + 官方 token baseline。两组共享同一批原始生成、actor/reference、边界和 PolicyLoss；不混入旧 B 的 0.2×min 奖励。</p>'
body+='<p>原始 emitted IDs 与官方 processed IDs 分开保存。被截断的输出即使由官方路径补入 EOS，仍按原终止状态交给原验答器，不因此变成有效答案。PRM→actor token 使用官方离散 token 边界，无 CRLF 重写或重新编码推导。</p>'
body+='<h2>显存与吞吐</h2><p>共享 A100 80GB；以下是单次无更新联调实测。P-VR 阶段 PRM 也驻留显存，且没有 optimizer state，因此不能当作独立 P-VR 成本或正式训练峰值。</p>'
body+=f"<p>生成：<b>{r['emitted_tokens_per_second']:.1f} token/s</b>；PRM：<b>{r['prm_responses_per_second']:.2f} 条/s</b>、{r['prm_input_tokens_per_second']:.1f} input token/s。总耗时 {r['total_seconds']/60:.1f} 分钟（含加载、文件核对和参数 hash）。</p>"
body+=table(['阶段','秒','Torch allocated GiB','Torch reserved GiB','NVML 本进程 GiB'],measure)
body+='<h2>空回答与边界输入</h2><p>下列固定边界输入单独前向，不混入 32 条生成 batch。边界/EOS 的直接 advantage 被置零不等于 PRM 已学会判断“空回答无效”；Thank you 等内容 token 仍可能得到高分。评分质量风险与实现验收必须分开。</p>'+table(['固定文本','原始 token 数','actor 边界','BF16 p(correct)','BF16 signed score'],edge_rows)
body+='<p>CRLF 固定案例在官方 token 规则下仍只有一步。本版本没有移植旧 B 的 CRLF 规范化，以免把自定义处理混入官方路径。空 EOS 整条直接梯度为零由 CPU fixture 覆盖；固定空输入的 PRM 前向不能被说成“模型真实生成了空回答”。</p>'
body+='<h2>实现范围与限制</h2><ul><li>官方 openrlhf commit 41f9363b8cc4af044b7fc5f7f9bbbf62139e9553，PURE PRM revision b0cce6763f2317762c9cb46720c8fe7b07f44a88。</li><li>复用经 AST 和源文件 hash 校验的官方信号函数；模型/LoRA/vLLM 初始化仍用当前可运行的 TRL 后端，其 reward、loss、train、optimizer 路径被禁止。没有跑完整 Ray/DeepSpeed 分布式程序。</li><li>1.5B actor、LoRA r16、32 条、2048 token 上限、原数据/提示词/验答器是明确的资源与协议适配。</li><li>初始零 LoRA 下 old/reference 相同；真实非零 KL 未覆盖，非零 KL 由 CPU fixture 验证。</li><li>仅验证一个固定 batch，不据此选择模型、估计泛化准确率、宣称解决空回答或 PRM 漏判。</li><li>没有启动正式训练；没有改原 A/B 和 veRL 结果，也没有新增奖励或 loss。</li></ul>'
body+='<h2>可核验文件</h2><ul>'
for name in ['result.json','preservation.json','generated_exact_ids.jsonl','scored_alignment.jsonl','P_VR_signal_receipt.json','P_PRM_signal_receipt.json','P_VR_loss_audit.json','P_PRM_loss_audit.json','signals.pt','nvml_peaks.json','run_config.json']:
 body+=f'<li><a href="remote/runs/{R.name}/{name}">{name}</a></li>'
body+='</ul>'
css='body{font-family:system-ui,-apple-system,sans-serif;background:#f4f6f9;color:#1f2937;margin:0;padding:44px 22px}main{max-width:1100px;margin:auto;background:white;padding:42px;border-radius:18px}h1{font-size:32px;line-height:1.3}h2{margin-top:36px;font-size:22px}.tag{color:#16715c;font-weight:650}.lead{font-size:18px;line-height:1.8}p,li{line-height:1.8}table{border-collapse:collapse;width:100%;font-size:13px;table-layout:auto}th,td{border-bottom:1px solid #e5e7eb;padding:12px 9px;text-align:left;overflow-wrap:anywhere}th{background:#edf4f8}a{color:#176a92}li{margin-bottom:8px}'
(O/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PURE OpenRLHF 联调验收</title><style>'+css+'</style><main>'+body+'</main></html>')
print(O/'review.html')
