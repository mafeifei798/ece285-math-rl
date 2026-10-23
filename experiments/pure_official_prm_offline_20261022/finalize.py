"""Finalize offline evidence; never changes a reward, model, or old artifact."""
from pathlib import Path
import hashlib, html, json, statistics, datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'execution/pure_official_prm_offline_20261022'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
s=read(OUT/'summary.json');before=read(OUT/'protected_before.json')
changed=[name for name,d in before.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=d]
assert not changed,changed
p=rows(OUT/'remote/scoring_single/scored.jsonl');q=rows(OUT/'remote/qwen_single/scored.jsonl')
old=rows(ROOT/'execution/pure_empty_boundary_20261022/remote/frozen_scoring_v2/scored.jsonl')
assert len(p)==len(q)==len(old)==288
qd=[];qe=[];cross=0
for a,b in zip(old,q):
 assert a['index']==b['index'] and a['input_sha256']==b['input_sha256']
 assert a['native']['input_ids']==b['native']['input_ids']
 assert a.get('alignment')==b.get('alignment') and a.get('alignment_error')==b.get('alignment_error')
 for x,y in zip(a['native']['step_scores'],b['native']['step_scores']):
  qd.append(abs(x-y));cross+=(x>.5)!=(y>.5)
 if b['empty']:qe.extend(abs(x-y) for x,y in zip(a['native']['step_scores'],b['native']['step_scores']))
metrics=rows(OUT/'per_response_metrics.jsonl');oldm=rows(OUT/'qwen_old_batch2_sensitivity/per_response_metrics.jsonl')
num=dict(primary_microbatch=1,qwen_old_batch2_vs_new_single=dict(rows=288,steps=len(qd),max_probability_abs_difference=max(qd),mean_probability_abs_difference=statistics.mean(qd),step_positive_sign_crossings=cross,empty_max_probability_abs_difference=max(qe),reward_rank_changed_rows=sum(a['variants']['QWEN_PRM']['rank_in_question_by_task_reward']!=b['variants']['QWEN_PRM']['rank_in_question_by_task_reward'] for a,b in zip(oldm,metrics)),whitened_terminal_sign_changed_rows=sum((a['variants']['QWEN_PRM']['whitened_terminal_advantage']>0)!=(b['variants']['QWEN_PRM']['whitened_terminal_advantage']>0) for a,b in zip(oldm,metrics) if a['advantage_available'])),pure_empty_batch2_vs_single={})
for v in ('native_raw','official_rl_prompt'):
 es=[r['variants'][v] for r in p if r['empty']]
 num['pure_empty_batch2_vs_single'][v]=dict(n=len(es),max_probability_abs_difference=max(e['batch2_vs_single_max_abs_error'] for e in es),differences_above_initial_tolerance=sum(e['batch2_vs_single_max_abs_error']>.005 for e in es),signed_score_sign_crossings=sum(e['batch2_sign_crossing'] for e in es),all_batch_input_and_padding_verified=all(e['batch2_input_and_padding_verified'] for e in es),max_repeated_single_error=max(e['direct_adapter_max_abs_error'] for e in es))
write('numerical_sensitivity.json',num)
qp=read(OUT/'remote/qwen_single/result.json');pp=read(OUT/'remote/scoring_single/result.json')
for e in (qp,pp):
 assert e['passed'] and e['model_frozen'] and e['optimizer_steps']==e['new_generations']==0 and not e['policy_model_loaded'] and e['primary_microbatch']==1
 assert e['input_sha256']==s['provenance']['input_sha256']
assert s['passed'] and s['advantage']['oracle_checks']==32 and s['advantage']['whitening_calls_per_batch']==1
assert s['skipped_batches'][0]['batch_id']=='B:210' and len(s['skipped_batches'])==1
receipt=dict(passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task='Official PURE PRM vs Qwen PRM on fixed saved288 responses',scoring_rows=288,empty_rows=9,advantage_rows=256,advantage_empty_rows=8,independent_signal_checks=32,original_local_protected_files=len(before),original_local_files_changed=changed,original_pure_release_sha256=qp['source_release_sha256'],primary_single_for_both=True,model_weights_verified=True,model_grads_disabled=True,training_started=False,optimizer_steps=0,new_generations=0,answer_verifier_changed=False,reward_or_loss_installed=False,token_ids='Reconstructed historical IDs; not exact original RL token replay',kl=0,conclusion='Both official PRMs assign positive scores and positive RLOO advantages to empty completions in the fixed diagnostic sample; swapping PRM does not remove this failure mode.',initial_batch2_failure_preserved=True,numerical_sensitivity=num)
write('acceptance_receipt.json',receipt)
report=(OUT/'review.html').read_text()
qtime=sum(c['seconds'] for c in qp['costs'])
section=f'''<div class="box"><b>验收结论：换评分器没有解决空回答问题。</b><p>Qwen 与 PURE 官方 PRM 都对 9/9 条空回答给出正分。8 条可回放空回答的 raw RLOO 和白化后 advantage 均为正；P-VR 为 0/8。独立公式核对通过 32 组，whitening 每批仅调用一次。因此这个现象在归一化之前已经存在，不能归咎于重复归一化。</p><p>例如 A:16 中的一条空回答：PURE 正类概率 0.9914 → signed 分 0.9828 → 任务奖励 0.4914 → raw advantage 0.5230 → 白化后 3.1818；它在同题 8 条回答中奖励最高，压过 4 条答案正确的回答。全样本逐例表在下方。</p><p><b>与旧 B 的边界：</b>旧 A/B 对这些空回答没有调用 PRM，原奖励都是 0。这是对 PURE 信号的离线反事实检查，不能拿它解释旧 B 的历史准确率下降，也不能由此断言 PRM 整体无用。</p></div>
<div class="box"><b>为何比较 PURE，而不是放弃 Qwen？</b><p>解题策略仍为 Qwen2.5-Math-1.5B。本次只冻结加载两个评分器，没有加载策略模型。PURE-PRM-7B 也基于 Qwen2.5-Math-7B；Qwen 官方 PRM 基于其 Instruct 版本。两个 PRM 分别使用自己的官方输入格式。PURE 模型卡将 PRM800K 的 1/0 标签并为正类，因此 p(正类)不能直接解释为“答案正确的概率”。此次比较用于检验接近官方 PURE 实现后问题是否消失，不默认更换后的模型更好。</p><p><a href="https://huggingface.co/jinachris/PURE-PRM-7B/tree/b0cce6763f2317762c9cb46720c8fe7b07f44a88">固定版本 PURE 模型</a> · <a href="https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B/tree/0610740060112df12585d00a1c5f4624d2f59051">固定版本 Qwen 模型</a> · <a href="https://github.com/CJReinforce/PURE/tree/b174e508a4469663543cfcb366be78cf199d20bf">固定版本官方代码</a></p></div>
<div class="box"><b>数值一致性与协议调整</b><p>最初的 PURE batch2 评分被单条/批量差异检查拦下，失败日志保留，未放宽容差。主结果改为两种 PRM 都做单条评分；PURE 两个提示词版本各 288 条完整保留。空回答单条重复 forward 的误差全部为 0。PURE batch2 对单条在原生题目上最大概率差 {num['pure_empty_batch2_vs_single']['native_raw']['max_probability_abs_difference']:.5f}，不应称为完全一致；但 18 个空回答×格式检查没有正负号跨越。Qwen 单条复核亦完整保留；其批量/单条 2,919 个步骤分中，7 个跨过正负阈值，6 条回答的组内奖励排名改变，1 条非空回答的白化后终点优势改变正负。空回答的高分和正优势结论均不变。批量敏感性见 <a href="numerical_sensitivity.json">数值明细</a>。</p><p>空回答按官方 RL 的终止边界作为一段空步骤打分：题目后追加评分换行。模型卡若直接传入空 steps 列表会没有评分位置。因此这里不是宣称模型卡规定“空回答必有一分”，而是在核验官方 RL 边界接入后的实际分数。</p></div>
<div class="box"><b>A100 实测成本（仅冻结评分器，不含策略训练）</b><table><tr><th>评分器</th><th>完整任务耗时</th><th>CUDA峰值分配/保留</th></tr><tr><td>Qwen 单条，288 回答</td><td>{qp['elapsed_seconds']:.1f} 秒</td><td>{qp['peak_allocated_GiB']:.2f}/{qp['peak_reserved_GiB']:.2f} GiB</td></tr><tr><td>PURE，两个格式各288回答及空回答复核</td><td>{pp['elapsed_seconds']:.1f} 秒</td><td>{pp['peak_allocated_GiB']:.2f}/{pp['peak_reserved_GiB']:.2f} GiB</td></tr></table><p>Qwen 主要 scoring 调用合计 {qtime:.1f} 秒；PURE 每个格式约 176–178 秒且包含额外空回答复核。GPU 与其他作业共享，输入 token 数与复核工作不同，这不是严格吞吐赛跑。下载耗时未计入。</p></div>'''
marker='<div class="box"><b>先看空回答：</b>'
assert marker in report
report=report.replace(marker,section+marker,1)
report=report.replace('<h2>成本与限制</h2>','<p>验收凭据：<a href="acceptance_receipt.json">acceptance_receipt.json</a>；原始 132 个本地保护文件均未改动。当前没有新训练、无新奖励、无验答修改。</p><h2>完整 PURE 成本凭据</h2>',1)
(OUT/'review.html').write_text(report)
(OUT/'HANDOFF.txt').write_text('Completed frozen offline comparison. No training authorized or started.\nPrimary report: review.html\nPrimary PURE/Qwen scores: remote/scoring_single/scored.jsonl and remote/qwen_single/scored.jsonl\nFixed input: ../pure_empty_boundary_20261022/inputs.jsonl\nCPU replay: experiments/pure_official_prm_offline_20261022/analyze.py with explicit --qwen-scores, --pure-scores, --evidence paths above. Then finalize.py.\nQwen original batch2 retained. Secondary old-batch comparison: qwen_old_batch2_sensitivity/.\nPURE first batch2 attempt failed .005 check, retained remote/supporting/scoring.log; single scoring is primary.\n9 empty positive scores for both PRMs; 8 replayable empties positive raw and whitened advantage. This is not historical B advantage or proof of whole-model training quality.\nUser has NOT authorized formal training.\n')
manifest={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
write('artifact_manifest.json',manifest)
print(json.dumps(dict(passed=True,protected=len(before),numerical=num),indent=2))
