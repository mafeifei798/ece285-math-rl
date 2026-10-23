"""CPU-only accounting of saved signals. No model imports, updates, or rewards changed."""
from pathlib import Path
import hashlib,json,math
import torch

H=Path(__file__).resolve().parent
J=H/'evidence/P_PRM_batch_0002.json';T=H/'evidence/P_PRM_batch_0002_signals.pt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in (J,T)}
torch.set_num_threads(2)
d=json.loads(J.read_text());s=torch.load(T,map_location='cpu',weights_only=True)
assert all(t.device.type=='cpu' for t in s.values())
mask=s['mask'];boundary=s['boundary'];lens=mask.sum(-1);B,W=mask.shape
assert B==32 and d['signal_receipt']['group_size']==8
beta=d['signal_receipt']['kl_coefficient'];assert beta==.001
task=s['task'].double();totals=task.sum(-1)
checks={};errors={}
def check(name,got,expected,tolerance=3e-6):
 error=float((got.double()-expected.double()).abs().max());errors[name]=error
 assert error<=tolerance,(name,error,tolerance)
 checks[name]=True

# Independently reconstruct each peer baseline, boundary adjustment, and returns.
base=torch.zeros_like(task);peer_data=[]
for i in range(B):
 peer=[j for j in range((i//8)*8,(i//8+1)*8) if j!=i]
 reward_sum=float(totals[peer].sum());token_sum=int(lens[peer].sum());b=reward_sum/token_sum
 positions=d['rows'][i]['alignment']['actor_boundary_positions'];previous=-1
 for end in positions:
  n=end-previous-1;base[i,end]=task[i,end]-b*n;previous=end
 peer_data.append(dict(indices=peer,reward_sum=reward_sum,token_sum=token_sum,b_per_token=b))
check('baseline_boundary_rewards',base,s['baseline_reward'])
kl=(s['old_logps'].double()-s['ref_logps'].double())*mask
check('k1_kl',kl,s['kl'])
reward=base-beta*kl
check('point_rewards',reward,s['reward'])
raw=reward.flip(-1).cumsum(-1).flip(-1)
check('reverse_returns',raw,s['raw_returns'])
masked=raw.masked_fill(boundary,0)
check('boundary_masked_returns',masked,s['returns'])
# Recompute normalization from the saved, native FP32 returns to isolate each stage.
v=s['returns'][mask].double();nonzero=v!=0;count=int(nonzero.sum())
mu=v.sum()/count;variance=((v-mu)**2).sum()/count;sigma=variance.clamp(min=1e-8).sqrt()
expected=(s['returns'].double()-mu)/sigma
expected[(s['returns']==0)|~mask]=0
check('once_normalized_advantages',expected,s['advantages'])
checks['one_normalization_call']=d['signal_receipt']['normalization_calls']==1
checks['all_32_loss_traces_no_normalization']=all(t['normalization_calls_in_loss']==0 for t in d['update']['loss_trace'])
checks['all_loss_ratios_one']=all(t['avg_ratio']==1 for t in d['update']['loss_trace'])
checks['all_boundaries_zero']=bool((s['advantages'][boundary]==0).all())
assert all(checks.values())

selected=[];peer_rows={}
for i in (2,4,15):
 row=d['rows'][i];score=d['prm'][i];L=int(lens[i]);n=L-1
 assert row['alignment']['actor_boundary_positions']==[L-1]
 assert row['alignment']['process_sequences_changed'] is False
 logits=torch.tensor(score['step_logits'],dtype=torch.bfloat16)
 prob=logits.softmax(-1);signed=prob[:,1]-prob[:,0]
 assert float(signed[0])==float(task[i,L-1])==score['native_signed_scores'][0]
 assert float(prob[0,1])==score['native_step_scores'][0]
 b=peer_data[i]['b_per_token'];r=float(task[i,L-1]);baseline_delta=-b*n
 # Subtract the saved baseline point value to measure actual FP32 KL contribution.
 saved_q=float(s['baseline_reward'][i,:L].sum())
 content=s['returns'][i,:n].double();kl_delta=float((content-saved_q).mean())
 pre=float(content.mean());factor=n/L
 components=dict(prm=r*factor/float(sigma),baseline=baseline_delta*factor/float(sigma),
  kl=kl_delta*factor/float(sigma),centering=-float(mu)*factor/float(sigma))
 trace=d['update']['loss_trace'][i]
 actual=float(s['advantages'][i,:L].double().mean());total=sum(components.values())
 assert abs(total-trace['avg_advantage'])<1e-6
 assert abs(actual-trace['avg_advantage'])<1e-6
 entry=dict(row=i,text=row['emitted_completion'],question=row['question'],gold=row['gold'],
  outcome=row['outcome_details'],content_tokens=n,action_tokens_including_eos=L,
  native_logits=score['step_logits'],native_class_probabilities=prob.float().tolist(),
  prm_positive_score=float(prob[0,1]),signed_prm_reward=r,softmin_weight=1.,
  peer_baseline=peer_data[i],baseline_delta=baseline_delta,saved_baseline_adjusted_reward=saved_q,
  mean_kl_return_delta=kl_delta,mean_raw_content_return=pre,
  raw_content_return_range=[float(content.min()),float(content.max())],
  normalized_content_advantage_mean=float(s['advantages'][i,:n].double().mean()),
  eos_advantage=float(s['advantages'][i,L-1]),loss_mean_advantage=trace['avg_advantage'],
  final_mean_components_fixed_saved_statistics=components,
  component_sum_error=total-trace['avg_advantage'],scaled_loss=trace['scaled_loss'])
 selected.append(entry)
 group=i//8
 if str(group) not in peer_rows:
  peer_rows[str(group)]=[dict(row=j,length=int(lens[j]),step_count=d['rows'][j]['alignment']['step_count'],
   outcome=d['rows'][j]['outcome'],task_total=float(totals[j]),
   mean_advantage=d['update']['loss_trace'][j]['avg_advantage']) for j in range(group*8,(group+1)*8)]
positive=sum(max(t['avg_advantage'],0) for t in d['update']['loss_trace'])
assert before=={str(p):sha(p) for p in (J,T)}
report=dict(passed=True,source_pin=d['signal_receipt']['pin'],inputs=before,
 cpu_only=True,models_loaded=False,training_modified=False,checks=checks,max_abs_errors=errors,
 normalization=dict(mean=float(mu),standard_deviation=float(sigma),rescale_factor=1/float(sigma),
  nonzero_count=count,valid_actions=int(mask.sum()),zero_actions=int((~nonzero).sum()),
  definition='Mean uses nonzero count; variance numerator includes original zero boundary positions; original zeros restored after standardization. Computed over32 responses, not per question.'),
 rows=selected,peer_groups=peer_rows,
 positive_response_mean_advantage_fraction=sum(x['loss_mean_advantage'] for x in selected)/positive,
 attribution='High positive PRM scores are the source of the wrong positive signals. Negative peer baselines add only0.00095-0.00349; KL changes about1e-5-1e-4. Shared normalization rescales by2.163 and centers by+0.020714 before rescaling, but none of the selected raw content returns changes sign.',
 caveats=['Fixed-statistics additive accounting is not a counterfactual retraining or replacement reward.',
  'Positive response-mean advantage is a loss coefficient, not gradient norm or measured capability change.',
  'For these negative baselines, fixed reward and longer content would get a larger addition, not a smaller one; no claim of universally short-favoring baseline.',
  'PRM class scores are not calibrated probabilities of mathematical correctness.',
  'k1 KL estimates can be signed on individual tokens.'])
out=H/'prm_signal_decomposition.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(passed=True,normalization=report['normalization'],rows=[{k:r[k] for k in ('row','signed_prm_reward','baseline_delta','mean_kl_return_delta','mean_raw_content_return','loss_mean_advantage','final_mean_components_fixed_saved_statistics')} for r in selected],errors=errors),ensure_ascii=False,indent=2))

import html
labels=['错误答案243（标准答案77）','错误答案9（标准答案77）','要求补充已完整提供的题目']
table='';parts='';examples=''
for label,r in zip(labels,selected):
 vals=[label,f"{r['prm_positive_score']:.6f}",f"{r['signed_prm_reward']:.6f}",f"{r['baseline_delta']:+.6f}",f"{r['mean_kl_return_delta']:+.6f}",f"{r['mean_raw_content_return']:.6f}",f"{r['loss_mean_advantage']:.6f}"]
 table+='<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in vals)+'</tr>'
 c=r['final_mean_components_fixed_saved_statistics']
 parts+='<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in [label]+[f'{c[k]:+.6f}' for k in ['prm','baseline','kl','centering']])+'</tr>'
 examples+='<details><summary>'+html.escape(label)+'</summary><p>'+html.escape(r['question'])+'</p><pre>'+html.escape(r['text'])+'</pre></details>'
page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>三条短回答：PRM到advantage逐项拆解</title><style>body{{font:16px/1.75 system-ui,-apple-system,sans-serif;background:#f5f7fa;color:#172331;margin:0}}main{{max-width:1170px;padding:36px 28px;margin:auto}}h1{{font-size:30px}}h2{{font-size:21px}}section{{background:white;border:1px solid #dce3e9;border-radius:12px;padding:22px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #e3e8ec;text-align:left}}th{{background:#edf3f7}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:14px}}a{{color:#215d8c}}.scroll{{overflow:auto}}details{{padding:10px 0}}</style><main><p>ECE285 · 已保存真实训练数据 · P-PRM第2批 · CPU离线重算</p><h1>主要是PRM误判；归一化放大幅度，baseline不是主因</h1><p>原始训练继续运行。没有加载模型、执行梯度更新、改变奖励/验答规则或替换任何训练信号。</p>
<section><h2>1. 每一项实际加了多少</h2><div class="scroll"><table><tr><th>回答</th><th>PRM正类分数</th><th>原始有符号奖励</th><th>baseline增量</th><th>KL平均增量</th><th>归一化前正文均值</th><th>进入loss的平均advantage</th></tr>{table}</table></div><p>PRM分数不是校准过的数学正确率。原始奖励按原生BF16的 p正−p负 计算；这三条各只有一步，softmin权重均为1。即使移除baseline和归一化，三个原始奖励仍明显为正。</p></section>
<section><h2>2. baseline为何加分，而不是扣分</h2><pre>bᵢ = 同题其余7条的过程奖励总和 / 同题其余7条的实际token总数
该步骤调整后奖励 = 原始步骤奖励 − bᵢ × 该步骤正文token数</pre><p>三条的b分别为−0.000073237、−0.000118669、−0.000112549；正文长度为13、9、31。因此相减变成很小的加分：+0.000952、+0.001068、+0.003489。它没有把负的PRM分数翻成正数。此处baseline全为负，固定PRM分数和伙伴时，增加正文长度反而会加更多分，不能说本例是baseline特别偏爱短答案。</p></section>
<section><h2>3. KL和单次归一化</h2><p>KL采用有符号的k1样本估计，系数0.001，并在反向累加return之前扣除。三条平均影响约10⁻⁵至10⁻⁴，对正负号没有实质影响。</p><pre>本批32条的μ = {float(mu):.9f}，σ = {float(sigma):.9f}
正文token的advantage = (return − μ) / σ
EOS/步骤边界的advantage恢复为0；没有第二次归一化。</pre><p>除以σ约等于乘2.163，再因μ为负加上正偏移。这会改变数值尺度，但并不是这三条正号的来源。原始零边界不计入均值分母、仍计入方差分子，以上按已固定官方实现精确重算。</p><pre>错误答案9：
0.7890625 + 0.0010680 + 0.0000111 = 0.7901416
(0.7901416 + 0.0207139) / 0.4623165 ≈ 1.753897（9个正文token的均值）
1.753897 × 9/10 ≈ 1.578507（加上advantage=0的EOS后，loss实际均值）</pre></section>
<section><h2>4. 最终advantage的加法拆账</h2><div class="scroll"><table><tr><th>回答</th><th>PRM项</th><th>baseline项</th><th>KL项</th><th>归一化中心偏移项</th></tr>{parts}</table></div><p>每行各项之和就是实际loss平均advantage。此表固定本批已保存的μ和σ作代数拆解，不是重新运行另一种训练方法，也不是各项对参数梯度的因果比例。</p><p>官方loss先对每条回答的token取平均，再对32条回答取平均。短回答不会因为只有几个token就自动获得很小的回答级权重；这三条占该批正的逐回答平均advantage之和27.9%，但该数字不等于实际梯度范数占比，也不证明模型已经学会投机。EOS置零挡不住正文里的错误答案和无效澄清请求被正向强化。</p></section>
<section><h2>5. 重算验收与原文</h2><p>独立逐token重建baseline、k1 KL、反向累加、边界mask和一次归一化。与实际保存张量的最大误差为{max(errors.values()):.3g}；32条loss均无额外归一化。三条原始token没有被截断补EOS或重新分词。</p>{examples}<p>本检查可以确定实际信号来源，不能解释PRM内部为什么给高分，也不能凭这三个样本判定能力下降。后者仍需既定对照及统一评测。</p></section><p><a href="prm_signal_decomposition.json">完整数值与输入哈希</a> · <a href="decompose_prm_batch2.py">CPU重算代码</a> · <a href="review.html">试验进度</a></p></main></html>'''
(H/'signal_decomposition_review.html').write_text(page)
