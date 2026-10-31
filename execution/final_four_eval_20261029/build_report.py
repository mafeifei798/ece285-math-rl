"""Local report/figures from completed receipts. Incomplete results are labelled running."""
from pathlib import Path
import json,html,os,hashlib,csv,datetime
HERE=Path(__file__).resolve().parent
labels={'initial':'Initial','A':'A: outcome','C_mix':'C-mix','C_bounded':'C-bounded'}
def load(p):return json.loads(p.read_text())
def esc(x):return html.escape(str(x))
style='''body{margin:0;background:#f2f5f8;color:#172a3a;font:16px/1.85 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1120px;margin:32px auto;background:white;border-radius:14px;padding:38px 44px}h1{font-size:29px;line-height:1.4}h2{font-size:21px;margin-top:34px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px 12px;border-bottom:1px solid #dce4eb;text-align:left;vertical-align:top}th{background:#f0f4f7}a{color:#155c91}small,.meta{color:#5b6d7d;font-size:13px}.note{padding:15px 20px;background:#edf5fa;border-left:4px solid #2876a0}.warning{background:#fff7e7;border-left-color:#c18a26}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:16px;font-size:13px}img{width:100%;height:auto}.scroll{overflow:auto}code{overflow-wrap:anywhere} @media(max-width:700px){main{margin:0;padding:22px 16px}}'''
def page(body):return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>数学RL四模型统一评测</title><style>'+style+'</style><main>'+body+'</main></html>'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
if not (HERE/'comparison.json').exists():
 d=load(HERE/'latest_snapshot.json') if (HERE/'latest_snapshot.json').exists() else {}
 p=d.get('progress') or {};body='<p class="meta">2026-10-29 · 一张共享A100 · 仅评测</p><h1>四模型统一评测进行中</h1><div class="note">初始模型、A、C-mix、C-bounded；MATH500＋GSM8K，共7,276条回答。不启动新训练，不加载PRM，不更改提示或验答规则。</div>'
 body+=f'<p>已完成 {p.get("done",0):,} / 7,276。当前状态：{esc((d.get("controller") or {}).get("stage","准备中"))}。未完成前不展示最终准确率。</p>'
 body+=table(['模型','MATH500','GSM8K'],[[labels[a],(p.get('counts') or {}).get(a,{}).get('math500',0),(p.get('counts') or {}).get(a,{}).get('gsm8k',0)] for a in labels])
 body+='<p><a href="CONTINUATION.txt">范围与续跑说明</a> · <a href="dispatch.json">唯一派发记录</a> · <a href="admission.json">CPU预检</a> · <a href="latest_snapshot.json">真实状态</a></p>'
 (HERE/'review.html').write_text(page(body));print('Running report updated');raise SystemExit

c=load(HERE/'comparison.json');assert c['passed'] and load(HERE/'retrieval_acceptance.json')['passed']
evidence=HERE/'evidence/runs/Final_four_eval_seed17_1M_20261029'
contract=load(evidence/'contract.json');curves=load(evidence/'training_curves.json')
rows=[]
for arm in labels:
 math=c['summary'][arm]['math500'];gsm=c['summary'][arm]['gsm8k'];t=c['training'].get(arm,{})
 rows.append([labels[arm],f'{t.get("accounting",{}).get("generated_completion_tokens",0):,}',f'{int(math["correct"])}/500 ({math["accuracy"]:.2%})',f'{int(gsm["correct"])}/1319 ({gsm["accuracy"]:.2%})'])
with (HERE/'accuracy.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['arm','dataset','n','correct','accuracy','generated_tokens','inference_seconds','grading_seconds','empty','short_wrong','truncated','no_extractable_answer'])
 for arm in labels:
  for dataset,d in c['summary'][arm].items():writer.writerow([arm,dataset,*[d[k] for k in ['n','correct','accuracy','generated_tokens','inference_seconds','grading_seconds','empty','short_wrong','truncated','no_extractable_answer']]])
paired_rows=[]
for dataset,comparisons in c['paired'].items():
 for name,d in comparisons.items():paired_rows.append([dataset,name,f'{d["delta_pp"]:+.2f}',f'[{d["paired_question_bootstrap95_pp"][0]:+.2f}, {d["paired_question_bootstrap95_pp"][1]:+.2f}]',f'{d["wins"]}/{d["losses"]}',f'{d["exact_mcnemar_two_sided_p"]:.4g}'])
diagnostic=[]
for arm in labels:
 for dataset,d in c['summary'][arm].items():diagnostic.append([labels[arm],dataset,d['empty'],d['short_wrong'],d['truncated'],d['no_extractable_answer'],d['multiple_boxes'],d['early_last_box'],f'{d["mean_generated_tokens"]:.1f}'])
def error_category(r):
 if r['finish_reason']!='stop':return '截断'
 if r['reward']:return '正确'
 if r.get('answer') is None or r['status'] in ['unparseable','missing_or_invalid_box'] or r['status'].startswith('verifier_error:'):return '无可靠可验答案'
 return '完整可解析但验答不通过'
decomposition={};decomposition_rows=[]
for dataset in ['math500','gsm8k']:
 rr={a:{r['problem_id']:r for r in [json.loads(s) for s in (evidence/a/f'{dataset}.jsonl').read_text().splitlines()]} for a in labels}
 bins={k:{'wins':0,'losses':0} for k in ['截断','无可靠可验答案','完整可解析但验答不通过']}
 for ident in rr['A']:
  a=rr['A'][ident];b=rr['C_bounded'][ident]
  if b['reward']>a['reward']:bins[error_category(a)]['wins']+=1
  elif a['reward']>b['reward']:bins[error_category(b)]['losses']+=1
 n=len(rr['A'])
 assert sum(x['wins']-x['losses'] for x in bins.values())==c['paired'][dataset]['C_bounded minus A']['wins']-c['paired'][dataset]['C_bounded minus A']['losses']
 decomposition[dataset]=bins
 for category,d in bins.items():decomposition_rows.append([dataset,category,d['wins'],d['losses'],d['wins']-d['losses'],f'{100*(d["wins"]-d["losses"])/n:+.2f}'])
(HERE/'error_category_decomposition.json').write_text(json.dumps({'scope':'Mutually exclusive verifier/output categories on paired wins/losses, not a causal decomposition of mathematical ability.','data':decomposition,'official_scores_unchanged':True},ensure_ascii=False,indent=2)+'\n')
costrows=[]
for arm in labels:
 t=c['training'].get(arm);ss=c['summary'][arm]
 costrows.append([labels[arm],f'{t["accounting"]["prm_input_tokens"]:,}' if t else '0',f'{t["accounting"]["prm_forward_seconds"]:.1f}' if t else '0',f'{t["successful_training_seconds_sum"]/60:.1f}' if t else '0',f'{t["peak_allocator_reserved_GiB"]:.2f}' if t else '—',f'{sum(v["generated_tokens"] for v in ss.values()):,}',f'{sum(v["inference_seconds"] for v in ss.values())/60:.1f}'])

os.environ.setdefault('MPLCONFIGDIR',str(HERE/'plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,2,figsize=(10,4))
colors=['#8998a6','#3975a8','#d49b44','#318977']
for ax,dataset in zip(axes,['math500','gsm8k']):
 vals=[100*c['summary'][a][dataset]['accuracy'] for a in labels]
 bars=ax.bar(list(labels.values()),vals,color=colors,width=.65)
 for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+.5,f'{val:.2f}%',ha='center',fontsize=9)
 ax.set_ylim(0,min(100,max(vals)+12));ax.set_title(dataset.upper());ax.set_ylabel('Greedy accuracy (%)');ax.tick_params(axis='x',rotation=15)
fig.suptitle('Fixed final checkpoints; ~1M training completion tokens; one training seed')
fig.tight_layout();fig.savefig(HERE/'accuracy.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
names=['A minus initial','C_mix minus A','C_bounded minus A','C_bounded minus C_mix','C_bounded minus initial']
for ax,dataset in zip(axes,['math500','gsm8k']):
 data=[c['paired'][dataset][n] for n in names];point=np.array([d['delta_pp'] for d in data]);lo=np.array([d['paired_question_bootstrap95_pp'][0] for d in data]);hi=np.array([d['paired_question_bootstrap95_pp'][1] for d in data]);y=np.arange(len(names))
 ax.errorbar(point,y,xerr=np.array([point-lo,hi-point]),fmt='o',capsize=3,color='#235e83');ax.axvline(0,color='#9ba6b0',lw=1);ax.set_yticks(y,names);ax.invert_yaxis();ax.set_title(dataset.upper());ax.set_xlabel('Accuracy difference (percentage points)')
fig.suptitle('95% paired question bootstrap intervals; not training-seed uncertainty');fig.tight_layout();fig.savefig(HERE/'paired_differences.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.7))
for a,color in zip(['A','C_mix','C_bounded'],colors[1:]):
 v=curves[a];x=[r['generated_completion_tokens']/1e6 for r in v]
 axes[0].plot(x,[r['correct_completions']/r['generated_completions'] for r in v],label=labels[a],color=color)
 axes[1].plot(x,[r['loss_wrong_positive'] for r in v],label=labels[a],color=color)
 axes[2].plot(x,[r['truncated_completions']/r['generated_completions'] for r in v],label=labels[a],color=color)
for ax,title in zip(axes,['Cumulative rollout correctness','Wrong positive-advantage rows','Cumulative truncation fraction']):ax.set_title(title);ax.set_xlabel('Generated training tokens (millions)')
axes[0].legend();fig.suptitle('Training diagnostics; these curves are not test accuracy');fig.tight_layout();fig.savefig(HERE/'training_diagnostics.png');plt.close(fig)

body='<p class="meta">ECE285 · 完整交付版 · 四模型统一测试 · '+datetime.datetime.fromtimestamp(c['finished_unix'],datetime.timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M PT')+'</p><h1>单卡小模型数学推理RL：统一评测报告</h1>'
body+='<div class="note">初始模型＋A／C-mix／C-bounded现有最终checkpoint。三个训练臂均48次更新、约1M实际生成token。统一提示、greedy、2048上限、冻结验答器；评测不调用PRM、不投票、不选择checkpoint。本轮新增训练为0。</div>'
delta_descriptions=[]
for dataset in ['math500','gsm8k']:
 d=c['paired'][dataset]['C_bounded minus A'];interval=d['paired_question_bootstrap95_pp']
 delta_descriptions.append(f'{dataset.upper()}上C-bounded−A为{d["delta_pp"]:+.2f}pp，题目配对95%区间[{interval[0]:+.2f},{interval[1]:+.2f}]pp')
headline='；'.join(delta_descriptions)+'。'
all_positive=all(c['paired'][n]['C_bounded minus A']['paired_question_bootstrap95_pp'][0]>0 for n in ['math500','gsm8k'])
interpretation=('在本轮单训练seed和固定最终checkpoint上，两项题目配对区间都支持正向差异；还没有跨训练seed或更长预算的确认。' if all_positive else '本轮没有建立C-bounded在两个基准上都稳定优于A的证据；不能把修复某类学习信号等同于已提升整体数学能力。')
body+='<p><strong>核心结论：</strong>'+headline+interpretation+'</p>'
body+='<h2>1. 数学准确率</h2>'+table(['模型','训练生成token','MATH500','GSM8K'],rows)+'<img src="accuracy.png" alt="四模型数学准确率">'
body+='<p>这些是MATH500／GSM8K测试集结果，与此前500题MATH验证集是不同集合。不能与历史6M checkpoint混成同预算比较。完整预测保留原始token、文本、终止原因、冻结评分与adapter hash。</p>'
body+='<h2>2. 同题比较与不确定性</h2>'+table(['数据集','比较','差值pp','配对95%区间pp','胜／负','McNemar p'],paired_rows)+'<img src="paired_differences.png" alt="配对准确率差和区间">'
body+='<p>每项10,000次题目配对bootstrap，固定seed285；McNemar为精确双侧、未作多重比较校正。训练只有seed17，不能据此证明跨训练seed稳定。区间跨0不等于等效，点估计更高不等于已确认提升。</p>'
body+='<h2>3. 输出健康与主要问题</h2>'+table(['模型','数据集','空答','短错答≤32token','截断','未提取答案','多框','末框偏早','平均token'],diagnostic)
body+='<p>短正确解答不算投机；“短错答”“多框”“末框偏早”只是诊断信号。末框偏早指最后一个框起点处于文本前25%，不能据此判定正文与答案冲突。数学推导有效性需另行审读，不能用PRM自己的评分替代。</p>'
body+='<p>下面把C-bounded相对A的同题胜负，按<strong>答错一侧的输出／验答状态</strong>分类。类别互斥并与准确率净差对账，但不是数学能力变化的因果分解；“完整可解析但验答不通过”也不自动等于整段推导错误。</p>'+table(['数据集','错误侧类别','C-bounded胜','C-bounded负','净题数','净差pp'],decomposition_rows)
body+='<p><a href="evidence/runs/Final_four_eval_seed17_1M_20261029/case_review_pack.json">分层实际回答审阅包</a>保留同题四模型输出。它按胜负定向抽样，用于找原因，不能估计总体错误比例。</p>'
if (HERE/'qualitative_review.json').exists():
 review=load(HERE/'qualitative_review.json');body+='<div class="note warning">以下包含训练诊断与测试同题样例，逐例标注来源。它们是助手的定性审阅，不是人工金标准，不改变正式验答或准确率，也不估计总体错误比例。</div>'
 for case in review.get('cases',[]):
  body+='<p><strong>'+esc(case['problem_id'])+'</strong> · '+esc(case['finding'])+'</p><small>'+esc(case.get('scope',''))+'</small><blockquote>'+esc(case.get('evidence',''))+'</blockquote><p>'+esc(case.get('explanation',''))+'</p>'
else:body+='<p class="meta">尚无单独标注的数学过程金标准；不把自动结构统计解释为已确认的数学机制。</p>'
body+='<h2>4. 实际资源成本</h2>'+table(['模型','训练PRM输入token','评分秒','成功训练分钟','训练reserved峰值GiB','本轮评测token','评测生成分钟'],costrows)
cost=c['cost'];body+=f'<p>评测合计生成 <strong>{cost["evaluation_generated_tokens"]:,}</strong> token；生成进程总时长 {cost["generation_process_seconds"]/60:.1f} 分钟，其中启动 {cost["startup_seconds"]:.1f} 秒。每15秒采样的本评测进程及子进程显存峰值约 {(cost["own_GPU_memory_peak_MiB"] or 0)/1024:.2f}GiB。抽样可能漏掉瞬时峰值，不能和训练allocator reserved直接等同。</p>'
dispatch=load(evidence/'dispatch.json');finished=load(evidence/'controller.json')
assert finished['stage']=='complete'
full_wall=finished['unix']-dispatch['unix']
body+=f'<p>从唯一派发到生成＋CPU验收完成的总墙钟时间为 {full_wall/60:.1f} 分钟（不含本地下载和报告整理）。这是共享GPU下的本轮实际成本。</p>'
body+='<p>GPU1与其他作业共享，各臂按题目块交错执行；单臂时间受共享负载影响，不能作为独占GPU速度结论。训练时间仅合计成功smoke与continuation，之前的失败联调、环境修复另列为历史开销。PRM输入token、训练生成token、评测生成token分别计账。</p>'
body+='<p><strong>共享准备成本另计：</strong>本轮的冻结C来自先前PRM适配。该次记录中C适配约61.9分钟、约2.04M输入token；D适配约43.9分钟、约2.04M输入token，但D未用于本轮。C的适配是两个过程奖励臂共享的一次准备成本，不应重复计入每臂，也不能当成免费。本表不是整个项目所有历史探索的累计账单。<a href="../cd_prm_7b_pair_20261028/RESULTS.txt">原适配成本与诊断记录</a>。</p>'
body+='<h2>5. 训练机制及已有证据</h2><img src="training_diagnostics.png" alt="训练正确率、错答正优势与截断诊断">'
body+='<p>A使用纯结果优势；C-mix将y+0.2m统一归一化；C-bounded保留相同结果优势，仅在答对子集内加入0.2乘以中心化过程分，合成后不再次归一化。C-mix实际优势减去结果优势的差不能视为唯一可分离的过程分量。</p><p>门控保证错答不因过程项获得正优势，这不是独有创新；它是否提高数学表现由上面的共同测试检验。正确终答也不能认证全部推导正确。没有与完整PAPO官方配方竞争或跨模型推广的证据。</p>'
body+='<pre>A：          aᵢ = (yᵢ − mean(y)) / (std(y) + 10⁻⁴)\nC-mix：      aᵢ = (rᵢ − mean(r)) / (std(r) + 10⁻⁴)，rᵢ = yᵢ + 0.2mᵢ\nC-bounded：  aᵢ = aᵢ(outcome) + 0.2yᵢ(mᵢ − mean(m | y=1))</pre><p>每组8条，std为样本标准差；y是冻结验答器的0/1结果，m是相同冻结C的最低步骤分。无有效过程评分的输出用0占位；无答对回答的组，C-bounded过程项定义为0。其过程增量绝对值至多0.2，合成后不再归一化。<a href="../../experiments/c_bounded_pilot_20261029_v2/advantages.py">实际优势实现</a>。</p>'
if (HERE/'training_reward_diagnostics.json').exists():
 td=load(HERE/'training_reward_diagnostics.json');srows=[]
 for arm in ['A','C_mix','C_bounded']:
  for group,d in td['arms'][arm]['strata'].items():srows.append([labels[arm],group,d['groups'],d['rows'],d['wrong_positive_rows'],f'{d["mean_abs_advantage"]:.4f}'])
 body+=table(['模型','组类型','题组数','回答数','错答正优势','平均|优势|'],srows)
 body+='<p>C-mix的82条错答正优势中，80条来自全错组，2条来自混合组。并非每条错答正优势都一定有害：答错也可能含有合理过程。定向案例确认了其中有明确错误与跑题回答获得正优势，但没有证明它们造成测试准确率差异。</p><p>正优势表示策略项的系数方向，不是参数梯度占比，也不保证整次更新后该回答概率净上升；clipping、KL和其他样本共同影响更新。结果优势为0也不等于总梯度为0，KL仍可能产生梯度。</p>'
 scoring_rows=[]
 for arm in ['C_mix','C_bounded']:
  raw=[json.loads(s) for s in (HERE/'evidence/runs/C_bounded_triplet_seed17_1M_20261029_v2'/arm/'rollouts.jsonl').read_text().splitlines()]
  scored=[r for r in raw if r['valid_for_process_reward']]
  early=[r['min_score'] for r in scored if r['optimizer_step_before_update']<16]
  late=[r['min_score'] for r in scored if r['optimizer_step_before_update']>=32]
  scoring_rows.append([labels[arm],len(scored),f'{np.mean([r["step_count"] for r in scored]):.2f}',f'{np.mean([r["step_count"]==1 for r in scored]):.1%}',f'{np.mean(early):.4f}',f'{np.mean(late):.4f}'])
 body+=table(['训练臂','实际PRM评分回答','平均评分步骤','单步骤占比','前16步平均最低分','后16步平均最低分'],scoring_rows)
 body+='<p>步骤统计只包含实际经过PRM评分的回答；A没有请求PRM，不把其占位0当作评分结果。前后分数来自不同的生成回答，受题目和输出分布影响，不是PRM能力或奖励投机的因果检验。</p>'
if (HERE/'signal_case_analysis.json').exists():
 d=load(HERE/'signal_case_analysis.json')['actual_mixed_group']
 body+=f'<p><strong>混合组也会出现：</strong>实际题组{esc(d["problem_id"])}中，一条错答reward={d["wrong_row_reward"]:.6f}，略高于整组均值{d["mean"]:.6f}，所以实际advantage={d["actual_FP32_advantage"]:+.6f}。FP64离线复算一致。</p>'
 body+='<p>一个说明性反例（不是实验结果）：8条回答中一条答对，奖励1；一条错答过程分为1，奖励λ；其余六条为0。错答高于组均值的条件为λ&gt;1/7，因此λ=0.2仍可能给错答正优势。这是既定混合目标的行为，不是重复归一化的实现错误。</p>'
body+='<h2>6. 验收、限制和复现</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in c['limitations'])+'</ul>'
body+=f'<p>CPU独立验答重算 <strong>{c["acceptance"]["regraded"]:,}</strong> 条全部一致；本地取回后再次核对8份预测hash、题目ID、token计数和正确数。受保护旧文件 <strong>{c["acceptance"]["protected_files_unchanged"]}</strong> 个均未变化。新优化器更新0、PRM调用0。</p>'
body+='<p><a href="comparison.json">完整比较JSON</a> · <a href="accuracy.csv">准确率与输出统计CSV</a> · <a href="retrieval_acceptance.json">本地取回验收</a> · <a href="evidence/runs/Final_four_eval_seed17_1M_20261029/acceptance.json">CPU验收</a> · <a href="evidence/runs/Final_four_eval_seed17_1M_20261029/contract.json">固定配置和模型hash</a> · <a href="evidence/experiments/final_four_eval_20261029/">本次评测源代码</a> · <a href="CONTINUATION.txt">执行与恢复范围</a></p>'
body+='<p class="meta">模型：Qwen/Qwen2.5-Math-1.5B，revision '+esc(contract['model_revision'])+'。环境：'+esc(contract['versions'])+'。</p>'
body+='<p>基础实现依据：<a href="https://github.com/huggingface/trl/blob/4529a1c8b1813480a85b02c9c8e7f75a29085d65/trl/trainer/grpo_trainer.py">固定版本TRL GRPO源码</a>；相关门控／解耦优势见<a href="https://github.com/tanzelin430/PAPO/blob/7d260f0a7a6e23d289e32bbc96f80408dddfa1d5/verl/verl/trainer/ppo/core_algos.py#L333-L481">PAPO官方核心函数</a>。本项目不主张这些设计首次提出。本轮两个过程奖励臂都使用相同冻结C，不回答C适配是否优于原始公开PRM。</p>'
body+='<p><a href="evidence/data/prepared/manifest.json">数据版本与排重清单</a> · <a href="evidence/receipts/assets.json">基础模型来源与文件hash</a> · <a href="README.txt">复现说明</a> · <a href="completed_evidence.tar.gz">完整证据与adapter压缩包</a>。推理使用vLLM原生LoRA请求，没有将增量合并进BF16基础权重。</p>'
(HERE/'review.html').write_text(page(body))
lines=['单卡小模型数学推理RL：四模型统一测试报告','训练：既有1M seed17 checkpoint；本轮训练0。','核心结论：'+headline+interpretation,'准确率：']
lines.extend(' | '.join(x) for x in rows)
lines.append('\n同题比较（pp）：');lines.extend(' | '.join(x) for x in paired_rows)
lines.append('\n输出诊断：');lines.extend(' | '.join(map(str,x)) for x in diagnostic)
lines.append('\n成本：');lines.extend(' | '.join(x) for x in costrows)
lines.append(f'评测生成token合计：{cost["evaluation_generated_tokens"]:,}；完整评测与CPU验收墙钟：{full_wall/60:.1f}分钟。')
if (HERE/'qualitative_review.json').exists():
 lines.append('\n选定实际回答的助手审读（非人工金标准，不估计总体比例）：')
 for case in load(HERE/'qualitative_review.json').get('cases',[]):lines.extend([case['problem_id']+'：'+case['finding'],case.get('scope',''),case.get('evidence',''),case.get('explanation','')])
lines.append('\n限制：');lines.extend('- '+x for x in c['limitations'])
(HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
(HERE/'MANIFEST.json').write_text(json.dumps({'sha256':manifest,'complete':True,'training_started':False},indent=2)+'\n')
print(json.dumps({'report':str(HERE/'review.html'),'complete':True,'models':4,'datasets':2}))
