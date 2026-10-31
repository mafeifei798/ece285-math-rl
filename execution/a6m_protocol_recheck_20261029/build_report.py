"""Local completed-result report. Never launches GPU/CPU model inference."""
from pathlib import Path
import json,html,csv,os,hashlib
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def table(head,rows):
 return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
style='body{margin:0;background:#f1f5f8;color:#1b2d3d;font:16px/1.85 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1120px;margin:28px auto;padding:32px 40px;background:white;border-radius:14px}h1{font-size:28px;line-height:1.5}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #d9e3ec;text-align:left;vertical-align:top}th{background:#edf3f7}.scroll{overflow:auto}.note{padding:16px 20px;background:#edf6fb;border-left:4px solid #327da3}small{color:#607384}a{color:#1c638d}img{width:100%}pre{white-space:pre-wrap;background:#f2f5f8;padding:16px;overflow-wrap:anywhere} @media(max-width:700px){main{margin:0;padding:20px 16px}}'
def page(s):return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>旧6M A统一重评</title><style>'+style+'</style><main>'+s+'</main></html>'
if not (HERE/'comparison.json').exists():
 s=read(HERE/'latest_snapshot.json') if (HERE/'latest_snapshot.json').exists() else {};d=s.get('new_attempt',{});p=d.get('progress') or {}
 body='<small>ECE285 · 2026-10-29 PT · 仅评测</small><h1>旧6M A统一协议重评进行中</h1><div class="note">只生成旧checkpoint-284的MATH500/GSM8K回答，共1,819条。当前1M A和初始模型复用已验收结果。原始结果保留，不启动训练或PRM。</div>'
 body+=f'<p>已完成{p.get("done",0):,}/1,819；阶段：{html.escape(str((d.get("controller") or {}).get("stage","准备中")))}。</p>'
 body+='<p><a href="CONTINUATION.txt">执行范围</a> · <a href="dispatch.json">唯一派发记录</a></p>'
 (HERE/'review.html').write_text(page(body));print('Running report updated');raise SystemExit
c=read(HERE/'comparison.json');assert c['passed'] and read(HERE/'retrieval_acceptance.json')['passed']
run=HERE/'evidence/runs/A6M_protocol_recheck_20261029';contract=read(run/'contract.json')
prior_contract=read(HERE/'evidence/runs/Final_four_eval_seed17_1M_20261029/contract.json')
labels={'initial':'初始模型','A_1M':'当前A，约1M','A_6M_historical':'旧A，原6M评测','A_6M_recheck':'旧A，本轮统一重评'}
rows=[];csvrows=[]
for a,label in labels.items():
 state=({'training_generated_tokens':0,'training_updates':0} if a=='initial' else prior_contract['states']['A'] if a=='A_1M' else contract['states']['A_6M'])
 values=c['summary'][a];rows.append([label,f'{state["training_generated_tokens"]:,}',str(state['training_updates'])]+[f'{int(values[d]["correct"])}/{values[d]["n"]}（{100*values[d]["accuracy"]:.2f}%）' for d in ['math500','gsm8k']])
 for d,v in values.items():csvrows.append([a,d,v['n'],v['correct'],v['accuracy'],v['generated_tokens'],v['truncated'],v['empty'],v['short_wrong'],v['no_extractable_answer']])
with (HERE/'accuracy.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['state','dataset','n','correct','accuracy','generated_tokens','truncated','empty','short_wrong','no_extractable_answer']);w.writerows(csvrows)
pairrows=[]
for d,contrasts in c['paired'].items():
 for label,v in contrasts.items():pairrows.append([d,label,f'{v["delta_pp"]:+.2f}',f'[{v["paired_question_bootstrap95_pp"][0]:+.2f}, {v["paired_question_bootstrap95_pp"][1]:+.2f}]',f'{v["wins"]}/{v["losses"]}',f'{v["exact_mcnemar_two_sided_p"]:.4g}'])
reprows=[[d,f'{v["same_text"]}/{v["n"]}',f'{v["same_text_and_token_count"]}/{v["n"]}',f'{v["same_reward"]}/{v["n"]}','是' if v['same_correct_count'] else '否'] for d,v in c['reproduction'].items()]
reproduced=all(v['same_correct_count'] for v in c['reproduction'].values())
conclusion='原54.0%／67.02%的正确数在本轮统一协议下均精确复现。' if reproduced else '至少一项原正确数没有在本轮统一协议下精确复现；具体差值和同题变化见下表。'
delta='；'.join(f'{d.upper()}相对当前1M A为{v["A_6M_recheck minus A_1M"]["delta_pp"]:+.2f}pp，配对95%区间[{v["A_6M_recheck minus A_1M"]["paired_question_bootstrap95_pp"][0]:+.2f}, {v["A_6M_recheck minus A_1M"]["paired_question_bootstrap95_pp"][1]:+.2f}]pp' for d,v in c['paired'].items())+'。'
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,d in zip(axs,['math500','gsm8k']):
 vals=[c['summary'][a][d]['accuracy']*100 for a in labels]
 bars=ax.bar(['Initial','A 1M','A 6M old','A 6M recheck'],vals,color=['#8998a6','#3975a8','#d49b44','#318977'],width=.65)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.5,f'{v:.2f}%',ha='center',fontsize=9)
 ax.set_ylim(0,min(100,max(vals)+12));ax.set_title(d.upper());ax.set_ylabel('Greedy accuracy (%)');ax.tick_params(axis='x',rotation=12)
fig.suptitle('One historical checkpoint re-evaluated; no new training');fig.tight_layout();fig.savefig(HERE/'accuracy_comparison.png');plt.close(fig)
body='<small>ECE285 · 旧6M checkpoint独立重评 · 原始结果保留</small><h1>旧6M A能否复现？与当前1M A的统一比较</h1><div class="note">'+conclusion+'</div>'
body+='<p>'+delta+'</p><p>只新增旧A checkpoint-284的1,819条回答；初始模型和当前1M A复用上一轮已验收结果。本轮优化器更新0、PRM调用0。</p>'
body+='<h2>1. 准确率与复现结果</h2>'+table(['模型状态','训练生成token','参数更新次数','MATH500','GSM8K'],rows)+'<img src="accuracy_comparison.png" alt="旧6M A复现与1M A准确率">'
body+=table(['数据集','正文逐字相同','正文及token数量相同','验答结果相同','正确总数相同'],reprows)
body+='<p>“正文相同”是字符串比较；旧记录没有保存原始token ID，因此不能声称旧、新生成token逐ID一致。即使正确数相同，也需看同题胜负和正文一致率。生成的代码只作为文本，不执行。</p>'
runtime_note=''
if (HERE/'runtime_audit.json').exists():
 runtime=read(HERE/'runtime_audit.json')
 runtime_note='旧历史评测实际使用vLLM V0，本轮1M A与6M重评实际都使用V1；缓存、分块预填充、异步输出及内存配置也有差别。未做单因素切换，不能量化每项对分数的因果贡献。'
 body+='<h2>2. 为什么没有逐题精确复现</h2><p><strong>'+runtime_note+'</strong></p>'
 body+=table(['记录','实际引擎','显存利用率参数','前缀缓存','分块prefill','异步输出'],[[label,v['engine'],v['gpu_memory_utilization'],v['prefix_cache'],v['chunked_prefill'],v['async_output_processing']] for label,v in runtime['runs'].items()])
 body+='<p>三次使用相同vLLM 0.10.2、torch 2.8.0及同一base；历史和本次6M的LoRA权重hash相同。相同正文的验答／status不一致数均为0，全部7,276条冻结验答重算也一致，所以已观察到的得分变化来自生成内容，不能归咎于换了验答规则。本轮不再追加V0重跑或更改推理设置。</p>'
 body+='<p><a href="runtime_audit.json">运行路径审计</a> · <a href="runtime_log_excerpts/historical_6M.txt">旧6M V0日志</a> · <a href="runtime_log_excerpts/current_1M.txt">当前1M V1日志</a> · <a href="runtime_log_excerpts/current_6M.txt">本次6M V1日志</a>。vLLM该版本的<a href="https://docs.vllm.ai/en/v0.10.2/usage/reproducibility.html">官方可复现性说明</a>要求额外控制运行方式；greedy与固定seed本身不应被当作跨引擎逐位一致的保证。</p>'
body+='<h2>3. 同题比较与不确定性</h2>'+table(['数据集','比较','差值pp','配对95%区间pp','胜／负','精确McNemar p'],pairrows)
body+='<p>每项10,000次题目配对bootstrap，seed285；p值未作多重比较校正。区间反映题目抽样，不能代表训练seed方差。</p>'
body+='<h2>4. 对训练量问题意味着什么</h2><p>6M A与1M A具有相同base、初始LoRA、训练数据及主要训练参数，但来自不同训练轨迹：旧A的训练vLLM显存参数为0.3，当前为0.15，重启历史也不同。它们不能称为同一次不间断训练的两个checkpoint，差异不能全部归因额外5M token。</p>'
body+='<p>若较长训练的收益得到支持，说明100万pilot不足以判断训练潜力；本轮仍没有6M C-mix/C-bounded结果，因此不能证明过程奖励随规模改善或失效。本次只回答历史A的可复现性和描述性规模参照。</p>'
body+='<h2>5. 输出健康与资源</h2>'+table(['状态','数据集','空答','短错答≤32token','截断','无可提取答案','平均生成token'],[[labels[a],d,v['empty'],v['short_wrong'],v['truncated'],v['no_extractable_answer'],f'{v["mean_generated_tokens"]:.1f}'] for a,ds in c['summary'].items() for d,v in ds.items()])
cost=c['cost'];wall=read(run/'controller.json')['unix']-read(run/'dispatch.json')['unix']
body+=f'<p>本轮新生成{cost["generated_tokens"]:,} token；生成进程{cost["generation_seconds"]/60:.1f}分钟，派发至CPU验收完成{wall/60:.1f}分钟。本任务每15秒采样的进程树显存峰值约{cost["sampled_own_GPU_peak_MiB"]/1024:.2f}GiB；可能漏掉瞬时峰值。GPU1共享，耗时不代表独占卡速度。</p>'
if (HERE/'qualitative_review.json').exists():
 body+='<h2>6. 选定实际回答审读</h2><p>以下为助手的定向审读，不是人工金标准，不估计总体比例，不改变正式评分。</p>'
 for case in read(HERE/'qualitative_review.json')['cases']:body+='<p><strong>'+html.escape(case['problem_id'])+'</strong>：'+html.escape(case['finding'])+'</p><pre>'+html.escape(case['evidence'])+'</pre><p>'+html.escape(case['explanation'])+'</p>'
body+='<h2>7. 验收与复现证据</h2><p>CPU重算新旧共7,276条验答全部一致；本地再次核对8份预测文件hash、新生成1,819条原始token长度和旧checkpoint权重hash。受保护旧文件'+str(c['acceptance']['protected_files_unchanged'])+'个均未变化。</p>'
body+='<p>生成协议与上一轮一致：BF16、vLLM原生LoRA、adapter ID1、greedy、seed285、2048生成上限、4096上下文、batch64、相同prepared prompts与reward-repair-v3。单模型重评不执行四模型交替调度，这项调度差异单独披露，不假设逐位确定。</p>'
body+='<ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in c['limitations'])+'</ul>'
body+='<p><a href="comparison.json">完整比较JSON</a> · <a href="accuracy.csv">结果CSV</a> · <a href="retrieval_acceptance.json">本地取回验收</a> · <a href="evidence/runs/A6M_protocol_recheck_20261029/acceptance.json">CPU验收</a> · <a href="evidence/runs/A6M_protocol_recheck_20261029/contract.json">冻结协议与hash</a> · <a href="evidence/runs/A6M_protocol_recheck_20261029/case_review_pack.json">同题真实回答</a> · <a href="completed_evidence.tar.gz">完整证据包</a> · <a href="CONTINUATION.txt">执行范围</a>。</p>'
(HERE/'review.html').write_text(page(body))
lines=['旧6M A统一协议重评：已完成',conclusion,delta,'\n模型 | 训练生成token | 参数更新次数 | MATH500 | GSM8K：']+[' | '.join(r) for r in rows]
lines+=['\n复现正文与验答一致性：']+[' | '.join(map(str,r)) for r in reprows]
lines+=['\n同题比较：']+[' | '.join(map(str,r)) for r in pairrows]
if runtime_note:lines+=['\n运行路径审计：'+runtime_note,'相同正文的评分与status不一致数均为0。未做单因素重生成，不把差异归因某一项配置。']
lines += [f'\n新生成token：{cost["generated_tokens"]:,}；派发至验收：{wall/60:.1f}分钟；采样显存峰值：{cost["sampled_own_GPU_peak_MiB"]/1024:.2f}GiB。','\n限制：']+c['limitations']
(HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
(HERE/'MANIFEST.json').write_text(json.dumps({'complete':True,'new_training':False,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.name!='MANIFEST.json'}},indent=2)+'\n')
print(json.dumps({'complete':True,'report':str(HERE/'review.html'),'accuracy_reproduced':reproduced}))
