"""Build a self-contained 50-answer, 469-step audit with no external scripts."""
import collections,copy,hashlib,html,json,math,re
from pathlib import Path
P=Path(__file__).resolve().parent
read=lambda name:json.loads((P/name).read_text())
esc=lambda value:html.escape(str(value),quote=True)
full=read('selected_full.json');review=read('final_review.json')
rv={r['review_id']:r for r in review['records']}
align=read('alignment.json');al={r['review_id']:r for r in align['results']}
av={r['review_id']:r for r in read('sample_advantages.json')}
replay=read('forward_replay.json');selection=read('selection.json')
assert align['all_checks_passed'] and replay['complete']
assert len(full)==len(rv)==len(al)==len(av)==50
assert hashlib.sha256((P/'initial_review.json').read_bytes()).hexdigest()==review['initial_review_sha256']
names={'C':'正确/局部成立','E':'含明确错误','U':'无法确定/非数学上下文'}
chains={'clean_valid':'全程核对无错','contains_errors':'含明确错误','uncertain':'有无法确定内容','ill_posed':'题面与标准解矛盾'}
cats={'uniform_random':'随机抽样','answer_correct_min_le_005':'可疑：答对但最低分≤0.05','answer_wrong_min_ge_080':'可疑：答错但最低分≥0.80','single_native_segment_long_text':'可疑：长回答仅切成一段'}
stats={}
for kind in ['R','S']:
 rows=[r for r in full if r['review_id'].startswith(kind)]
 clean=[r for r in rows if rv[r['review_id']]['chain_verdict']=='clean_valid']
 errors=[r for r in rows if rv[r['review_id']]['chain_verdict']=='contains_errors']
 stats[kind]={'n':len(rows),'chains':dict(collections.Counter(rv[r['review_id']]['chain_verdict'] for r in rows)),
  'clean_valid_low_min_below_05':[r['review_id'] for r in clean if r['saved']['min_score']<.5],
  'error_containing_all_steps_ge_05':[r['review_id'] for r in errors if r['saved']['min_score']>=.5],
  'first_error_scored_ge_05':[r['review_id'] for r in errors if r['saved']['step_scores'][rv[r['review_id']]['first_math_error']-1]>=.5]}
stats['random_valid_answer_negative_advantages']=[r['review_id'] for r in full[:30] if rv[r['review_id']]['chain_verdict']=='clean_valid' and av[r['review_id']]['mixed_advantage']<0]
stats['CRLF_affected']=[{'id':r['review_id'],'native_steps':r['steps'],'normalized_steps':r['steps_if_CRLF_normalized']} for r in align['results'] if r['steps']!=r['steps_if_CRLF_normalized']]
summary={'scope':'Fixed 30 random +20 targeted B-6M answers, 469 native segments; no new training',
 'reviewer':'Codex step-by-step audit with independent arithmetic/CAS verification; not independent multi-human labels',
 'threshold':.5,'random_seed':selection['seed'],'population':selection['eligible_complete_parseable_rows'],'stats':stats,
 'alignment_passed':True,'forward_replayed_records':5,'max_replayed_min_difference':max(r['min_score_difference'] for r in replay['records']),
 'max_prefix_vs_full_difference':max(r['causal_prefix_vs_full_difference'] for r in replay['records']),
 'outcome_verifier_anomaly':'R04: mathematically correct 20% versus gold20 gives reward0; independent replay confirms parse20% as1/5',
 'input_anomaly':'R18: question largest has no maximum; reference solution instead minimizes and yields592',
 'limitations':['30-answer random sample only; do not estimate global rate from targeted20 or pool all50',
 'A score below0.5 is a fixed diagnostic classification; it is not by itself proof of negative RL advantage or poor global ranking',
 'Locally correct steps after previous errors are not counted as unambiguous false penalties',
 'Corrected initial guesses and unrepaired mathematical errors are reported separately',
 'No causal attribution of the full A/B accuracy gap to a single observed failure mode',
 'Two random cases retain uncertain text and one has an ill-posed question; do not force labels'],
 'initial_review_sha256':review['initial_review_sha256']}
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['B真实回答逐步审阅：30条随机 +20条可疑','',
'结论：确实观察到正确数学步骤被低分、错误推导被高分；这些分数有时进入了相反方向的GRPO优势。50条中未发现系统性评分位置错位，双换行切分粒度和CRLF处理却有明确问题。另外发现1条结果奖励单位口径问题、1条题面/标准解矛盾。',
'本轮没有新训练、参数更新、奖励规则修改或B恢复操作。只做静态日志审阅、CPU分词/代数复核及5条固定回答的CPU PRM推理。','',
'样本与审阅口径',
'从B最初6M的9120条回答中，选择7472条完整、可判分且实际被PRM评分的回答为抽样框；其余550截断、985缺框、113不可解析，不把这些占位零当作PRM判错。',
'随机种子20261021，无放回均匀抽30条，不按长度或分数筛选；另外8条答对且min≤0.05、8条答错且min≥0.8、4条长回答单段。50条恰为50个不同题目。20条可疑样本不能用于计算总体误判率。',
'先隐藏PRM分数、gold、结果奖励和标准解，逐步写下469段的判断，保存initial_review.json后再揭示分数并做独立计算；最终修订及理由保存在final_review.json。审阅者是Codex，不冒充独立人工金标准。',
'C表示局部数学/程序步骤成立；E表示段内有明确数学错误；U表示不明、未定或运行上下文不可确认。此前已有错误后的局部正确步骤不直接算PRM误罚；早期错答后重新正确推导也单列。','',
'随机30条的描述性结果',
'7条全程核对无明确错误，其中6条min<0.5；这些数字来自很小的随机样本，不外推成总体误判率。',
'20条含明确错误，其中3条整个回答的所有原生步骤分数都≥0.5（R01、R07、R14）；其中包括中间函数代入、几何推理和虚假的计算输出。',
'2条保留无法确定的论坛式文本（R09、R11），1条题面本身不成立（R18），均单独报告。',
'答对但含早期错误的随机样本并非全都“碰巧答对”：R03/R21/R22/R28/R29随后出现正确推导，R27则存在错误几何论证后报出正确终答。不能只看结果奖励区分这些情况。','',
'最直接的训练信号反例',
'R02：gcd(5!,6!)=120，全部步骤成立；第8段明确说GCF是120，却仅得0.066448。该原始八回答组全答对，这条回答的混合奖励优势=-0.3355，去掉PRM后在同一组的纯结果优势=0。',
'R01：乘积正确值3/161840，却声称Pascal三角形及Python输出给1/4；整条最低分0.542863，全错组中优势=+2.1468。',
'S11：把等角共轭线当作角平分线，最后又说7.222四舍五入到9；最低分0.926145，全错组中优势=+2.3222。此例来自针对性筛选，不代表常见比例。',
'这些是同一保存B组的优势重算，不是A策略实际生成的对照结果，也不能单凭这些案例确定A/B准确率差的因果来源。低过程分不必然产生负优势；例如R10和S08仍获得正优势。','',
'其余两个问题',
'R04：问有多少百分比，20个/100个=20%；数学正确。但gold是20，math-verify把20%读作20/100=1/5，结果奖励为0，原组优势=-0.9229。独立复算20%→0、20→1确认。A/B共享这个验证器，所以尚不能归因于B独有实现。',
'R18：题目要求仅含0、8且为15倍数的最大正整数，却无位数限制；任何合条件数乘10仍合条件，所以无最大值。标准解求的是最小8880，gold592。该条排除出PRM误判统计，不改原始数据。','',
'分段与评分对齐',
'50/50的步骤列表、469个分数及token前缀终点全部匹配；评分点恰落在各原生前缀末尾，包含标点与双换行合并成同一token的情况，这是原生编码而非错位。奖励公式及min聚合一致。',
'5条固定回答CPU FP32重新前向，并在最低分位置独立重跑该前缀，前缀与完整序列分数最大差约2.24e-8。对比原GPU BF16最低分最大差0.0225，五例高低判断均相同；这不是位级完全复现。',
'单段常混合多个数学步骤；例如S20在一段中包含除2、配方、半径、面积，多处错误无法靠一个分数定位。S19则是数学正确的多步反函数推导，但也只得到一个分数。',
'R20/R23/S03/S08含CRLF空行。仅用于诊断地规范化换行后，段数分别5→9、19→20、3→5、19→21。没有把这一改法用于现有训练、重算奖励或改四组设置。',
'代码中的空行也会被当成步骤边界，容易把import、注释或代码中间位置当成一个“解题步骤”。分数索引对齐正确，不代表每段都是独立数学步骤。','',
'判断与限制',
'目前有证据说明B的奖励信号存在多种噪声，且其中部分会放大成不合数学正确性的优势；没有发现足以解释一切的统一索引错位。审阅不能证明整个A/B差距完全由这些现象导致，也没有验证任何新修复一定有效。',
'先保留原始日志、抽样和这份审阅；任何后续修复都应另立版本并在相关实验臂共享，不能只替换C/D使主对照混入另一种奖励算法。','',
'逐条结果（完整原文、逐步分数与备注见review.html和reviewed_records.json）',
'ID | 类型 | 原始行 | 段数 | 审阅结论 | min分数 | 结果奖励 | 混合优势 | 首个明确错误']
output=[]
for r in full:
 k=r['review_id'];v=rv[k];s=r['saved'];a=al[k];g=av[k]
 assert len(v['step_labels'])==len(r['steps'])==len(a['prefix_positions'])
 rows=[];prefix='valid'
 for i,(text,label,score,pos) in enumerate(zip(r['steps'],v['step_labels'],s['step_scores'],a['prefix_positions']),1):
  before=prefix
  if label=='E':prefix='invalid'
  elif label=='U' and prefix=='valid':prefix='uncertain'
  rows.append({'step':i,'text':text,'judgment':label,'prefix_status':prefix,'PRM_score':score,'score_token_position_0based':pos,
    'note':v['step_notes'].get(str(i),'逐段核对为局部成立；完整数学依据见本条的独立核算。' if label=='C' else ('此段含明确错误，具体位置见本条独立核算。' if label=='E' else '语义/运行上下文无法确定，不强行判为数学正确或错误。')),
    'score_vs_judgment_flag':'high_at_error' if label=='E' and score>=.5 else ('low_on_valid_prefix' if label=='C' and prefix=='valid' and score<.5 else None)})
 case={**r,'review':v,'alignment':a,'advantage':g,'step_review':rows}
 output.append(case)
 lines.append(f"{k} | {cats[r['sampling_category']]} | {r['source_row']} | {len(rows)} | {chains[v['chain_verdict']]} | {s['min_score']:.6f} | {int(s['outcome_reward'])} | {g['mixed_advantage']:+.4f} | {v['first_math_error'] or '-'}")
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
(P/'reviewed_records.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')

css='''body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;margin:0;background:#f3f5f8;color:#1a2638;line-height:1.65}main{max-width:1220px;margin:30px auto;padding:0 22px}h1{font-size:29px;line-height:1.3;margin:0 0 12px}h2{font-size:19px}a{color:#2457c5}p{margin:9px 0}.lead{background:#fff;border-left:5px solid #3563cc;padding:20px 24px;border-radius:10px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.metric{background:white;padding:15px;border-radius:10px;border:1px solid #e0e5ec}.metric b{display:block;font-size:24px;color:#254da7}.muted{color:#617087;font-size:13px}.filters{position:sticky;top:0;background:#f3f5f8ee;backdrop-filter:blur(8px);padding:12px 0;display:flex;gap:10px;flex-wrap:wrap;z-index:3}input,select,button{font:inherit;font-size:14px;padding:8px 11px;border:1px solid #ccd5e2;border-radius:7px;background:white}input[type=search]{min-width:270px}button{cursor:pointer}.case{background:white;border:1px solid #dce2eb;border-radius:10px;margin:14px 0;overflow:hidden}.case>summary{cursor:pointer;padding:15px 18px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-weight:600}.case>summary:hover{background:#f8faff}.tag{font-size:12px;background:#edf1f8;border-radius:20px;padding:2px 9px;font-weight:500}.contains_errors{background:#fff0e9;color:#a23e11}.clean_valid{background:#e7f5ed;color:#17613c}.uncertain,.ill_posed{background:#fff5d6;color:#7a5c00}.body{padding:0 18px 20px;border-top:1px solid #e4e9ef}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px;line-height:1.62;background:#f6f8fb;padding:13px;border-radius:7px}.proof{border-left:3px solid #4d75cc;background:#f0f5ff;padding:12px 16px;margin:15px 0}.warning{border-left:3px solid #c4831a;background:#fff6e6;padding:12px 16px;margin:15px 0}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid #e5e9f0;vertical-align:top;padding:9px}th{background:#f1f4f9;white-space:nowrap}td pre{margin:0;background:transparent;padding:0}.n{width:38px}.p{width:74px;font-variant-numeric:tabular-nums}.judge{min-width:92px}.reason{min-width:180px;max-width:260px;color:#526176}.mathtext{min-width:320px}.high_at_error{background:#fff1ec}.low_on_valid_prefix{background:#eef4ff}.scoreoff .score,.scoreoff .gold{visibility:hidden}.reviewoff .judgment,.reviewoff .sensitive{visibility:hidden}.reviewoff .high_at_error,.reviewoff .low_on_valid_prefix{background:transparent}.source{font-size:12px;overflow-wrap:anywhere}.footer{margin:25px 0 60px;font-size:13px;color:#64748b}@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:0 12px}.mathtext{min-width:270px}.reason{min-width:150px}}'''
parts=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B 真实回答逐步审阅 · 30+20</title><style>'+css+'</style></head><body><main>',
'<h1>B 真实回答：30 条随机 + 20 条可疑</h1><p class="muted">固定 seed 20261021 · B 前 6M 日志 · 469 个原生段 · 只读诊断，无新训练</p>',
'<div class="lead"><b>核对结果：评分索引没有整体错位，但确实存在正确推导低分、错误推导高分。</b><p>部分异常已进入 GRPO 优势。另外确认了百分比验证口径问题，以及一条“最大/最小”题面矛盾。下方可以逐条展开查看原文、分数、审阅判断和证据。</p><p class="muted">单一 Codex 审阅者，配合独立代数/有理数计算；不是多人独立人工金标准。无法确定的内容保留 U。所有 50 条均保留。</p></div>',
'<div class="cards"><div class="metric"><b>6 / 7</b>随机样本中，全程核对无错但 min&lt;0.5 的回答<div class="muted">30 条中有 7 条全程无错，不外推总体比例</div></div><div class="metric"><b>3 / 20</b>随机样本中含错误、却所有段都≥0.5<div class="muted">R01 / R07 / R14</div></div><div class="metric"><b>469 / 469</b>原生分段与评分位置检查通过<div class="muted">5 条另做 CPU 前向复现</div></div><div class="metric"><b>2 类额外问题</b>百分比奖励口径、最大/最小题面冲突<div class="muted">R04 / R18</div></div></div>',
'<details><summary>抽样、盲审、阈值与统计范围</summary><div class="proof">从 7,472 条完整且可评分回答无放回均匀抽 30 条；另外选 8 条答对低分、8 条答错高分、4 条长单段。先保存不显示分数与 gold 的初审，再揭示分数及标准解复核。20 条针对性样本只用于找问题。<br>0.5 是固定诊断阈值；分数低不等于优势必为负。已有错误之后的局部正确步骤，不直接算 PRM 错罚。修正过的初始错误与一直未修正的错误分别记录。<br>本轮不能证明 A/B 准确率差完全由这些问题造成；未重跑 RL、改奖励、改切分、修改数据或恢复 B。</div></details>',
'<div class="filters"><input id="search" type="search" placeholder="搜索 ID、题目或错误关键词"><select id="set"><option value="all">全部 50 条</option><option value="R">随机 30 条</option><option value="S">可疑 20 条</option></select><select id="verdict"><option value="all">全部判断</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in chains.items())+'</select><button id="score">隐藏分数/gold</button><button id="review">隐藏审阅结论</button><button id="expand">展开可见条目</button><span id="count" class="muted"></span></div>']
for c in output:
 k=c['review_id'];s=c['saved'];v=c['review'];g=c['advantage'];a=c['alignment'];status=v['chain_verdict']
 search=k+' '+c['question']+' '+v['verification']+' '+' '.join(v['flags'])
 parts.append(f'<details class="case" id="{k}" data-set="{k[0]}" data-verdict="{status}" data-search="{esc(search)}"><summary><b>{k}</b><span class="tag">{"随机" if k[0]=="R" else "可疑"}</span><span class="tag sensitive {status}">{chains[status]}</span><span>{len(c["steps"])} 段</span><span class="score">min {s["min_score"]:.4f} · 优势 {g["mixed_advantage"]:+.4f}</span></summary><div class="body">')
 parts.append(f'<p class="muted sensitive">{cats[c["sampling_category"]]} · 原始行 {c["source_row"]} · {esc(s["problem_id"])} · optimizer step {s["optimizer_step_before_update"]}</p><h2>题目</h2><pre>{esc(c["question"])}</pre>')
 parts.append(f'<p class="gold">标准答案：<code>{esc(s["gold"])}</code>　验证器提取：<code>{esc(s["answer"])}</code>　结果奖励：{int(s["outcome_reward"])}</p>')
 parts.append(f'<p class="score">原 B 组：{esc(g["group_type"])}；混合优势 {g["mixed_advantage"]:+.6f}；同一组仅结果优势 {g["outcome_only_advantage_on_same_group"]:+.6f}。后者只是固定 B 组上的反事实重算，不是 A 策略训练结果。</p>')
 parts.append(f'<div class="proof judgment"><b>独立核算</b><br>{esc(v["verification"])}<br><b>首个明确数学错误：</b>{v["first_math_error"] or "无已确认错误"}；<b>后续出现正确推导/改正：</b>{"是" if v["later_correct_derivation"] else "未作此确认"}</div>')
 if v.get('post_reveal_note'):parts.append(f'<div class="warning judgment">揭示分数/标准解后的复核：{esc(v["post_reveal_note"])}</div>')
 parts.append('<div class="table-wrap"><table><thead><tr><th class="n">段</th><th class="mathtext">模型原文（按双换行切分）</th><th class="p score">PRM</th><th class="judge judgment">逐步判断</th><th class="reason judgment">依据 / 上下文</th></tr></thead><tbody>')
 for r in c['step_review']:
  prefix={'valid':'此前前缀确定有效','invalid':'前缀已含错误','uncertain':'前缀存在未定内容'}[r['prefix_status']]
  parts.append(f'<tr class="{r["score_vs_judgment_flag"] or ""}"><td>{r["step"]}</td><td><pre>{esc(r["text"])}</pre><div class="muted">评分 token 索引 {r["score_token_position_0based"]}（从 0 起）</div></td><td class="score">{r["PRM_score"]:.6f}</td><td class="judgment">{names[r["judgment"]]}<div class="muted">{prefix}</div></td><td class="judgment">{esc(r["note"])}</td></tr>')
 parts.append('</tbody></table></div>')
 parts.append(f'<p class="muted">原生对齐：通过。{a["input_tokens"]} 输入 token，{a["steps"]} 个评分位置；若仅规范化 CRLF，段数会为 {a["steps_if_CRLF_normalized"]}（此处未改分、未训练）。</p>')
 parts.append(f'<details><summary>查看原始完整回答 / 标准解 / 评分位置</summary><h3>未改写的 completion</h3><pre>{esc(s["completion"])}</pre><div class="gold"><h3>现有标准解，仅用于复核，可能自身有错误</h3><pre>{esc(c["standard_solution"])}</pre></div><pre>{esc(json.dumps(a["score_tokens"],ensure_ascii=False,indent=2))}</pre></details></div></details>')
parts.append('<p class="footer">证据文件：selection.json / initial_review.json / final_review.json / reviewed_records.json / alignment.json / independent_math.json / sample_advantages.json / forward_replay.json。所有原始字段和 50 条选择均固定；HTML 可离线打开，不调用外部脚本。</p></main>')
parts.append('''<script>const cases=[...document.querySelectorAll('details.case')];const q=document.getElementById('search'),set=document.getElementById('set'),verdict=document.getElementById('verdict');function apply(){let n=0;for(const c of cases){let show=(set.value==='all'||c.dataset.set===set.value)&&(verdict.value==='all'||c.dataset.verdict===verdict.value)&&c.dataset.search.toLowerCase().includes(q.value.toLowerCase());c.hidden=!show;if(show)n++}document.getElementById('count').textContent=n+' / 50 条'}[q,set,verdict].forEach(x=>x.addEventListener('input',apply));document.getElementById('score').onclick=function(){const hidden=document.body.classList.toggle('scoreoff');this.textContent=hidden?'显示分数/gold':'隐藏分数/gold'};document.getElementById('review').onclick=function(){const hidden=document.body.classList.toggle('reviewoff');this.textContent=hidden?'显示审阅结论':'隐藏审阅结论'};document.getElementById('expand').onclick=()=>cases.filter(x=>!x.hidden).forEach(x=>x.open=true);apply();</script></body></html>''')
(P/'review.html').write_text(''.join(parts))
print(json.dumps({'records':len(output),'steps':sum(len(x['step_review']) for x in output),'summary':summary['stats'],'html_bytes':(P/'review.html').stat().st_size},ensure_ascii=False))
