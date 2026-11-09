from pathlib import Path
import json,hashlib,html
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];E=HERE/'completed_evidence'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]
def main():
    checked=read(HERE/'completion_verification.json');assert checked['passed']
    c=read(E/'comparison.json');stats=c['statistics'];cpu=read(E/'admission/independent_acceptance.json')
    diagnostic=read(E/'admission/placeholder_diagnostic/result.json');cost={}
    for arm in ['A','C_bounded']:
        old=read(E/'admission'/f'source_accounting_{arm}.json')
        end=read(E/arm/'stage_576_result.json')['accounting'];rr=rows(E/arm/'resources.jsonl')
        cost[arm]={'additional_updates':288,'new_completion_tokens':end['generated_completion_tokens']-old['generated_completion_tokens'],
            'cumulative_completion_tokens':end['generated_completion_tokens'],
            'new_PRM_input_tokens':end['prm_input_tokens']-old['prm_input_tokens'],
            'new_PRM_seconds':end['prm_forward_seconds']-old['prm_forward_seconds'],
            'training_seconds':sum(read(E/arm/f'stage_{k}_result.json')['seconds'] for k in [432,576]),
            'sampled_GPU_peak_GiB':max(r.get('own_MiB',0) for r in rr)/1024}
        assert cost[arm]['new_completion_tokens']==cpu['arms'][arm]['new_generated_tokens']
    p=c['paired_comparisons']['endpoints_math_C_minus_A_at576'];lo,hi=p['paired_question_bootstrap95CI_pp']
    independent_ci=checked['primary_endpoint_comparisons']['endpoints_math_C_minus_A_at576']['independent_paired_question_bootstrap95CI_pp']
    if min(lo,independent_ci[0])>0:judgment='本轮12M共同评测支持C-bounded优于A；结论限于seed17，仍需独立seed确认。'
    elif max(hi,independent_ci[1])<0:judgment='本轮12M共同评测中C-bounded较低；不据此扩大过程奖励方案。'
    else:judgment='本轮12M未确认C-bounded优于A；报告点估计和区间，不把小差距写成已证实的提升。'
    lines=['A/C-bounded：6M→12M固定预算续训结果',judgment,
        'seed17；各自原始checkpoint288→432→576。题序与更新数一致，生成token实数另报。学习率5e-6 constant、num_iterations1、beta.04、原GRPO loss、原验答规则、冻结C-PRM均保持。',
        '沿用已验收原生LoRA V1/eager同步，标量loss AST与原版一致。源checkpoint各状态恢复，vLLM采样RNG重建；不声称位级等同于历史不中断训练。','',
        '验证集：固定64题批次，独立进程复测两次；平均时保留500个题目单位。']
    validation_rows=[]
    for cohort,label in [('baseline','6M'),('nine','9M'),('final','12M')]:
        v=stats[cohort]
        for arm in ['A','C_bounded']:
            s=v[arm];flips=s['same_checkpoint_repeat_binary_flips']
            lines.append(f"{label}/{arm}: 两次{[round(x*100,2) for x in s['per_repeat_accuracy']]}%；平均{s['mean_accuracy']*100:.2f}%；同题结果翻转{flips}/500。")
            validation_rows.append((label,arm,s['mean_accuracy']*100,s['per_repeat_accuracy'],flips))
    lines+=['','共同端点评测：6M原checkpoint和12M新checkpoint同一批协议重评；旧测试结果保留，以下是本轮共同结果。']
    test_rows=[]
    for cohort,label in [('endpoints_math','MATH500，两次'),('endpoints_gsm','GSM8K，一次')]:
        for step in [288,576]:
            for arm in ['A','C_bounded']:
                s=stats[cohort][f'{arm}_{step}'];repeat=[round(v*100,2) for v in s['per_repeat_accuracy']]
                details=f"两次{repeat}%，同题翻转{s['same_checkpoint_repeat_binary_flips']}/{s['n_questions']}" if s['repeats']==2 else '单次，不测运行间复现'
                test_rows.append((label,6 if step==288 else 12,arm,s['mean_accuracy']*100,details))
                lines.append(f"{label}/{arm}/{6 if step==288 else 12}M：{s['mean_accuracy']*100:.2f}%；{details}。")
    lines+=['','配对差异（百分点，题目bootstrap95%区间）：']
    for k,v in c['paired_comparisons'].items():
        l,h=v['paired_question_bootstrap95CI_pp'];lines.append(f"{k}: {v['difference_pp']:+.2f} [{l:+.2f},{h:+.2f}]。")
    lines.append('独立标准库Monte Carlo区间见completion_verification.json；靠近0的离散边界若重算改变符号，不解释为稳健显著。MATH500的12M C−A为预设主端点，其余为未校正的探索性比较。')
    lines+=['','格式监测（不作为新增奖励）：']
    format_data={}
    for cohort in ['baseline','nine','final']:
        format_data[cohort]={}
        for arm in ['A','C_bounded']:
            sums=[read(E/'evaluation'/cohort/f'repeat_{i}/summary.json')['summary'][arm] for i in [0,1]]
            placeholder=sum(x['format_counts']['leading_placeholder_box_with_body']/x['n'] for x in sums)/2
            missing=sum(x['status_counts'].get('missing_or_invalid_box',0)/x['n'] for x in sums)/2
            trunc=sum(x['truncated']/x['n'] for x in sums)/2
            eligible=sum(sum(x['status_counts'].get(s,0) for s in ['exact','symbolic','unit_exact','unit_symbolic']) for x in sums)
            correct=sum(x['correct'] for x in sums);total=sum(x['n'] for x in sums)
            format_data[cohort][arm]={'placeholder_with_body':placeholder,'missing_or_invalid_box':missing,'truncated':trunc,
                'protocol_eligible_rate':eligible/total,'conditional_answer_accuracy_among_eligible':correct/eligible if eligible else None}
            lines.append(f"{cohort}/{arm}：占位首框+正文{placeholder*100:.2f}%，无效/缺框{missing*100:.2f}%，截断{trunc*100:.2f}%。")
            lines.append(f"输出协议合格率{eligible/total*100:.2f}%；其中终答正确率{correct/eligible*100:.2f}%。两者乘积等于总准确率；子集难度/长度有选择偏差，不能直接称为数学能力。" if eligible else '本组没有协议合格输出。')
    lines+=['首框前至多20个非空白字符、框后至少100个非空白字符；占位包括空、?、.等。它是输出格式指标，不证明是否事后编理由；非占位也不保证数学正确。',
        '',f"独立冻结PRM诊断：200条答对且占位框开头的回答，原文−删除占位框的最低分均差{diagnostic['mean_min_original_minus_deleted']:+.6f}，区间{diagnostic['paired_mean_bootstrap95CI']}。",
        f"20个针对性真实组中{diagnostic['groups_with_correct_process_rank_change']}组正确子集排序变化，{diagnostic['advantage_sign_changes']}个优势符号变化。仅说明此编辑的打分敏感性，不能证明PRM诱发或不可能诱发格式偏移；分段变化与选择条件需保留。诊断没有改变训练。",
        '旧C6M GSM8K的123条缺框输出中，122条以占位框开头，全部123条无非占位完整框；A为83条且83条同结构。不能把它们都计为可挽回的正确数学答案。','',
        '实际新增资源：']
    for arm,v in cost.items():lines.append(f"{arm}：新增{v['new_completion_tokens']:,}token，累计{v['cumulative_completion_tokens']:,}；训练{v['training_seconds']/3600:.2f}小时；PRM前向{v['new_PRM_seconds']/60:.2f}分钟；采样显存峰值{v['sampled_GPU_peak_GiB']:.2f}GiB。")
    evaluation_seconds=sum(read(p)['seconds'] for p in (E/'evaluation').glob('*/repeat_*/summary.json'))
    lines.append(f'统一评测墙钟合计{evaluation_seconds/3600:.2f}小时；共享卡耗时不等于孤立吞吐或FLOPs。显存为采样峰值。')
    gate=c['seed42_eligibility'];lines+=['',f"预设seed42跟进资格：{'满足' if gate['eligible'] else '未满足'}；检查项{gate['checks']}。资格在测试前只用验证集冻结，是资源分配规则，不是显著性检验；本轮未自动启动seed42。",
        '',f"验收：独立CPU重算{checked['new_training_batches']}批/{checked['new_training_responses']}条训练回答；原验答器复核15276条预测；原checkpoint与{cpu['protected_files_verified']}项历史文件哈希不变。优势未再次归一化，错答无正优势；采样校正关闭，无cached old log-prob，不宣称重建了采样比率。",
        '限制：只有一个续训seed；测试和验证已有历史使用。两次评测只量到本次运行间波动，不证明完全确定，也不能把1000条当成1000道独立题；题目bootstrap不包含训练seed与完整运行数值方差。6M前A/C重启历史不对称的旧限制保留，本次共同起点与续训阶段已固定。准确率没有独立验证每条推导的有效性；C-PRM原有训练数据/锚点限制保留。']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    summary={'judgment':judgment,'validation':{k:stats[k] for k in ['baseline','nine','final']},'endpoint_statistics':{k:stats[k] for k in ['endpoints_math','endpoints_gsm']},
        'paired_comparisons':c['paired_comparisons'],'format_metrics':format_data,'costs':cost,'evaluation_seconds':evaluation_seconds,'seed42_eligibility':gate,'independent_acceptance_passed':True}
    (HERE/'result_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A/C 6M→12M 结果</title><style>body{font:16px/1.85 -apple-system,"PingFang SC",sans-serif;background:#f4f6fa;color:#21344c}main{max-width:1100px;margin:30px auto;padding:35px;background:white;border-radius:12px}.callout{background:#edf5ff;padding:18px;border-left:4px solid #4675b3}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #dde4ed;text-align:left}th{background:#f4f7fb}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:#f7f8fa;padding:18px}a{color:#2866a4}</style><main><h1>A/C-bounded · 6M→12M 固定预算结果</h1><div class="callout">'+html.escape(judgment)+'</div><h2>验证曲线</h2><table><tr><th>预算点</th><th>模型</th><th>平均准确率</th><th>两次准确率</th><th>同题翻转</th></tr>'
    for label,arm,mean,repeat,flips in validation_rows:page+=f'<tr><td>{label}</td><td>{arm}</td><td>{mean:.2f}%</td><td>{repeat[0]*100:.2f}% / {repeat[1]*100:.2f}%</td><td>{flips}/500</td></tr>'
    page+='</table><h2>共同端点评测</h2><table><tr><th>评测</th><th>预算</th><th>模型</th><th>准确率</th><th>复测情况</th></tr>'
    for label,budget,arm,accuracy,details in test_rows:page+=f'<tr><td>{label}</td><td>{budget}M</td><td>{arm}</td><td>{accuracy:.2f}%</td><td>{html.escape(details)}</td></tr>'
    page+='</table><h2>证据、成本与限制</h2><pre>'+html.escape('\n'.join(lines))+'</pre><p><a href="REPORT.txt">完整文本</a> · <a href="result_summary.json">结果JSON</a> · <a href="completion_verification.json">独立验收</a> · <a href="completed_manifest.json">文件哈希</a> · <a href="placeholder_diagnostic/result.json">PRM格式对照</a></p></main></html>'
    (HERE/'review.html').write_text(page)
    files=['REPORT.txt','result_summary.json','review.html','completion_verification.json','completed_manifest.json','missing_box_structure.json']
    scripts=['independent_acceptance.py','verify_local.py','render_final.py']
    (HERE/'delivery_receipt.json').write_text(json.dumps({'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'files':{n:{'bytes':(HERE/n).stat().st_size,'sha256':hashlib.sha256((HERE/n).read_bytes()).hexdigest()} for n in files},
        'analysis_code_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in scripts},
        'training_release_sha256':read(HERE/'launch_receipt.json')['release_sha256'],'passed':True},indent=2)+'\n')
    print(json.dumps({'judgment':judgment,'cost':cost,'seed42_eligible':gate['eligible']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
