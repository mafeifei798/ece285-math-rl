"""Render verified final evidence; no model calls or remote mutations."""
from pathlib import Path
from collections import Counter
import html,json,hashlib
HERE=Path(__file__).resolve().parent
E=HERE/'completed_evidence'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main():
    v=read(HERE/'completion_verification.json');assert v['passed']
    c=read(E/'comparison.json');a=read(HERE/'independent_tensor_acceptance.json');assert a['passed']
    pair=c['paired_comparisons']['fixed_length_minus_grpo'];low,high=pair['paired_bootstrap_95CI_pp']
    labels={'source':'源 checkpoint（第288步）','grpo':'原 C-bounded，再训48步','fixed_length':'仅固定长度分母，再训48步'}
    tab=[]
    for arm in labels:
        s=c['validation_summary'][arm]
        tab.append([labels[arm],f"{int(s['correct'])}/500 ({s['accuracy']*100:.1f}%)",
                    f"{s['truncated']}/500 ({s['truncation_rate']*100:.1f}%)",
                    f"{s['missing_or_invalid_box']}/500 ({s['missing_or_invalid_box_rate']*100:.1f}%)",
                    f"{s['tokens']/500:.1f}"])
    cost=[]
    for arm in ['grpo','fixed_length']:
        s=c['training_costs'][arm]
        cost.append([labels[arm],48,f"{s['new_generation_tokens']:,}",f"{s['truncated']}/1536 ({s['truncation_rate']*100:.2f}%)",
                     f"{s['missing_or_invalid_box']}/1536 ({s['missing_or_invalid_box_rate']*100:.2f}%)",
                     f"{s['total_seconds']/3600:.2f} h",f"{s['peak_sampled_GPU_GiB']:.2f} GiB"])
    left=rows(E/'evaluation/fixed_length/predictions.jsonl');right=rows(E/'evaluation/grpo/predictions.jsonl')
    wins=Counter(r['status'] for l,r in zip(left,right) if l['reward']==1 and r['reward']==0)
    losses=Counter(l['status'] for l,r in zip(left,right) if l['reward']==0 and r['reward']==1)
    breakdown=[dict(loser_status=s,fixed_length_wins=wins[s],fixed_length_losses=losses[s],net=wins[s]-losses[s]) for s in sorted(set(wins)|set(losses))]
    if low>0:decision='固定分母在本次单seed续跑验证中更高；仍不能外推为跨seed稳定改进。'
    elif high<0:decision='固定分母在本次验证中更低，不建议扩大该变体的预算。'
    elif pair['difference_percentage_points']>0:decision='固定分母的点估计更高，但区间跨零，尚不能确认收益。'
    else:decision='没有确认固定分母提高准确率；本轮不自动扩预算。'
    tg=c['training_costs']['grpo'];tf=c['training_costs']['fixed_length']
    vg=c['validation_summary']['grpo'];vf=c['validation_summary']['fixed_length']
    mechanism=(f"训练截断率：{tg['truncation_rate']*100:.2f}% → {tf['truncation_rate']*100:.2f}%；"
               f"验证截断率：{vg['truncation_rate']*100:.1f}% → {vf['truncation_rate']*100:.1f}%。"
               '训练输出变短没有转化为本轮验证改善。胜负题中，parsed/symbolic错答各17胜17负，缺框各3胜3负；'
               '截断对应2胜5负，描述性净差为−3题。此分解不证明因果。')
    source_sha=read(E/'admission/no_update/result.json')['source_adapter_sha256']
    notes=[
        '仅loss长度分母不同：原版按各回答有效长度平均；新版本统一除以2048。奖励、C PRM、group标准差、advantage、KL系数、LR与生成上限不变。原有policy项与KL项一起经过该reduction，没有另加缩放补偿。',
        '同源checkpoint288、同优化器/调度器/Torch RNG、同seed17题序，各48次更新到336；实际token数不同，分别列出。',
        '两臂共用已验证的native-LoRA/eager运行时。初始化引擎seed0；历史vLLM RNG未逐位恢复，相同seed也不保证随机生成逐token一致。',
        '缺框率按原验答器missing_or_invalid_box状态统计，分母500；截断单列，不把截断后的缺框重复计入该类别。',
        '验证集已在此前研究中使用；这里没有运行MATH500/GSM8K测试或据测试集选设置。',
        'bootstrap反映题目抽样，不包含训练seed不确定性；这是约6M后续跑48步的小试，不是从头训练6M。',
        '第一次无更新联调因启动环境缺CPATH而在生成前失败。失败日志与目录完整保留；恢复既有头文件路径后通过。该故障前训练更新和已生成训练批次均为0。',
        f"旧文件保护：原保护清单{c['old_protected_files_verified']}项，本轮新增历史清单{c['additional_historical_files_verified']}项，均已核验未变。",
        '只完成本次授权的两组短试，不自动启动新seed、长训或其他方法。',
    ]
    if (HERE/'evaluation_repeatability.json').exists():
        repeat=read(HERE/'evaluation_repeatability.json')
        notes.append(f"重评波动：同一源checkpoint旧轮{int(repeat['old_source_correct'])}/500，本轮{int(repeat['new_source_correct'])}/500，净差0.4个百分点；20题的正确性发生变化。源适配器与共同contract字段一致，但两轮模型切换数量/顺序不同，具体原因未定位。只使用本轮统一结果作主比较；bootstrap也不覆盖这种运行间差异。")
    text=['两组48步续跑：loss长度归一化对照',decision,'',
          f"主比较（固定分母−原版）：{pair['difference_percentage_points']:+.2f}个百分点，配对bootstrap95%区间[{low:+.2f}, {high:+.2f}]；胜{pair['wins']}题、负{pair['losses']}题。",'','验证集500题：']
    text+=[' | '.join(map(str,r)) for r in [['模型','准确率','截断率','缺框率','平均生成token']]+tab]
    text+=['',mechanism,'','训练成本与输出统计：']+[' | '.join(map(str,r)) for r in [['模型','新更新','新token','截断','缺框','墙钟','采样峰值显存']]+cost]
    text+=['','胜负题中失败一方的类型（仅描述，不是因果分解）：',json.dumps(breakdown,ensure_ascii=False,indent=2),'','控制与限制：']+notes
    text+=['',f'源适配器SHA256：{source_sha}','无更新联调、历史输入hash、最终checkpoint与独立FP64 token/advantage验收见本目录JSON文件。']
    (HERE/'REPORT.txt').write_text('\n'.join(text)+'\n')
    def table(headers,data):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table>'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>长度归一化配对短试</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#203047;background:#f5f7fb;margin:0}main{max-width:1080px;margin:42px auto;padding:30px;background:white;border-radius:16px}h1{font-size:28px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{padding:12px;border-bottom:1px solid #dce3ed;text-align:left}th{background:#eef3fa}.result{padding:18px;background:#edf4ff;border-left:4px solid #3b6bc5}code{word-break:break-all;font-size:12px}li{margin:9px 0}</style><main><h1>两组48步续跑：loss 长度归一化对照</h1><p class="result">'+html.escape(decision)+'</p><p>'+html.escape(text[3])+'</p><h2>同一验证集 · 500题</h2>'+table(['模型','准确率','截断','缺框','平均token'],tab)+'<h2>新增训练成本</h2>'+table(['模型','更新','生成token','截断','缺框','墙钟','峰值显存'],cost)+'<h2>胜负题的失败类型</h2>'+table(['失败方状态','固定分母胜','固定分母负','净胜'],[[x['loser_status'],x['fixed_length_wins'],x['fixed_length_losses'],x['net']] for x in breakdown])+'<p>此分解仅描述结果，不能证明训练改动的因果机制。</p><h2>实验控制与限制</h2><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in notes)+'</ul><p>源适配器 <code>'+source_sha+'</code></p></main></html>'
    page=page.replace('<h2>新增训练成本</h2>','<p class="result">'+html.escape(mechanism)+'</p><h2>新增训练成本</h2>')
    (HERE/'review.html').write_text(page)
    (HERE/'result_summary.json').write_text(json.dumps({'decision':decision,'primary_comparison':pair,'mechanism_description':mechanism,'validation_table':tab,'training_table':cost,'discordance_breakdown':breakdown,'notes':notes},ensure_ascii=False,indent=2)+'\n')
    deliver=['REPORT.txt','review.html','result_summary.json','completion_verification.json','independent_tensor_acceptance.json','completed_manifest.json','evaluation_repeatability.json']
    (HERE/'delivery_receipt.json').write_text(json.dumps({'passed':True,'files':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in deliver}},indent=2)+'\n')
    print(json.dumps({'decision':decision,'primary_comparison':pair,'validation':tab},ensure_ascii=False))


if __name__=='__main__':main()
