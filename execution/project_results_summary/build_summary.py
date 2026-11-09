"""Summarize frozen local evidence. No SSH, model loading, or training."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from html import escape
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EX = ROOT / 'execution'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+escape(str(x))+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def csv_file(name, headers, rows):
    with (HERE/name).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    paths = {
        'seed17': EX/'abc_seed17_resumed_v1_20261104/completed_evidence/comparison.json',
        'seed42': EX/'ac_seed42_v1_20261106/completed_evidence/comparison.json',
        'two_seed': EX/'ac_seed42_v1_20261106/two_seed_summary.json',
        'short': EX/'prefix_triplet_seed17_48_20261107/completed_evidence/comparison.json',
        'uniform': EX/'prefix_triplet_seed17_48_20261107/uniform_followup_assessment.json',
        'short_check': EX/'prefix_triplet_seed17_48_20261107/completion_verification.json',
        'case_audit': EX/'abc_seed17_resumed_v1_20261104/blind_audit_20261105/audit_summary.json',
        'data_manifest': EX/'final_four_eval_20261029/evidence/data/prepared/manifest.json',
        'assets': EX/'final_four_eval_20261029/evidence/receipts/assets.json',
    }
    src = {name: read(path) for name, path in paths.items()}
    assert all(src[n]['passed'] for n in ('seed17','seed42','two_seed','short','short_check'))
    assert src['uniform']['source_comparison_sha256'] == sha(paths['short'])

    checked = []
    order = {}
    main_rows, mean_rows, interval_rows, cost_rows = [], [], [], []
    for seed in (17,42):
        key = f'seed{seed}'
        x = src[key]
        assert x['seed'] == seed and x['global_step'] == 288
        contract = read(paths[key].parent/'evaluation/288/contract.json')
        assert contract['cap'] == 2048 and contract['temperature'] == 0
        assert contract['verifier'] == 'reward-repair-v3-20261021'
        assert x['acceptance']['same_prompt_order_and_update_counts']
        if seed == 17:
            reference_contract = contract
        else:
            for k in ('engine','seed','temperature','top_p','top_k','cap','dtype','prompt','verifier','repair_sha256','unit_metadata_sha256','dataset_sha256'):
                assert contract[k] == reference_contract[k], k
        for arm, scores in x['summary'].items():
            for dataset, s in scores.items():
                p = paths[key].parent/'evaluation/288'/arm/(dataset+'.jsonl')
                assert sha(p) == s['predictions_sha256'], str(p)
                rr = records(p)
                assert len(rr) == s['n'] and sum(r['reward'] for r in rr) == s['correct']
                ids = [r['problem_id'] for r in rr]
                if dataset in order: assert ids == order[dataset]
                else: order[dataset] = ids
                checked.append(str(p.relative_to(ROOT)))
            main_rows.append([seed,arm,f"{scores['math500']['accuracy']*100:.2f}%",f"{scores['gsm8k']['accuracy']*100:.2f}%",f"{x['accounting'][arm]['generated_completion_tokens']:,}",288])
            wall = x.get('training_seconds', x.get('resumed_training_seconds_since_20261104'))[arm]/3600
            cost_rows.append([seed,arm,f"{x['accounting'][arm]['generated_completion_tokens']:,}",f'{wall:.2f}h', '11/4之后续跑段，非完整历史成本' if seed==17 else '已记录训练段合计'])
        for dataset in ('math500','gsm8k'):
            stat = x['paired'][dataset]['C_bounded minus A']
            lo,hi = stat['paired_question_bootstrap95_pp']
            interval_rows.append([seed,dataset,f"{stat['delta_pp']:+.2f}",f'[{lo:+.2f}, {hi:+.2f}]',stat['wins'],stat['losses']])

    for dataset in ('math500','gsm8k'):
        means = {arm: sum(src[f'seed{s}']['summary'][arm][dataset]['accuracy'] for s in (17,42))/2 for arm in ('A','C_bounded')}
        expected = src['two_seed']['two_seed_means'][dataset]
        assert abs(means['A']-expected['A'])<1e-12
        assert abs(means['C_bounded']-expected['C_bounded'])<1e-12
        mean_rows.append([dataset,f"{means['A']*100:.2f}%",f"{means['C_bounded']*100:.2f}%",f"{(means['C_bounded']-means['A'])*100:+.2f}"])

    short = src['short']
    short_names = {'source':'共同起点 C-bounded / step288','bounded':'C-bounded 续跑','uniform':'均匀减罚','prefix':'前缀保护'}
    short_rows, short_ci_rows, short_costs = [], [], []
    for arm, s in short['validation_summary'].items():
        p = paths['short'].parent/'evaluation'/arm/'predictions.jsonl'
        assert sha(p)==s['predictions_sha256']
        rr = records(p)
        assert len(rr)==500 and sum(r['reward'] for r in rr)==s['correct']
        checked.append(str(p.relative_to(ROOT)))
        short_rows.append([short_names[arm],f"{int(s['correct'])}/500",f"{s['accuracy']*100:.2f}%",s['truncated'],s['empty'],s['short_wrong']])
    for pair,label in [('uniform_minus_bounded','均匀减罚 − C-bounded续跑'),('prefix_minus_bounded','前缀保护 − C-bounded续跑'),('prefix_minus_uniform','前缀保护 − 均匀减罚（预定主比较）')]:
        s = short['paired_comparisons'][pair]
        lo,hi = s['paired_bootstrap_95CI_pp']
        short_ci_rows.append([label,f"{s['difference_percentage_points']:+.2f}",f'[{lo:+.2f}, {hi:+.2f}]',s['wins'],s['losses']])
    for arm,s in short['training_costs'].items():
        short_costs.append([short_names[arm],48,f"{s['new_generation_tokens']:,}",f"{s['total_seconds']/3600:.2f}h",f"{s['peak_sampled_GPU_GiB']:.2f}GiB"])

    decomposition = [[r['loser_status'],r['uniform_wins'],r['uniform_losses'],r['net_correct']] for r in src['uniform']['descriptive_discordance_decomposition']]
    assert sum(row[-1] for row in decomposition) == 8
    wrong_adv = [[17,arm,x['loss_rows'],x['loss_wrong_positive']] for arm,x in src['seed17']['accounting'].items()]
    for arm in ('A','C_bounded'): assert src['seed42']['accounting'][arm]['loss_wrong_positive']==0

    limitations = [
        '测试集主实验与500题验证短试是两组不同数据，不能直接比较53%与65%。短试是从已完成约6M训练的C-bounded继续训练，并非从初始模型开始。',
        '两个seed均值仅为描述性平均；相同测试题在两个seed上重复使用，不能当成独立翻倍样本。题目bootstrap区间不衡量训练seed方差，seed42的两项区间均包含零。',
        'seed17因用户暂停存在不对称重启历史。起点、优化器、题序与更新数已核对，但vLLM采样随机状态在重启时重建，不能声称逐位连续复现。seed42两组重启边界匹配。',
        '短试uniform首个更新前曾被过严的随机输出逐字一致检查拦住；复用了保存的32条合法on-policy样本，没有重采样或重跑bounded；故障后的采样随机状态未逐位恢复。失败尝试时间已计入成本。',
        '旧训练的LoRA合并/反合并与BF16采样行为另有数值审计；新短试统一改用已核验的原生LoRA/eager。三个短试组共享该设置，历史主实验与短试不能只按训练token数量拼成一条学习曲线。',
        '验证集已多次用于研究，且其500道题全部出现在审计过的PRM800K问题集合中；这不等同于证明它们进入本项目实际锚点子集，但该验证集不属于对PRM全新且未见的测验。MATH500/GSM8K的已知精确问题重合审计为0，不能排除未知预训练与改写暴露。',
        'C-mix到C-bounded同时涉及正确答案门控、过程项幅度和组合后的归一化方式；现有结果不能把收益全部归因于其中一个改动。',
        '错答没有正策略优势是已验证的训练信号约束；它不能单独证明数学能力提升。最终答案准确率也不等同于整段推导有效性。',
        '共享GPU和不完整历史成本使墙钟时间不适合作为算法加速证据。seed17列出的时间只覆盖11/4之后的续跑，不包含完整旧阶段、暂停和全部故障修复。',
    ]
    finding = [
        '主结果：C-bounded相对纯结果奖励A，在seed17和seed42的MATH500/GSM8K上均呈正向变化；两seed平均为+3.40和+1.48个百分点，幅度随seed变化。',
        '机制：seed17的9216条训练回答中，C-mix有526条错答得到正优势，C-bounded为0。直接混合过程分的风险在日志中得到量化，但这不是准确率提升的单因素因果证明。',
        '短试：均匀减罚65.4%，前缀保护65.6%；两者仅差1题，位置特有收益没有得到支持。两者对原C-bounded续跑的区间也均包含零。',
        '当前定位：以C-bounded的已完成主实验收尾；将均匀减罚与前缀保护列为探索性消融，不替换主方法、不宣称已证实新提升。',
    ]
    output = {
        'generated_at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'scope':'Frozen main test results and exploratory validation pilot; local summary only.',
        'source_files':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for k,p in paths.items()},
        'prediction_files_verified':checked,'prediction_rows_verified':sum(len(records(ROOT/p)) for p in checked),
        'main_test_rows':main_rows,'two_seed_means':mean_rows,'main_paired_intervals':interval_rows,
        'main_training_cost_rows':cost_rows,'short_validation_rows':short_rows,'short_intervals':short_ci_rows,
        'short_training_cost_rows':short_costs,'wrong_positive_advantage_rows':wrong_adv,'uniform_discordance':decomposition,
        'findings':finding,'limitations':limitations,'new_training_started':False,'server_state_polled':False,
    }
    (HERE/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    main_h=['训练seed','方法','MATH500测试集','GSM8K测试集','实际生成token','更新数']
    short_h=['模型','正确题数','验证准确率','截断','空回答','≤32token错答']
    csv_file('main_test_results.csv',main_h,main_rows)
    csv_file('short_validation_results.csv',short_h,short_rows)
    csv_file('two_seed_means.csv',['数据集','A','C-bounded','差值/百分点'],mean_rows)

    methods=[['A','只使用最终答案结果奖励的GRPO'],['C-mix','结果奖励与过程分直接混合，再做组内标准化'],['C-bounded','保留结果优势，只在答对的回答内部加入有界过程偏好；错答不获得正策略优势'],['均匀减罚','对PRM筛出的错答，将整条回答负优势缩小；本轮209/1536条触发，仍依赖7B PRM'],['前缀保护','对相同类型错答，去掉PRM低分步骤之前的负策略优势；本轮244/1536条触发']]
    config='策略模型 Qwen2.5-Math-1.5B base；LoRA rank16；GRPO；过程组使用既有C适配后的冻结Qwen2.5-Math-PRM-7B。MATH训练6993题、验证500题；MATH500测试500题、GSM8K测试1319题。每次更新4题×每题8条回答；主实验各288次更新，约6.1–6.3M实际生成token。统一评测使用V1、greedy、2048生成token上限和reward-repair-v3验答规则。'
    case=src['case_audit']
    case_note=f"已有隐藏模型身份的胜负题复核覆盖{case['cases']}道题、{case['responses']}段回答，{case['outcome_agreements']}/{case['responses']}与冻结验答规则一致。选定案例中有{len(case['answer_correct_invalid_process'])}条终答正确但推导存在问题、{len(case['format_only_losses'])}条数学核心正确但缺有效答案框。样本针对胜负题选取，不能据此估计总体过程有效率。"
    blocks=[]
    def add(title,paragraphs=(),headers=None,data=None):
        blocks.append('<section><h2>'+escape(title)+'</h2>'+''.join('<p>'+escape(p)+'</p>' for p in paragraphs)+(table(headers,data) if headers else '')+'</section>')
    add('结论',finding)
    add('实验设置与方法',[config],['方法','具体含义'],methods)
    add('约6M训练：正式测试结果',['C-mix只有seed17结果；不填补或推测seed42。测试结果按各自最终checkpoint冻结。'],main_h,main_rows)
    add('两个seed的描述性平均',['MATH500与GSM8K分别求均值，不把不同基准混成一个准确率。'],['数据集','A','C-bounded','差值/百分点'],mean_rows)
    add('C-bounded对A的题目配对区间',['区间是95%配对题目bootstrap；seed17较强，seed42的区间仍包含零。'],['seed','基准','差值/百分点','95%区间','独胜','独败'],interval_rows)
    add('训练信号诊断与案例',[case_note,'C-bounded过程项为 0.2 × y × (m − 答对回答的平均m)，幅度不超过0.2；相加后不再次标准差归一化。它保留原KL约束。'],['seed','方法','训练回答数','错答获得正优势'],wrong_adv)
    add('同一起点续跑48次：验证集短试',['以下为原validation.jsonl的500题，绝不是MATH500测试成绩。每组新增1536条回答，题序、更新次数、运行设置一致；起点是seed17 C-bounded checkpoint-288。'],short_h,short_rows)
    add('短试配对比较',['前缀保护 − 均匀减罚是预先确定的主比较；均匀减罚比C-bounded净多对8题仍属探索性结果。'],['比较','差值/百分点','95%区间','独胜','独败'],short_ci_rows)
    add('均匀减罚的8题净增益来自哪些胜负题',['按输的一方失败类型做描述性拆分，不是因果贡献分解。symbolic表示答案可解析但不等于标准答案，不代表推导有效。均匀减罚没有取消PRM，也不是对所有错答乘同一个固定常数。'],['输方记录状态','均匀减罚独胜','均匀减罚独败','净多对'],decomposition)
    add('实际成本：主实验',['seed17时间不完整；共享GPU的墙钟时间不能用来证明算法更快。'],['seed','方法','实际生成token','记录墙钟','口径'],cost_rows)
    add('实际成本：新增短试',['三组训练累计约4.30小时；四模型验证约50.4分钟。uniform已计入失败尝试约92秒，首批PRM forward计时缺失。显存是采样到的进程峰值。'],['方法','新增更新','新增token','训练墙钟','采样峰值显存'],short_costs)
    add('结论边界',limitations)
    links=[('seed17主实验','../abc_seed17_resumed_v1_20261104/review.html'),('seed42主实验','../ac_seed42_v1_20261106/review.html'),('48次更新短试','../prefix_triplet_seed17_48_20261107/review.html'),('结构化汇总','results.json'),('主结果CSV','main_test_results.csv'),('验证短试CSV','short_validation_results.csv'),('纯文本报告','REPORT.txt')]
    blocks.append('<section><h2>证据与下载</h2><p>'+' · '.join('<a href="'+escape(url)+'">'+escape(label)+'</a>' for label,url in links)+'</p><p>本次仅整理本地已完成证据，未启动训练，也未查询当前服务器占用。</p></section>')
    style='body{margin:0;background:#f3f5f9;color:#203047;font:15px/1.7 system-ui,-apple-system,sans-serif}main{max-width:1120px;margin:auto;padding:44px 22px 70px}h1{font-size:30px;line-height:1.3;margin:0 0 12px}h2{font-size:21px;margin:0 0 14px}p{margin:10px 0}section{background:white;border:1px solid #dce2eb;border-radius:12px;padding:24px;margin:20px 0;break-inside:avoid}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e2e7ef;vertical-align:top}th{background:#eef2f8;color:#425777}td:nth-child(n+3){font-variant-numeric:tabular-nums}a{color:#3154a3}.muted{color:#65758b}.scroll{overflow-x:auto}@media print{body{background:white}main{padding:0}section{border-radius:0;break-inside:auto}h2,thead{break-after:avoid}tr{break-inside:avoid}}'
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECE285 数学推理RL：结果汇总</title><style>'+style+'</style></head><body><main><h1>ECE285 数学推理RL：结果汇总</h1><p class="muted">已完成主实验（seed17 / seed42）与48次更新探索短试｜生成时间 '+escape(output['generated_at'])+'</p>'+''.join(blocks)+'</main></body></html>'
    (HERE/'review.html').write_text(page)
    text=['ECE285 数学推理RL：结果汇总',output['generated_at'],'',*finding,'',config]
    for title,headers,data in [('正式测试结果',main_h,main_rows),('两个seed均值',['数据集','A','C-bounded','差值/百分点'],mean_rows),('主结果配对区间',['seed','基准','差值/百分点','95%区间','独胜','独败'],interval_rows),('新增短试验证结果',short_h,short_rows),('短试配对区间',['比较','差值/百分点','95%区间','独胜','独败'],short_ci_rows),('均匀减罚胜负题拆分',['输方状态','独胜','独败','净多对'],decomposition),('主实验记录成本',['seed','方法','token','墙钟','口径'],cost_rows),('短试成本',['方法','新增更新','token','墙钟','显存'],short_costs)]:
        text += ['',title,' | '.join(headers),*(' | '.join(map(str,row)) for row in data)]
    text += ['', '案例复核',case_note,'','限制',*limitations,'','仅整理本地结果；未启动新训练。']
    (HERE/'REPORT.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps({'generated':str(HERE),'main_rows':len(main_rows),'prediction_files_verified':len(checked),'prediction_rows_verified':output['prediction_rows_verified'],'new_training_started':False},ensure_ascii=False))


if __name__=='__main__':
    main()
