"""Build the proposed route from inspected evidence; never starts GPU work."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from html import escape
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def main():
    metadata=read(HERE/'model_metadata_audit.json');assert not metadata['errors']
    server=read(HERE/'server_readonly.json')
    literature=read(HERE/'source_audit.json')['sources']
    historical=read(ROOT/'experiments/abc_seed17_resumed_v1_20261104/protocol.json')
    short=read(ROOT/'execution/prefix_triplet_seed17_48_20261107/completed_evidence/comparison.json')
    hours=sum(r['total_seconds'] for r in short['training_costs'].values())/3600
    tokens=sum(r['new_generation_tokens'] for r in short['training_costs'].values())
    rate=hours/(tokens/1e6)
    assert historical['learning_rate']==5e-6
    headline='建议转向：7B老师蒸馏热启动，再用结果奖励强化学习；以OPD后接GRPO作为预定改进路线。'
    summary=[
        '这个转向能给学生提供老师的解题轨迹与逐token分布信息，具备比继续微调PRM奖励更大的潜在收益。它是已有方法的受控复现与组合，不包装成新算法，也不预设一定提分。',
        '保留A/C-mix/C-bounded全部旧结果作为前期研究；不再为新热启动额外重跑C-bounded，优先回答蒸馏与后续RL是否带来实际增益。',
        '主比较预先指定为 SFT→OPD→GRPO 对 SFT→GRPO；纯OPD用于检查收益是否已由蒸馏拿到，SFT-only用于量化在线训练的净贡献。',
    ]
    corrections=[
        '学习率：项目实际为5e-6，已不是1e-6默认值。“LoRA最优LR约为全参数的10倍”是特定实验的相对关系，不能推出把现有LoRA学习率再乘10。短续跑仅涨0.4点也不能诊断学习率过低。',
        f'成本：本轮三组共{tokens:,}生成token，训练墙钟{hours:.2f}小时，约{rate:.2f}小时/百万token。按该实现外推，三臂各6M约{18*rate:.1f}小时，再补两臂各6M则在线训练共约{30*rate:.1f}小时，均未计老师生成、SFT与评测。原20小时全计划不能直接成立。',
        '教师前向：7B权重规模相近不代表开销相同。当前PRM是过程判别模型，生成式老师需要LM词表投影和softmax；全词表GKD还要保留老师与学生分布。用原始采样token的log-prob路径更贴近现有逐token loss，但仍须实测峰值显存与吞吐。',
        '格式：现有学生输入为纯文本 Question: ... Answer:；老师官方使用chat template。保留学生当前接口时，应为老师单独渲染同一道题，并在其后接学生原始回答token；不能把两种prompt长度混用。',
        '分词：已确认三模型vocab.json、merges.txt和新增token映射逐字相同；但base学生EOS=151643，Instruct老师EOS=151645（其generation_config也允许151643）。这是停止语义差异，不能声称同分词器就无需EOS处理。老师LM head维度152064，学生151936，无法直接对两张未对齐的完整logits张量算KL。',
        '监督含义：老师log-prob衡量老师对某个token的偏好，不是数学正确概率。OPD允许错误回答中的部分token获得正信号；旧C-bounded的“错答不得有正优势”断言不能直接套到OPD。验收仍看固定预算下的数学成绩。',
        '污染：新加入OlympiadBench只能检查跨基准泛化，不能证明它不在老师训练数据中。历史测试成绩已被查看，不能再称为从未触碰的测试；新方案应在方法与预算冻结后统一评测。',
    ]
    arms=[
        ['W','只做老师解答SFT','0','新热启动本身能达到什么水平'],
        ['R','W → GRPO','96次更新，约2M级别','同一起点上结果奖励还能增加多少'],
        ['D','W → OPD','96次更新，约2M级别','稠密教师信号单独能做到什么'],
        ['D→R','W → OPD 48次 → GRPO 48次','总96次更新，约2M级别','同预算下，先蒸馏再RL是否优于纯GRPO、纯OPD'],
    ]
    phases=[
        ['1. 教师与吞吐pilot','老师在现有500题验证集按官方CoT格式评测；另在固定512道训练题生成解答。512题将复用于后续SFT数据。补测公开1.5B-Instruct作为强起点参考。','先确认7B老师在当前2048上限下确有明显优势；记录真实token/s、截断、答对保留率与峰值显存。仅有论文分数不足以通过。'],
        ['2. 离线蒸馏','从原始1.5B base开始。最多对6993道训练题每题生成1条；按原验答器只保留答对且未截断的解答。根据512题实测吞吐，批量生成前一次性决定用全部题还是固定训练子集；不生成验证/测试题作为训练数据。LoRA SFT先1个epoch。','SFT后W应明显优于原base，并至少超过C-bounded当前63.4%验证参考；若仍只有约1点波动，先检查数据质量，不开三臂大训练。目标是明显改善，不把过关称为已显著。'],
        ['3. 在线四组对照','所有在线组从同一W开始，题序、batch、更新数、生成上限和验答规则匹配。D与D→R共用前48次OPD checkpoint，再分别续OPD或GRPO。','3个最终训练臂各96次更新，实际只需240次独立在线更新，按旧长度估算约5M生成token。先做真实模型无更新联调，再做固定小预算训练。'],
        ['4. 有限确认','主方法固定为D→R，不在结果出来后从多个变体挑赢家替换主张。只有预算剩余且主比较有明确改善，才补seed42的R与D→R短复核；离线老师数据复用。','不预先承诺每臂6M。20卡时作为总占用墙钟上限；共享争用会降低可完成范围，确认试验是否容纳由真实吞吐决定。'],
    ]
    objectives=[
        '建议采用已查阅的OPD-PPO参考变体：d_t = stopgrad(clip(log q_teacher(y_t|teacher_prompt,y_<t) − log p_old(y_t|student_prompt,y_<t), −5, 5))，用PPO clipped surrogate更新学生。clip边界与EOS规则来自选定官方实现，不作为我们的新奖励设计。',
        '这里是即时token信号：折扣因子为0，不累计后续token回报；这与通常“不折扣=gamma为1”的说法不同。teacher与old-student log-prob均来自冻结的更新前快照，无组内均值/标准差归一化。',
        'teacher侧只处理真实生成token，单独处理EOS；prompt、padding不能进入教师优势。base与instruct的原始token ID映射虽相同，prompt offset、stop token和长度上限仍要逐条核验。截断末尾普通token不能当EOS随意丢掉。',
        'TRL v0.23.1 GKDTrainer的默认目标会构造完整词表KL/JSD；它不是“把采样token概率差填进现有advantage”的同一种实现。该项目先采用与现有token loss更接近的sampled-token路径，不整体迁移到新的veRL/FSDP栈。',
        '新在线对照按参考论文固定共同reference-KL系数为0，不继承旧C-bounded的beta=0.04再无意叠加一层约束；这是新实验协议，不是修改旧结果。PPO clipping、长度分母与公共mask在新臂间一致。',
        '新臂学习率做有限校准，不做大网格。优先比较5e-6和1e-5的小预算稳定性，在线正式对照前统一冻结选择并记录代价；SFT与在线阶段的学习率不应被当成同一个参数。',
        'W保留原生LoRA适配器，阶段切换不反复merge/unmerge BF16权重。所有新臂的共同W、采样同步、旧log-prob及阶段间optimizer边界须在真实模型联调时验收。',
    ]
    budget=[
        ['教师验证、512题pilot及数据生成','必须实测；不能承诺1小时生成全部6993×2条'],
        ['SFT 1个epoch','必须实测，按实际保留token数计费'],
        ['R / D / D→R首轮','共享OPD前半段，约5M独立生成token；旧实现外推约'+f'{5*rate:.1f}'+'小时在线训练'],
        ['四模型500题验证','按最近实测约50分钟，老师评测另计'],
        ['seed42两臂确认','约4M独立生成token；旧实现外推约'+f'{4*rate:.1f}'+'小时，预算允许才开'],
        ['总预算','最多20小时实际GPU占用墙钟；先完成seed17受控对照，不承诺20小时包办所有6M训练'],
    ]
    evidence=[
        'Qwen3报告支持“离线蒸馏再在线蒸馏”的路线，其十倍效率是在Qwen3-8B及更大老师的原设置中观察到；不能直接转移为本机工时承诺。',
        'Sequential Beats Joint已研究OPD后接RL，因此该顺序是应复现的已有基线，不是本项目新算法。',
        '近期研究也显示：学生变强后继续从同一老师蒸馏可能退化；所以不建议直接把旧C-bounded checkpoint改接OPD后假定必然更好。',
    ]
    # Keep paper-derived statements brief; the detailed recipe above is our proposal.
    plan={'status':'proposed_not_dispatched','generated_at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'recommendation':headline,'summary':summary,'corrections':corrections,'arms':arms,'phases':phases,
        'objectives':objectives,'budget':budget,'existing_hours_per_million_generated_tokens':rate,
        'online_updates_per_final_arm_proposed':96,'independent_online_updates_with_shared_prefix':240,
        'update_budget_caveat':'96 updates is a proposed ceiling. Stronger W may generate longer answers. Freeze a common lower count before all arms if pilot cost exceeds the cap; do not stop individual arms at unequal update counts to force equal tokens.',
        'success_criteria':'Aim for at least about5pp pipeline improvement over current C-bounded as a practical project target, not a forecast. Separate W improvement from online learning: R-W and D→R-R are the important controls; report intervals, not just the largest percentage.',
        'server_snapshot':server,'teacher_weights_cached':False,'metadata_verified':True,'real_model_OPD_profile_verified':False,
        'new_training_started':False,'new_GPU_evaluation_started':False,'sources':literature,
        'evidence_files':{p.name:digest(p) for p in [HERE/'model_metadata_audit.json',HERE/'source_audit.json',HERE/'server_readonly.json']}}
    (HERE/'recommendation.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    model_rows=[]
    for model,v in metadata['models'].items():model_rows.append([model.split('/')[-1],v['config']['vocab_size'],v['config']['eos_token_id'],v['revision'][:12]])
    text=[headline,'状态：仅完成论文、官方源码、配置与资源只读核查；没有加载老师权重或启动任何新GPU评测/训练。','',*summary,'','需要修正',*corrections,'','最小实验']
    text += [' | '.join(map(str,r)) for r in arms]
    text += ['','执行顺序']+['\n'.join(r) for r in phases]+['','损失与工程边界',*objectives,'','预算']+[' | '.join(r) for r in budget]+['',plan['update_budget_caveat'],plan['success_criteria'],'','已有证据',*evidence,'','引用']+[(k+': '+v['url']) for k,v in literature.items()]
    (HERE/'PLAN.txt').write_text('\n'.join(text)+'\n')
    blocks=[]
    def section(title,paras=(),headers=None,rows=None):blocks.append('<section><h2>'+escape(title)+'</h2>'+''.join('<p>'+escape(s)+'</p>' for s in paras)+(table(headers,rows) if headers else '')+'</section>')
    section('推荐主线',summary)
    section('先纠正这些预算与技术判断',corrections)
    section('四个清楚的对照',['相同W、固定题序、相同在线更新上限；D与D→R复用前半段checkpoint，节约一次重复训练。旧C-bounded仅作历史参考。'],['臂','训练路径','在线预算','回答的问题'],arms)
    section('执行顺序',(),['阶段','怎么做','判定与预算'],phases)
    section('实际核实的模型兼容性',['共享词表和BPE映射已核对；EOS及输出维度不同。配置兼容性可处理，但真实模型OPD显存和吞吐尚未实测。'],['模型','LM输出维度','配置EOS','固定revision'],model_rows)
    section('目标函数与复用边界',objectives)
    section('预算口径',[plan['update_budget_caveat'],plan['success_criteria']],['项目','口径'],budget)
    section('文献支持与贡献定位',evidence)
    links=''.join('<p><a href="'+escape(v['url'])+'">'+escape(k)+'</a></p>' for k,v in literature.items())
    blocks.append('<section><h2>原始论文与官方代码</h2>'+links+'<p><a href="model_metadata_audit.json">配置核验</a> · <a href="source_audit.json">源码与文献核验</a> · <a href="recommendation.json">结构化方案</a> · <a href="PLAN.txt">文本方案</a></p></section>')
    css='body{background:#f4f6fa;color:#203047;font:15px/1.75 system-ui;margin:0}main{max-width:1120px;margin:auto;padding:42px 22px 70px}h1{font-size:28px;line-height:1.45}h2{font-size:20px;margin-top:0}section{background:white;border:1px solid #dce3ed;border-radius:12px;padding:24px;margin:20px 0}.muted{color:#61728a}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;vertical-align:top;padding:11px;border-bottom:1px solid #e4e8ef}th{background:#eef2f8}.scroll{overflow-x:auto}a{color:#3153a0}'
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>数学推理：蒸馏与RL路线评估</title><style>'+css+'</style></head><body><main><h1>数学推理：从PRM小修转向老师蒸馏与RL</h1><p class="muted">状态：方案与只读核查完成，未启动新训练｜'+escape(plan['generated_at'])+'</p>'+''.join(blocks)+'</main></body></html>'
    (HERE/'review.html').write_text(page)
    print(json.dumps({'output':str(HERE),'hours_per_M':rate,'18M_online_hours_estimate':18*rate,'30M_online_hours_estimate':30*rate,'new_training_started':False,'real_model_OPD_profile_verified':False},ensure_ascii=False))

if __name__=='__main__':main()
