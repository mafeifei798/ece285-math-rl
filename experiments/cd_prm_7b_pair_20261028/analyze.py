"""CPU-only paired rank/advantage audit; no policy accuracy claim."""
from collections import Counter,defaultdict
import math
import statistics
from common import *


def mean(xs):return statistics.mean(xs) if xs else None
def fraction(n,d):return n/d if d else None
def pearson(a,b):
    if len(a)<2:return None
    av,bv=mean(a),mean(b);aa=sum((v-av)**2 for v in a);bb=sum((v-bv)**2 for v in b)
    return sum((x-av)*(y-bv) for x,y in zip(a,b))/math.sqrt(aa*bb) if aa and bb else None


def group_metrics(group):
    y=[r['outcome'] for r in group];m=[r['min_score'] if r['eligible'] else 0 for r in group]
    rewards=[yi+.2*mi for yi,mi in zip(y,m)];advs=cpu_advantages(rewards)
    ranks=[1+sum(v>p for v in m)+.5*(sum(v==p for v in m)-1) for p in m]
    return dict(outcomes=y,min_scores=m,rewards=rewards,advantages=advs,ranks=ranks,
        kind='all_wrong' if sum(y)==0 else 'all_correct' if sum(y)==len(y) else 'mixed',
        wrong_positive=sum(not yi and a>0 for yi,a in zip(y,advs)),
        wrong_count=sum(not yi for yi in y),mean_abs_adv=mean([abs(a) for a in advs]),
        outcome_auc=auc([p for p,yi in zip(m,y) if yi],[p for p,yi in zip(m,y) if not yi]))


def pb_metrics(items):
    valid=[r for r in items if r['label']==-1];error=[r for r in items if r['label']!=-1]
    accept=fraction(sum(min(r['step_scores'])>=.5 for r in valid),len(valid))
    hits=0
    for r in error:
        first=next((i for i,p in enumerate(r['step_scores']) if p<.5),-1)
        hits+=first==r['label']
    error_acc=fraction(hits,len(error));h=2*accept*error_acc/(accept+error_acc) if accept is not None and error_acc is not None and accept+error_acc else 0.
    pos=[r['min_score'] for r in valid if r['outcome']];neg=[r['min_score'] for r in error if r['outcome']]
    return dict(n=len(items),valid_count=len(valid),error_count=len(error),valid_acceptance=accept,first_error_accuracy=error_acc,harmonic_mean=h,
        answer_correct_valid_vs_invalid_AUROC=auc(pos,neg),mean_valid_min_score=mean(pos),mean_invalid_answer_correct_min_score=mean(neg))


def main():
    assert_root();verify_release();sources={a:rows(OUT/f'scores_{a}.jsonl') for a in ['B','C','D']}
    ix={a:{r['example_id']:r for r in data} for a,data in sources.items()}
    assert len({tuple(sorted(v)) for v in ix.values()})==1
    grouped={a:defaultdict(list) for a in ix}
    for a,data in sources.items():
        for r in data:
            if r['dataset'] in ['probe','illustrations']:grouped[a][(r['dataset'],str(r['group_id']))].append(r)
    out_groups=[];metrics={}
    for a,gs in grouped.items():
        stats=defaultdict(list)
        for (dataset,gid),g in gs.items():
            # Existing grouping is never changed by a model's scores.
            assert len(g)==8
            d=group_metrics(g);stats[(dataset,d['kind'])].append(d)
            out_groups.append(dict(arm=a,dataset=dataset,group_id=gid,example_ids=[r['example_id'] for r in g],**d))
        metrics[a]={}
        for (dataset,kind),items in stats.items():
            metrics[a][dataset+'/'+kind]=dict(groups=len(items),wrong_positive=sum(v['wrong_positive'] for v in items),wrong_count=sum(v['wrong_count'] for v in items),
                wrong_positive_fraction=fraction(sum(v['wrong_positive'] for v in items),sum(v['wrong_count'] for v in items)),
                mean_absolute_advantage=mean([v['mean_abs_adv'] for v in items]),
                mean_group_outcome_AUROC=mean([v['outcome_auc'] for v in items if v['outcome_auc'] is not None]))
    pb_ids=[i for i,r in ix['B'].items() if r['dataset']=='processbench' and all(ix[a][i]['eligible'] for a in ix)]
    pb={a:pb_metrics([ix[a][i] for i in pb_ids]) for a in ix}
    # Preserve all selected solutions for one question in each bootstrap draw.
    pb_original={r['example_id']:r for r in rows(OUT/'processbench.jsonl')}
    clusters=defaultdict(list)
    for i in pb_ids:clusters[question_key(pb_original[i]['problem'])].append(i)
    cluster_keys=list(clusters)
    bootstrap=random.Random(285);diffs=[]
    for _ in range(CFG['bootstrap_replicates']):
        ids=[i for k in bootstrap.choices(cluster_keys,k=len(cluster_keys)) for i in clusters[k]]
        cm=pb_metrics([ix['C'][i] for i in ids]);dm=pb_metrics([ix['D'][i] for i in ids])
        c=cm['answer_correct_valid_vs_invalid_AUROC'];d=dm['answer_correct_valid_vs_invalid_AUROC']
        if c is not None and d is not None:diffs.append(d-c)
    diffs.sort();ci=[diffs[int(.025*(len(diffs)-1))],diffs[int(.975*(len(diffs)-1))]] if diffs else None
    rank_changes={}
    primary={a:[r for r in data if r['dataset']=='probe' and r['eligible']] for a,data in sources.items()}
    common=[r['example_id'] for r in primary['B'] if all(ix[a][r['example_id']]['eligible'] for a in ix)]
    for a in ['C','D']:
        b=[ix['B'][i]['min_score'] for i in common];v=[ix[a][i]['min_score'] for i in common]
        flips=[]
        for k in grouped[a]:
            if k[0]!='probe':continue
            before=group_metrics(grouped['B'][k]);after=group_metrics(grouped[a][k]);flips.extend((x>0)!=(y>0) for x,y in zip(before['advantages'],after['advantages']))
        rank_changes[a]=dict(score_pearson_vs_B=pearson(b,v),mean_min_score_delta=mean([y-x for x,y in zip(b,v)]),positive_advantage_sign_flip_fraction=mean(flips))
    guards=protect()
    summary=dict(status='COMPLETE_PRM_ADAPTATION_AND_OFFLINE_ANALYSIS_NO_POLICY_RL',config=CFG,
        data=read(OUT/'batch_manifest.json'),buffer=read(OUT/'buffer_manifest.json'),anchors=read(OUT/'anchor_manifest.json'),
        paired_wiring=read(OUT/'paired_wiring.json'),groups=metrics,processbench=pb,D_minus_C_answer_correct_AUROC_CI95=ci,
        ranking_changes=rank_changes,preservation=guards,actor_optimizer_updates=0,processbench_unique_question_clusters=len(clusters),
        limitations=['Single PRM-adaptation seed; no C/D policy was trained or evaluated.',
        'Verifier-incorrect is not necessarily mathematically invalid reasoning.',
        'Primary probe is outcome-stratified, so aggregate rates are not prevalence estimates.',
        'ProcessBench subset is fixed and overlap-aware; inherited exposure is not removed.',
        'Absolute score reductions or positive-advantage changes do not prove improved mathematical ability.',
        'Floating-point scoring backend is shared B/C/D; original stored B scores are not silently overwritten.'])
    atomic(OUT/'summary.json',summary);write_rows(OUT/'group_advantages.jsonl',out_groups)
    lines=['7B C/D 配对 PRM 适配与离线检查完成','只训练了评分器 LoRA；没有策略 RL。',
        f"初始策略采样：512题×4＝{summary['buffer']['rows']}条。共享锚点：{summary['anchors']['count']}条。",
        f"适配更新：C/D 各{summary['data']['steps']}步，输入和初始adapter相同，只有成功项系数不同。",'',
        'ProcessBench 固定分层子集（不是官方完整成绩）：']
    for a,m in pb.items():lines.append(a+' '+json.dumps(m,ensure_ascii=False))
    lines+=['','分层组内信号：',json.dumps(metrics,ensure_ascii=False,indent=2),'','分数与advantage变化：',json.dumps(rank_changes,ensure_ascii=False,indent=2),
        '','D-C答对内部AUROC差区间：'+str(ci),'','限制：']+summary['limitations']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    import html
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>7B C/D 配对适配</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.8 system-ui;color:#20324a}pre{white-space:pre-wrap;background:#f2f6fa;padding:24px;border-radius:12px}</style><h1>7B C/D 配对 PRM 适配</h1><pre>'+html.escape('\n'.join(lines))+'</pre>')
    event('complete',actor_optimizer_updates=0,preservation=guards)


if __name__=='__main__':main()
