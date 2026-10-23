"""Frozen saved-rollout comparison; CPU only, no training or reward installation.

Usage (only after frozen remote scoring has completed):
  python -B experiments/pure_official_prm_offline_20261022/analyze.py
"""
import argparse
import collections
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pure_integration_20261022'))
from common import atomic, read, read_jsonl, sha, write_jsonl
from signals import task_rewards
from vendor import pure_core as official
from offline_check import independent_reference

ARMS = ('P_VR', 'QWEN_PRM', 'PURE_NATIVE', 'PURE_RL_PROMPT')
PRM_ARMS = ARMS[1:]
EPS = 1e-7


def sign(value):
    return 1 if value > EPS else (-1 if value < -EPS else 0)


def mean(values):
    return statistics.mean(values) if values else None


def average_ranks(values):
    """Ascending ranks with exact ties, no third-party statistics dependency."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in order[i:j]:
            ranks[k] = rank
        i = j
    return ranks


def spearman(left, right):
    x, y = average_ranks(left), average_ranks(right)
    mx, my = mean(x), mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v*v for v in dx) * sum(v*v for v in dy))
    return sum(a*b for a, b in zip(dx, dy)) / denom if denom else None


def scalar_reward(probabilities):
    r = 2 * torch.tensor(probabilities, dtype=torch.float64) - 1
    assert r.numel() and torch.isfinite(r).all()
    return float((.5 * torch.softmax(-r / .1, dim=0) * r).sum())


def group_type(outcomes):
    return 'all_correct' if all(outcomes) else ('all_wrong' if not any(outcomes) else 'mixed')


def checked_signals(task, mask, groups):
    raw, returns = official.compute_rloo_outcome_advantage(task, mask, groups, adv_norm=False)
    calls = []
    original = official.verl_F.masked_whiten
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    official.verl_F.masked_whiten = tracked
    try:
        white, _ = official.compute_rloo_outcome_advantage(task, mask, groups, adv_norm=True)
    finally:
        official.verl_F.masked_whiten = original
    assert len(calls) == 1
    oracle_white, oracle_returns = independent_reference(task, mask, groups)
    torch.testing.assert_close(returns, oracle_returns, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(white, oracle_white, atol=5e-6, rtol=5e-6)
    # Independent raw baseline audit, using double precision and fixed width.
    for i, g in enumerate(groups):
        other = [j for j, gg in enumerate(groups) if j != i and gg == g]
        expected = oracle_returns[i].double() - oracle_returns[other].double().sum() / (len(other)*2048)
        torch.testing.assert_close(raw[i].double(), expected, atol=5e-6, rtol=5e-6)
    return raw, white, returns


def compare_outcomes(indices, metrics, arm):
    correct = [i for i in indices if metrics[i]['outcome'] == 1]
    incorrect = [i for i in indices if metrics[i]['outcome'] == 0]
    wins = losses = ties = 0
    for a in correct:
        for b in incorrect:
            delta = metrics[a]['variants'][arm]['task_reward'] - metrics[b]['variants'][arm]['task_reward']
            wins += delta > EPS
            losses += delta < -EPS
            ties += abs(delta) <= EPS
    n = wins + losses + ties
    return dict(correct_incorrect_pairs=n, correct_higher=wins, incorrect_higher=losses,
                tied=ties, concordance=(wins + .5*ties)/n if n else None)


def summarize_costs(evidence):
    # Preserve exact remote evidence rather than guessing field semantics.
    names = ('elapsed_seconds', 'peak_allocated_GiB', 'peak_reserved_GiB',
             'peak_process_GiB', 'new_generations', 'optimizer_steps',
             'model_frozen', 'policy_model_loaded', 'revision', 'model_revision', 'costs', 'forward_seconds')
    return {k: evidence[k] for k in names if k in evidence}


def render(summary, empty_cases, out):
    def cell(v):
        if v is None:
            return '—'
        return html.escape(f'{v:.4f}' if isinstance(v, float) else str(v))
    labels = dict(P_VR='结果奖励 P-VR', QWEN_PRM='Qwen 官方 PRM',
                  PURE_NATIVE='PURE PRM：原生题目', PURE_RL_PROMPT='PURE PRM：官方 RL 提示词')
    score_rows = []
    for arm in PRM_ARMS:
        s = summary['scoring']['empty'][arm]
        a = summary['advantage']['empty'][arm]
        score_rows.append('<tr>' + ''.join(f'<td>{cell(v)}</td>' for v in [labels[arm],
            f"{s['signed_positive']}/{s['n']}",
            f"{s['positive_probability_min']:.6f}–{s['positive_probability_max']:.6f}",
            f"{a['positive_raw_terminal']}/{a['n']}",
            f"{a['positive_whitened_terminal']}/{a['n']}",
            f"{a['rank_first']}/{a['n']}",
            f"{a['outranked_correct_peer_count']}/{a['correct_peer_count']}"]) + '</tr>')
    sample_rows = []
    for e in empty_cases:
        for arm in ARMS:
            v = e['variants'][arm]
            sample_rows.append('<tr>' + ''.join(f'<td>{cell(x)}</td>' for x in [
                e['batch_id'], e['problem_id'], labels[arm], v.get('positive_probability'),
                v['task_reward'], v.get('raw_RLOO_terminal_advantage'),
                v.get('whitened_terminal_advantage'), v['rank_in_question_by_task_reward'],
                f"{v['higher_than_correct_peers']}/{v['correct_peers']}"]) + '</tr>')
    comparisons = []
    for name, data in summary['scoring']['group_comparisons'].items():
        comparisons.append(f'<li><b>{html.escape(name)}</b>：组内 Spearman 均值 {cell(data["mean_group_spearman"])}, '
                           f'可定义 {data["defined_groups"]}/{data["groups"]} 组。</li>')
    quality = []
    for arm in PRM_ARMS:
        c = summary['scoring']['outcome_concordance'][arm]
        quality.append(f'<li>{labels[arm]}：正确/错误回答成对一致率 {cell(c["concordance"])}, '
                       f'{c["correct_incorrect_pairs"]} 对；它只检验最终答案排序，不是过程正确性。</li>')
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PURE / Qwen PRM 离线对照</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;line-height:1.65;color:#17243b;background:#f7f9fc}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}.box{{background:white;border:1px solid #dce3ef;border-radius:12px;padding:18px 22px;margin:16px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{text-align:left;padding:9px 10px;border-bottom:1px solid #dce3ef}}th{{background:#eaf0fa}}code{{background:#eaf0fa;padding:2px 5px}}.scroll{{overflow:auto}}.note{{color:#59667b}}a{{color:#1757b6}}</style>
<h1>PURE 官方 PRM 与 Qwen 7B PRM：现有回答离线对照</h1>
<p>只打分与 CPU 信号回放；新生成 0，参数更新 0。策略模型仍是 Qwen，两个评分器也都基于 Qwen。此轮不决定换模型。</p>
<div class="box"><b>先看空回答：</b>同一批 288 条已保存回答中有 9 条空回答。评分全部保留；由于旧日志重构 token 的长度问题，advantage 仅回放 8 个完整批次、256 条回答，包含 8 条空回答。</div>
<div class="scroll"><table><thead><tr><th>评分器</th><th>空回答 signed 分&gt;0</th><th>空回答 p(正类)</th><th>raw advantage&gt;0</th><th>归一化后&gt;0</th><th>组内并列第一</th><th>压过正确同题回答</th></tr></thead><tbody>{''.join(score_rows)}</tbody></table></div>
<p class="note">后四列中 advantage、对应组内排序和正确同题回答比较均限于 8 个可回放批次。排名允许并列；“正优势”意味着此离线反事实信号会鼓励该 token，不等于已发生训练或未来训练一定坍塌。</p>
<h2>公平控制与官方实现边界</h2><div class="box"><ul>
<li>固定原始 288 条文本、步骤、组结构、actor 位置重构、0.5 权重、温度 0.1、固定宽度 2048、RLOO 和仅一次全批次 whitening；KL 固定 0。</li>
<li>Qwen 使用官方 chat + &lt;extra_0&gt;；PURE 使用官方模型卡的题目原文 + 逐步追加换行打分。另列 PURE 官方 RL 提示词敏感性结果，两者不混合挑最好。</li>
<li>PURE signed 分是 p₁−p₀，不是把 signed 分再做 2r−1。CPU 独立 return / advantage oracle 与官方函数一致。</li>
<li>canonical 步骤与旧 Qwen 保持一致；旧 rollout 没有原始 emitted token IDs。因而不是完整官方 RL 输入逐 token 重现，也不是历史 A/B advantage 的复算。</li>
<li>这批样本围绕历史空回答所在批次抽取，不代表总体；没有新的人工过程标注，不能宣称过程准确率改善。</li></ul></div>
<h2>组内排序与非空回答</h2><ul>{''.join(comparisons)}{''.join(quality)}</ul>
<h2>九条空回答逐例</h2><div class="scroll"><table><thead><tr><th>批次</th><th>题目</th><th>配置</th><th>p(正类)</th><th>总任务奖励</th><th>raw 终点优势</th><th>white 终点优势</th><th>组内排名</th><th>压过正确同题</th></tr></thead><tbody>{''.join(sample_rows)}</tbody></table></div>
<h2>成本与限制</h2><pre>{html.escape(json.dumps(summary['costs'],ensure_ascii=False,indent=2))}</pre>
<p>全部评分、逐例指标与按组类型分层数据见 <a href="summary.json">summary.json</a>、<a href="per_empty.json">per_empty.json</a>、<a href="per_response_metrics.jsonl">per_response_metrics.jsonl</a>。未启动正式训练，未改验答规则，未安装新奖励或过滤器。</p></html>'''
    (out/'review.html').write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'execution/pure_official_prm_offline_20261022')
    parser.add_argument('--inputs', type=Path, default=ROOT/'execution/pure_empty_boundary_20261022/inputs.jsonl')
    parser.add_argument('--qwen-scores', type=Path, default=ROOT/'execution/pure_empty_boundary_20261022/remote/frozen_scoring_v2/scored.jsonl')
    parser.add_argument('--pure-scores', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    out = args.output
    pure_path = args.pure_scores or out/'remote/scored.jsonl'
    evidence_path = args.evidence or out/'remote/result.json'
    evidence = read(evidence_path)
    assert evidence.get('passed'), 'Frozen scoring must complete and pass first'
    if 'new_generations' in evidence:
        assert evidence['new_generations'] == 0
    if 'optimizer_steps' in evidence:
        assert evidence['optimizer_steps'] == 0
    rows, qwen, pure = read_jsonl(args.inputs), read_jsonl(args.qwen_scores), read_jsonl(pure_path)
    assert len(rows) == len(qwen) == len(pure) == 288
    assert sum(not r['completion'].strip() for r in rows) == 9
    if 'input_sha256' in evidence:
        assert evidence['input_sha256'] == sha(args.inputs)
    torch.set_num_threads(2)
    metrics, batches, groups = [], collections.defaultdict(list), collections.defaultdict(list)
    for i, (r, q, p) in enumerate(zip(rows, qwen, pure)):
        digest = hashlib.sha256(r['completion'].encode()).hexdigest()
        assert q['index'] == p['index'] == i
        assert digest == q['input_sha256'] == p['input_sha256']
        probability_sets = dict(QWEN_PRM=q['native']['step_scores'],
            PURE_NATIVE=p['variants']['native_raw']['step_scores'],
            PURE_RL_PROMPT=p['variants']['official_rl_prompt']['step_scores'])
        nsteps = len(probability_sets['QWEN_PRM'])
        assert nsteps > 0 and all(len(ps) == nsteps for ps in probability_sets.values())
        for name in ('native_raw', 'official_rl_prompt'):
            v = p['variants'][name]
            assert len(v['positions']) == nsteps
            assert sorted(set(v['positions'])) == v['positions']
            assert len(v['signed_scores']) == nsteps
            assert max(abs(2*a-1-b) for a,b in zip(v['step_scores'],v['signed_scores'])) < 2e-5
        variants = {'P_VR':dict(task_reward=float(r['outcome']))}
        for arm, ps in probability_sets.items():
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in ps)
            variants[arm] = dict(task_reward=scalar_reward(ps), min_positive_probability=min(ps),
                mean_positive_probability=mean(ps), step_count=nsteps)
            if not r['completion'].strip():
                assert nsteps == 1
                variants[arm].update(positive_probability=ps[0], signed_score=2*ps[0]-1)
        metrics.append(dict(index=i,batch_id=r['batch_id'],problem_id=r['problem_id'],
            source_arm=r['source_arm'],source_line=r['source_line'],empty=not r['completion'].strip(),
            outcome=float(r['outcome']),terminated=r['terminated'],
            raw_sha256=digest,old_logged_reward=r['old_reward'],
            old_PRM_was_not_called=not r['old_prm_scores'],logged_length_matches=q['logged_length_matches'],
            reconstructed_length=len(q['reconstructed_ids']),advantage_available=False,variants=variants))
        batches[r['batch_id']].append(i)
        groups[(r['batch_id'],r['problem_id'])].append(i)
    assert len(batches) == 9 and all(len(ix) == 32 for ix in batches.values())
    for ix in groups.values():
        assert len(ix) == 8
        kind = group_type([metrics[i]['outcome'] for i in ix])
        for i in ix:
            metrics[i]['group_type'] = kind
            peers = [j for j in ix if j != i]
            for arm in ARMS:
                value = metrics[i]['variants'][arm]
                value.update(rank_in_question_by_task_reward=1+sum(
                    metrics[j]['variants'][arm]['task_reward'] > value['task_reward']+EPS for j in peers),
                    correct_peers=sum(metrics[j]['outcome'] == 1 for j in peers),
                    higher_than_correct_peers=sum(metrics[j]['outcome'] == 1 and
                        value['task_reward'] > metrics[j]['variants'][arm]['task_reward']+EPS for j in peers))
    skipped, usable_batches = [], []
    oracle_checks = 0
    for batch_id, ix in batches.items():
        failures = [i for i in ix if 'alignment_error' in qwen[i]]
        if failures:
            skipped.append(dict(batch_id=batch_id,indices=failures,
                reasons=[qwen[i]['alignment_error'] for i in failures],
                policy='Exclude the entire 32-row batch from advantage replay; retain all raw scoring'))
            continue
        usable_batches.append(batch_id)
        lengths = [len(qwen[i]['reconstructed_ids']) for i in ix]
        positions = [qwen[i]['alignment']['actor_positions'] for i in ix]
        outcomes = [rows[i]['outcome'] for i in ix]
        group_ids = [rows[i]['problem_id'] for i in ix]
        assert sorted(collections.Counter(group_ids).values()) == [8,8,8,8]
        mask = (torch.arange(2048)[None,:] < torch.tensor(lengths)[:,None]).float()
        prob_sets = dict(P_VR=[qwen[i]['native']['step_scores'] for i in ix],
            QWEN_PRM=[qwen[i]['native']['step_scores'] for i in ix],
            PURE_NATIVE=[pure[i]['variants']['native_raw']['step_scores'] for i in ix],
            PURE_RL_PROMPT=[pure[i]['variants']['official_rl_prompt']['step_scores'] for i in ix])
        for arm in ARMS:
            task = task_rewards('P_VR' if arm=='P_VR' else 'P_PRM',lengths,outcomes,prob_sets[arm],positions)
            if arm != 'P_VR':
                # Independent signed-logit task construction before the common RLOO oracle.
                double_task = torch.zeros_like(task, dtype=torch.float64)
                for j, i in enumerate(ix):
                    signed = [2*x-1 for x in qwen[i]['native']['step_scores']] if arm=='QWEN_PRM' else pure[i]['variants'][
                        'native_raw' if arm=='PURE_NATIVE' else 'official_rl_prompt']['signed_scores']
                    r = torch.tensor(signed,dtype=torch.float64)
                    double_task[j,positions[j]] = .5*torch.softmax(-r/.1,dim=0)*r
                torch.testing.assert_close(task.double(),double_task,atol=1e-5,rtol=1e-5)
            raw, white, returns = checked_signals(task,mask,group_ids)
            oracle_checks += 1
            for j,i in enumerate(ix):
                n = lengths[j]
                value = metrics[i]['variants'][arm]
                assert abs(float(task[j].sum())-value['task_reward']) < 2e-6
                value.update(raw_RLOO_terminal_advantage=float(raw[j,n-1]),
                    whitened_terminal_advantage=float(white[j,n-1]),
                    mean_raw_token_advantage=float(raw[j,:n].mean()),
                    mean_whitened_token_advantage=float(white[j,:n].mean()),
                    positive_whitened_tokens=int((white[j,:n]>EPS).sum()),
                    absolute_whitened_token_sum=float(white[j,:n].abs().sum()),
                    terminal_return=float(returns[j,n-1]),whitening_calls=1)
                metrics[i]['advantage_available'] = True
                metrics[i]['reconstruction_length_mismatches_in_batch'] = sum(not qwen[k]['logged_length_matches'] for k in ix)
    assert len(usable_batches) == 8 and len(skipped) == 1
    assert skipped[0]['batch_id'] == 'B:210', 'Unexpected exclusion; do not silently change prior replay population'
    empty_cases = [m for m in metrics if m['empty']]
    available = [m for m in metrics if m['advantage_available']]
    replay_empty = [m for m in empty_cases if m['advantage_available']]
    assert len(available)==256 and len(replay_empty)==8
    empty_scoring, empty_advantage, concordances = {}, {}, {}
    for arm in ARMS:
        values = [m['variants'][arm] for m in replay_empty]
        empty_advantage[arm] = dict(n=len(values),
            positive_raw_terminal=sum(v['raw_RLOO_terminal_advantage']>EPS for v in values),
            positive_whitened_terminal=sum(v['whitened_terminal_advantage']>EPS for v in values),
            rank_first=sum(v['rank_in_question_by_task_reward']==1 for v in values),
            correct_peer_count=sum(v['correct_peers'] for v in values),
            outranked_correct_peer_count=sum(v['higher_than_correct_peers'] for v in values),
            min_whitened_terminal=min(v['whitened_terminal_advantage'] for v in values),
            max_whitened_terminal=max(v['whitened_terminal_advantage'] for v in values),
            length_matched_batch_subset=dict(n=sum(m['reconstruction_length_mismatches_in_batch']==0 for m in replay_empty),
                positive_whitened=sum(m['variants'][arm]['whitened_terminal_advantage']>EPS for m in replay_empty
                    if m['reconstruction_length_mismatches_in_batch']==0)))
        if arm != 'P_VR':
            ps = [m['variants'][arm]['positive_probability'] for m in empty_cases]
            empty_scoring[arm] = dict(n=len(ps),signed_positive=sum(p>.5+EPS for p in ps),
                positive_probability_min=min(ps),positive_probability_max=max(ps),
                mean_positive_probability=mean(ps),
                reward_rank_first=sum(m['variants'][arm]['rank_in_question_by_task_reward']==1 for m in empty_cases))
        values = [compare_outcomes(ix,metrics,arm) for ix in groups.values()]
        total = {k:sum(v[k] for v in values) for k in ['correct_incorrect_pairs','correct_higher','incorrect_higher','tied']}
        total['concordance'] = (total['correct_higher']+.5*total['tied'])/total['correct_incorrect_pairs'] if total['correct_incorrect_pairs'] else None
        concordances[arm] = total
    group_comparisons, signal_comparisons = {}, {}
    for left,right in [('QWEN_PRM','PURE_NATIVE'),('QWEN_PRM','PURE_RL_PROMPT'),('PURE_NATIVE','PURE_RL_PROMPT')]:
        key = left+'__vs__'+right
        entries = []
        for (batch_id,pid),ix in groups.items():
            rho = spearman([metrics[i]['variants'][left]['task_reward'] for i in ix],
                           [metrics[i]['variants'][right]['task_reward'] for i in ix])
            entries.append(dict(batch_id=batch_id,problem_id=pid,group_type=metrics[ix[0]]['group_type'],spearman=rho))
        group_comparisons[key] = dict(groups=len(entries),defined_groups=sum(e['spearman'] is not None for e in entries),
            mean_group_spearman=mean([e['spearman'] for e in entries if e['spearman'] is not None]),groups_detail=entries)
        strata = {}
        for kind in ('all','mixed','all_correct','all_wrong'):
            subset = [m for m in available if kind=='all' or m['group_type']==kind]
            strata[kind] = dict(n=len(subset),
                raw_terminal_sign_changes=sum(sign(m['variants'][left]['raw_RLOO_terminal_advantage']) != sign(m['variants'][right]['raw_RLOO_terminal_advantage']) for m in subset),
                white_terminal_sign_changes=sum(sign(m['variants'][left]['whitened_terminal_advantage']) != sign(m['variants'][right]['whitened_terminal_advantage']) for m in subset),
                mean_token_advantage_sign_changes=sum(sign(m['variants'][left]['mean_whitened_token_advantage']) != sign(m['variants'][right]['mean_whitened_token_advantage']) for m in subset))
        signal_comparisons[key] = strata
    group_strata = {}
    for kind in ('mixed','all_correct','all_wrong'):
        subset = [m for m in available if m['group_type']==kind]
        group_strata[kind] = dict(rows=len(subset),groups=len(subset)//8,
            variants={arm:dict(positive_terminal=sum(m['variants'][arm]['whitened_terminal_advantage']>EPS for m in subset),
                mean_absolute_token_advantage=sum(m['variants'][arm]['absolute_whitened_token_sum'] for m in subset)/sum(m['reconstructed_length'] for m in subset) if subset else None)
                for arm in ARMS})
    summary = dict(passed=True, scored_rows=288,scored_empty_rows=9,
        replayed_rows=256,replayed_empty_rows=8,replayed_batches=usable_batches,skipped_batches=skipped,
        provenance=dict(inputs=str(args.inputs),input_sha256=sha(args.inputs),qwen_scored_sha256=sha(args.qwen_scores),
            pure_scored_sha256=sha(pure_path),evidence_sha256=sha(evidence_path),
            reconstruction_length_mismatches=sum(not q['logged_length_matches'] for q in qwen)),
        scoring=dict(empty=empty_scoring,group_comparisons=group_comparisons,
            outcome_concordance=concordances,all_group_types=dict(collections.Counter(metrics[ix[0]]['group_type'] for ix in groups.values()))),
        advantage=dict(empty=empty_advantage,comparisons=signal_comparisons,group_strata=group_strata,
            task_coefficient=.5,softmin_temperature=.1,response_width=2048,kl=0.,whitening_calls_per_batch=1,
            oracle_checks=oracle_checks,independent_signal_oracle_passed=True),
        costs=summarize_costs(evidence),remote_evidence=evidence,
        limitations=[
            '288 rows selected as nine full historical batches containing empty answers; not representative sampling.',
            'All raw scores retained. Whole B:210 batch excluded only from token-advantage replay because one reconstructed response exceeds fixed width.',
            'Old logs did not save emitted token IDs; reconstruction with logged-length agreement still does not prove exact original tokens.',
            'KL fixed to zero; this is an offline counterfactual, not historical A/B advantages or a prediction of future training.',
            'Native-input formats differ by PRM. Raw-question and official-RL-prompt variants use shared canonical steps, not the official literal token splitter.',
            'Final-answer concordance is not process correctness; there are no new human step labels.',
            'No winner selected, no training launched, no reward or answer-verification rules changed.'],
        training_started=False,new_generations=0,optimizer_steps=0)
    out.mkdir(parents=True,exist_ok=True)
    atomic(out/'summary.json',summary)
    atomic(out/'per_empty.json',empty_cases)
    write_jsonl(out/'per_response_metrics.jsonl',metrics)
    render(summary,empty_cases,out)
    print(json.dumps(dict(scoring_empty=empty_scoring,advantage_empty=empty_advantage,
        replayed_batches=len(usable_batches),oracle_checks=oracle_checks),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
