"""Offline reward counterfactuals, never a training reward implementation."""
import collections
import hashlib
import json
from pathlib import Path
import random
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'execution/pure_empty_boundary_20261022'
sys.path.insert(0, str(ROOT / 'experiments/pure_integration_20261022'))
from common import atomic, read, read_jsonl, sha
from signals import task_rewards
from vendor import pure_core as official
from offline_check import independent_reference


def rloo(task, mask, groups):
    raw, ret = official.compute_rloo_outcome_advantage(task, mask, groups, adv_norm=False)
    adv, _ = official.compute_rloo_outcome_advantage(task, mask, groups, adv_norm=True)
    independent_adv, independent_ret = independent_reference(task, mask, groups)
    torch.testing.assert_close(ret, independent_ret, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(adv, independent_adv, atol=5e-6, rtol=5e-6)
    return raw, adv, ret


def main():
    torch.set_num_threads(2)
    inputs = read_jsonl(OUT / 'inputs.jsonl')
    results = read_jsonl(OUT / 'remote/frozen_scoring_v2/scored.jsonl')
    evidence = read(OUT / 'remote/frozen_scoring_v2/result.json')
    assert len(inputs) == len(results) == 288 and evidence['passed']
    assert evidence['input_sha256'] == sha(OUT / 'inputs.jsonl')
    train = read_jsonl(ROOT / 'execution/ab_corrected_6m_seed17_20261022/final_evidence/raw/data/prepared/train.jsonl')
    ids = sorted(r['problem_id'] for r in train)
    gt_ids = set(random.Random(17).sample(ids, len(ids)//10))
    atomic(OUT / 'offline_gt10_manifest.json', dict(selection='seed17 sample of sorted full train IDs, not selected using outcomes',
        count=len(gt_ids), total=len(ids), fraction=len(gt_ids)/len(ids),
        problem_ids=sorted(gt_ids), training_use=False))
    batches = collections.defaultdict(list)
    for row, score in zip(inputs, results):
        assert score['input_sha256'] == hashlib.sha256(row['completion'].encode()).hexdigest()
        batches[row['batch_id']].append((row, score))
    empty_cases, skipped = [], []
    for batch_id, pairs in batches.items():
        assert len(pairs) == 32
        if any('alignment_error' in s for r, s in pairs):
            skipped.append(dict(batch_id=batch_id, reason='reconstructed token alignment unavailable'))
            continue
        rows = [r for r, s in pairs]; scores = [s for r, s in pairs]
        lengths = [len(s['reconstructed_ids']) for s in scores]
        positions = [s['alignment']['actor_positions'] for s in scores]
        probabilities = [s['native']['step_scores'] for s in scores]
        outcomes = [r['outcome'] for r in rows]
        groups = [r['problem_id'] for r in rows]
        assert sorted(collections.Counter(groups).values()) == [8,8,8,8]
        mask = (torch.arange(2048)[None,:] < torch.tensor(lengths)[:,None]).float()
        process = task_rewards('P_PRM', lengths, outcomes, probabilities, positions)
        vr = task_rewards('P_VR', lengths, outcomes, probabilities, positions)
        zero = process.clone()
        empty = [i for i, r in enumerate(rows) if not r['completion'].strip()]
        zero[empty] = 0  # COUNTERFACTUAL ONLY: intentionally NOT installed.
        availability = torch.tensor([r['problem_id'] in gt_ids for r in rows])[:,None]
        variants = dict(P_VR=vr, P_PRM=process,
            ZERO_EMPTY_NOT_ADOPTED=zero,
            OFFICIAL_MIX_GT_AVAILABLE=process + vr,
            OFFICIAL_MIX_FIXED_10PCT=process + vr * availability)
        signals = {k:rloo(v, mask, groups) for k,v in variants.items()}
        for i in empty:
            row, score = pairs[i]
            peers = [j for j in range(32) if groups[j] == groups[i] and j != i]
            metrics = {}
            for name, task in variants.items():
                raw, adv, ret = signals[name]
                total = task.sum(-1)
                n = lengths[i]
                metrics[name] = dict(task_reward=float(total[i]),
                    raw_RLOO_terminal_advantage=float(raw[i,n-1]),
                    whitened_terminal_advantage=float(adv[i,n-1]),
                    rewards_of_correct_peers=[float(total[j]) for j in peers if outcomes[j]],
                    higher_than_correct_peers=sum(float(total[i]) > float(total[j])+1e-7 for j in peers if outcomes[j]),
                    rank_in_question_by_task_reward=1+sum(float(total[j]) > float(total[i])+1e-7 for j in peers))
            empty_cases.append(dict(source_arm=row['source_arm'], source_line=row['source_line'],
                batch_id=batch_id, problem_id=row['problem_id'], raw=row['completion'],
                old_logged_reward=row['old_reward'], old_PRM_was_not_called=(not row['old_prm_scores']),
                native_positive_probability=probabilities[i][0], direct_probability=score['direct_official_positive_probabilities'][0],
                correct_peers=sum(outcomes[j] for j in peers), ground_truth_selected_10pct=groups[i] in gt_ids,
                reconstruction_length_mismatches_in_batch=sum(not s['logged_length_matches'] for s in scores),
                variants=metrics))
    aggregate = {}
    for name in ['P_VR','P_PRM','ZERO_EMPTY_NOT_ADOPTED','OFFICIAL_MIX_GT_AVAILABLE','OFFICIAL_MIX_FIXED_10PCT']:
        values = [r['variants'][name] for r in empty_cases]
        aggregate[name] = dict(n=len(values),
            positive_raw_advantage=sum(r['raw_RLOO_terminal_advantage'] > 1e-7 for r in values),
            positive_whitened_advantage=sum(r['whitened_terminal_advantage'] > 1e-7 for r in values),
            reward_rank_1=sum(r['rank_in_question_by_task_reward']==1 for r in values),
            min_whitened_advantage=min(r['whitened_terminal_advantage'] for r in values),
            max_whitened_advantage=max(r['whitened_terminal_advantage'] for r in values),
            length_matched_batch_subset=dict(
                n=sum(r['reconstruction_length_mismatches_in_batch']==0 for r in empty_cases),
                positive_whitened_advantage=sum(r['variants'][name]['whitened_terminal_advantage']>1e-7
                    for r in empty_cases if r['reconstruction_length_mismatches_in_batch']==0)))
    empty_scores = [s['native']['step_scores'][0] for r,s in zip(inputs,results) if not r['completion'].strip()]
    summary = dict(passed=True, observed_scoring=evidence, selection=read(OUT/'selection.json'),
        scored_historical_empty_responses=9, native_empty_score_range=[min(empty_scores),max(empty_scores)],
        reconstructed_batches_replayed=len(batches)-len(skipped),
        skipped_batches=skipped, empty_cases=empty_cases, aggregate=aggregate,
        counterfactual_scope='PURE numerical signal on saved A/B responses, reconstructed IDs, KL set to zero; not their historical training advantage or a prediction of a new run',
        GT_available_variant='Official code combination applied to every displayed question; illustrates the GT-present branch, NOT the paper 10 percent regime',
        GT10_variant='Fixed 699/6993 training-question selection, all eight responses share eligibility; no coefficient of 0.1 on every response',
        GT10_empty_cases_with_label=sum(r['ground_truth_selected_10pct'] for r in empty_cases),
        independent_float64_signal_oracle_passed=True,
        no_reward_variant_installed=True, training_started=False)
    atomic(OUT / 'offline_analysis.json', summary)
    changed=[]
    for rel, expected in read(OUT/'protected_before.json').items():
        if sha(ROOT/rel)!=expected:changed.append(rel)
    assert not changed, changed
    atomic(OUT / 'preservation.json', dict(passed=True, protected_files=len(read(OUT/'protected_before.json')),
        changed_files=changed, source_inputs_unchanged=True))
    print(json.dumps(dict(aggregate=aggregate, skipped=skipped,
        native_empty_scores=[r['native_positive_probability'] for r in empty_cases]),indent=2))


if __name__ == '__main__':
    main()
