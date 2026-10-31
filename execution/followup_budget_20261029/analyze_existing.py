"""CPU-only evidence for a proposed follow-up; never launches models or training."""
from pathlib import Path
import collections
import hashlib
import json
import statistics
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PREVIOUS = PROJECT / 'execution/final_four_eval_20261029'
PILOT = PREVIOUS / 'evidence/runs/C_bounded_triplet_seed17_1M_20261029_v2'

def read(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

inputs = [PREVIOUS / 'comparison.json']
comparison = read(inputs[0])
analysis = {}
for arm in ['A', 'C_mix', 'C_bounded']:
    rp, mp = PILOT / arm / 'rollouts.jsonl', PILOT / arm / 'metrics.jsonl'
    inputs.extend([rp, mp])
    rr = rows(rp)
    mm = [r for r in rows(mp) if 'kl' in r]
    groups = collections.defaultdict(list)
    for r in rr:
        groups[r['optimizer_step_before_update'], r['problem_id']].append(r)
    assert len(groups) == 192 and all(len(g) == 8 for g in groups.values())
    kinds = collections.Counter(
        'all_wrong' if sum(r['outcome_reward'] for r in g) == 0 else
        'all_correct' if sum(r['outcome_reward'] for r in g) == 8 else 'mixed'
        for g in groups.values()
    )
    item = {
        'optimizer_updates': len(mm),
        'unique_training_questions': len({r['problem_id'] for r in rr}),
        'training_question_fraction': len(groups) / 6993,
        'response_rows': len(rr),
        'group_counts': dict(kinds),
        'process_nonzero_rows': sum(abs(r['process_contribution']) > 1e-8 for r in rr),
        'wrong_positive_advantage_rows': sum(r['outcome_reward'] == 0 and r['advantage'] > 0 for r in rr),
        'KL_first16_mean': statistics.mean(r['kl'] for r in mm[:16]),
        'KL_last16_mean': statistics.mean(r['kl'] for r in mm[-16:]),
        'KL_max': max(r['kl'] for r in mm),
        'grad_norm_median': statistics.median(r['grad_norm'] for r in mm),
        'clip_region_max': max(r['clip_ratio/region_mean'] for r in mm),
    }
    if arm == 'C_bounded':
        item['absolute_process_coefficient_mass'] = sum(abs(r['process_contribution']) for r in rr)
        item['absolute_outcome_coefficient_mass'] = sum(abs(r['outcome_component']) for r in rr)
        item['absolute_process_to_outcome_mass_ratio'] = item['absolute_process_coefficient_mass'] / item['absolute_outcome_coefficient_mass']
        correct = [r for r in rr if r['outcome_reward']]
        item['correct_absolute_process_q50_q90_q99_max'] = np.quantile(
            [abs(r['process_contribution']) for r in correct], [.5, .9, .99, 1]
        ).tolist()
        item['coefficient_caution'] = 'Coefficient mass is not parameter gradient magnitude, causal contribution, or guaranteed probability change.'
    if arm != 'A':
        group_aucs = []
        n_pairs, pair_score = 0, 0.
        for g in groups.values():
            good = [r['min_score'] for r in g if r['valid_for_process_reward'] and r['outcome_reward']]
            bad = [r['min_score'] for r in g if r['valid_for_process_reward'] and not r['outcome_reward']]
            if good and bad:
                values = [float(a > b) + .5 * float(a == b) for a in good for b in bad]
                group_aucs.append(statistics.mean(values))
                n_pairs += len(values)
                pair_score += sum(values)
        item['valid_scored_only_mixed_groups'] = len(group_aucs)
        item['mean_within_group_outcome_rank_AUROC'] = statistics.mean(group_aucs)
        item['pair_weighted_outcome_rank_AUROC'] = pair_score / n_pairs
        item['outcome_ranking_caution'] = 'Association with final-answer correctness; not evidence that it ranks process validity among answer-correct solutions.'
    analysis[arm] = item

historical = PROJECT / 'execution/ab_corrected_6m_seed17_20261022/final_evidence/raw/runs/A_corrected_seed17_6M_20261022'
old_result = historical / 'evaluation/evaluation_summary.json'
old_config = historical / 'run_config.json'
inputs.extend([old_result, old_config])
hist, hist_cfg = read(old_result), read(old_config)
forecast = {
    'proposed_only': True,
    'main_arms': ['A', 'C_mix', 'C_bounded'],
    'training_seeds': [17, 42],
    'existing_seed17_updates': 48,
    'common_intermediate_updates': 144,
    'common_final_updates': 288,
    'questions_per_update': 4,
    'final_question_presentations_per_arm': 1152,
    'estimated_tokens_total_six_runs': sum(comparison['training'][a]['accounting']['generated_completion_tokens'] * 6 * 2 for a in analysis),
    'estimated_additional_tokens': sum(comparison['training'][a]['accounting']['generated_completion_tokens'] * 11 for a in analysis),
    'linear_additional_training_hours_at_observed_rates': sum(comparison['training'][a]['successful_training_seconds_sum'] * 11 for a in analysis) / 3600,
    'scheduled_accounting': 'Equal updates and prompt schedule are primary. 3M/6M are approximate token equivalents, not exact identical token totals.',
    'cost_caution': 'Shared-GPU rates vary. Extrapolation is not a throughput benchmark or completion guarantee.',
}
out = {
    'status': 'analysis_and_proposal_only_no_execution',
    'training_launched': False,
    'new_model_calls': 0,
    'arms': analysis,
    'historical_6M_A': {
        'scores_under_its_recorded_protocol': {k: v['accuracy'] for k, v in hist['datasets'].items()},
        'vllm_memory_utilization': hist_cfg['shared_training_arguments']['vllm_gpu_memory_utilization'],
        'current_training_vllm_memory_utilization': .15,
        'same_protocol_reevaluation_required_before_learning_curve_claim': True,
        'not_a_matched_final_control_for_current_C_arms': True,
    },
    'proposed_followup': forecast,
    'inputs_sha256': {str(p.relative_to(PROJECT)): digest(p) for p in inputs},
}
(HERE / 'analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': out['status'], 'bounded_coefficient_ratio': analysis['C_bounded']['absolute_process_to_outcome_mass_ratio'], 'proposal': forecast}, ensure_ascii=False, indent=2))
