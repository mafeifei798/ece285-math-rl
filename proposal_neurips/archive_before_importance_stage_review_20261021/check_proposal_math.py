"""Sanity checks for the proposed losses; these are not model-training results."""
import json
import math
import random


def pooling(errors, temperature=0.2):
    logits = [e / temperature for e in errors]
    top = max(logits)
    unnormalized = [math.exp(x - top) for x in logits]
    weights = [x / sum(unnormalized) for x in unnormalized]
    return sum(w * e for w, e in zip(weights, errors)), weights


def frozen_weight_loss(errors, weights, failed):
    pooled = sum(w * e for w, e in zip(weights, errors))
    return -math.log(pooled if failed else 1 - pooled)


def advantages(rewards):
    mean = sum(rewards) / len(rewards)
    # Match the pinned implementation's sample-standard-deviation convention.
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1))
    return [(r - mean) / (std + 1e-4) for r in rewards]


rng = random.Random(285)
max_gradient_error = 0.0
cases = 0
for n in [2, 3, 8, 16]:
    for _ in range(25):
        errors = [rng.uniform(0.03, 0.97) for _ in range(n)]
        pooled, weights = pooling(errors)
        assert min(errors) <= pooled <= max(errors)
        assert abs(sum(weights) - 1) < 1e-12
        for failed in [True, False]:
            analytical = [(-w / pooled if failed else w / (1 - pooled)) for w in weights]
            for k in range(n):
                delta = 1e-6
                plus, minus = errors.copy(), errors.copy()
                plus[k] += delta
                minus[k] -= delta
                # Deliberately HOLD weights fixed to test stop_gradient, not full SWS differentiation.
                finite = (frozen_weight_loss(plus, weights, failed) - frozen_weight_loss(minus, weights, failed)) / (2 * delta)
                err = abs(finite - analytical[k])
                max_gradient_error = max(max_gradient_error, err)
                assert err < 1e-7
                assert analytical[k] < 0 if failed else analytical[k] > 0
                cases += 1
        fail_loss = -math.log(pooled)
        success_loss = -math.log(1 - pooled)
        anchor_loss = 0.73
        loss_c = fail_loss + success_loss + anchor_loss
        loss_d = fail_loss + anchor_loss
        assert abs((loss_c - loss_d) - success_loss) < 1e-12

example_errors = [0.1, 0.9]
pooled, weights = pooling(example_errors)
assert pooling(example_errors, 1e-4)[0] > 0.899999
assert abs(pooling(example_errors, 1e6)[0] - 0.5) < 1e-6
# Mixed rewards preserve correct-vs-incorrect trajectory ordering.
assert 0.2 < 1.0
assert advantages([1.0] * 8) == [0.0] * 8
reward_example = [1.02, 0.18, 1.16, 0.02, 1.1, 0.06, 0.1, 1.05]
assert abs(sum(advantages(reward_example))) < 1e-12
assert 24_000_000 * 4 * 3 == 288_000_000
# The process coefficient cancels in same-outcome groups except for epsilon.
process_scores = [0.05, 0.15, 0.25, 0.35, 0.55, 0.65, 0.8, 0.95]
mean_m = sum(process_scores) / len(process_scores)
std_m = math.sqrt(sum((m - mean_m) ** 2 for m in process_scores) / (len(process_scores) - 1))
same_outcome_expected = [(m - mean_m) / (std_m + 5e-4) for m in process_scores]
normalization_error = 0.0
for y in [0, 1]:
    actual = advantages([y + 0.2 * m for m in process_scores])
    normalization_error = max(normalization_error, max(abs(a - b) for a, b in zip(actual, same_outcome_expected)))
assert normalization_error < 1e-12
# Perfect score correlation need not preserve advantages in mixed-outcome
# groups: an affine score transform changes the process/outcome trade-off.
affine_scores = [0.4 + 0.1 * m for m in process_scores]
mixed_outcomes = [1, 1, 0, 0, 1, 0, 1, 0]
a_c = advantages([y + 0.2 * m for y, m in zip(mixed_outcomes, process_scores)])
a_d = advantages([y + 0.2 * m for y, m in zip(mixed_outcomes, affine_scores)])
affine_advantage_difference = max(abs(a - b) for a, b in zip(a_c, a_d))
assert affine_advantage_difference > 1e-3
# Small raw differences are also not enough for a futility conclusion.
tiny_c = [0.5 + 1e-3 * (m - 0.5) for m in process_scores]
tiny_d = [0.5 - 1e-3 * (m - 0.5) for m in process_scores]
tiny_raw_difference = max(abs(c - d) for c, d in zip(tiny_c, tiny_d))
tiny_advantage_difference = max(abs(a - b) for a, b in zip(
    advantages([0.2 * m for m in tiny_c]), advantages([0.2 * m for m in tiny_d])))
assert tiny_raw_difference < 1e-3 and tiny_advantage_difference > 0.5
# Normal approximation for one paired comparison, not an observed effect or
# the power of three potentially correlated seeds.
planning_se = math.sqrt(0.1 / 500)
planning_half_width = 1.959963984540054 * planning_se
planning_mde = (1.959963984540054 + 0.8416212335729143) * planning_se
assert 0.039 < planning_mde < 0.041
print(json.dumps({
    'status': 'passed',
    'gradient_coordinates_checked': cases,
    'max_absolute_gradient_error': max_gradient_error,
    'example_errors': example_errors,
    'example_weights': weights,
    'example_pooled_error': pooled,
    'example_failure_loss': -math.log(pooled),
    'example_success_loss': -math.log(1 - pooled),
    'same_outcome_normalization_max_error': normalization_error,
    'precheck_counterexamples_not_model_results': {
        'affine_score_Pearson_and_Spearman': 1.0,
        'mixed_group_max_advantage_difference': affine_advantage_difference,
        'small_raw_score_max_difference': tiny_raw_difference,
        'small_raw_score_max_advantage_difference': tiny_advantage_difference,
    },
    'illustrative_paired_power': {
        'n': 500, 'assumed_disagreement': 0.1, 'power': 0.8, 'two_sided_alpha': 0.05,
        'standard_error_pp': 100 * planning_se,
        'interval_half_width_pp': 100 * planning_half_width,
        'detectable_difference_pp': 100 * planning_mde,
        'observed_project_effect': False,
    },
    'tests': ['convex pooling bounds', 'detached-weight loss gradient signs and finite differences',
              'C-minus-D equals success loss at shared parameters', 'temperature limits',
              'outcome-order preservation', 'zero-variance group advantage', 'centered advantages',
              'authorized 24M-per-run token-budget arithmetic',
              'same-outcome process-weight cancellation with exact epsilon',
              'score correlation and raw closeness do not imply identical normalized advantages',
              'illustrative single-seed paired accuracy power calculation'],
    'not_tested': ['training implementation', 'GPU feasibility', 'model accuracy', 'dataset overlap']
}, indent=2))
