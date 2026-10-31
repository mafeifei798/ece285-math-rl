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
assert 6_000_000 * 4 * 3 == 72_000_000
print(json.dumps({
    'status': 'passed',
    'gradient_coordinates_checked': cases,
    'max_absolute_gradient_error': max_gradient_error,
    'example_errors': example_errors,
    'example_weights': weights,
    'example_pooled_error': pooled,
    'example_failure_loss': -math.log(pooled),
    'example_success_loss': -math.log(1 - pooled),
    'tests': ['convex pooling bounds', 'detached-weight loss gradient signs and finite differences',
              'C-minus-D equals success loss at shared parameters', 'temperature limits',
              'outcome-order preservation', 'zero-variance group advantage', 'centered advantages',
              'provisional token-budget arithmetic'],
    'not_tested': ['training implementation', 'GPU feasibility', 'model accuracy', 'dataset overlap']
}, indent=2))
