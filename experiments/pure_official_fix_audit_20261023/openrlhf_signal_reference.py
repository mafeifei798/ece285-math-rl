"""Pinned OpenRLHF CPU signal reference on existing canonical trajectories.

No model forward, generation, optimizer, GPU access, or training launcher.
Official AST functions construct soft-min rewards and process_experiences; the
boundary_adapter executes official normalization once. This is not end-to-end
OpenRLHF replication: old emitted IDs were not saved, token positions and steps
are reconstructed, input serialization is fixed from the previous scoring run,
and KL is zero. No verifier or response is changed.
"""
from __future__ import annotations

import ast
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from boundary_adapter import apply_openrlhf_boundary_norm, SOURCE_COMMIT


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'execution/pure_official_fix_audit_20261023'
OFFICIAL = OUT / 'source_openrlhf/openrlhf/trainer/ppo_utils/experience_maker.py'
HELPERS = OUT / 'source_openrlhf_extra/openrlhf/utils/utils.py'
HASHES = {
    OFFICIAL: '05ae7184e2223ffeb3a329b05a551e884967264be76b5d7d756acad5409e30cb',
    HELPERS: 'c5de92d748498b20ecf673bab91c92c71e26dcc3c0dce180ce7c503f983b2924',
}
ARMS = {'OPENRLHF_VR': None, 'OPENRLHF_PURE_NATIVE': 'native_raw',
        'OPENRLHF_PURE_RL_PROMPT': 'official_rl_prompt'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def official_functions():
    namespace = {'torch': torch}
    metadata = []
    for path, class_name, names in (
        (HELPERS, None, ['right_padding', 'turn_process_reward_logits_to_reward']),
        (OFFICIAL, 'NaiveExperienceMaker', ['process_experiences']),
    ):
        if digest(path) != HASHES[path]:
            raise ValueError(f'Official source hash mismatch: {path}')
        tree = ast.parse(path.read_text(), filename=str(path))
        body = tree.body
        if class_name:
            classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name]
            assert len(classes) == 1
            body = classes[0].body
        for name in names:
            functions = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
            assert len(functions) == 1
            function = copy.deepcopy(functions[0])
            future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
            module = ast.Module(body=[future, function], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
            metadata.append(dict(path=str(path), sha256=HASHES[path], function=name,
                                 class_name=class_name, start_line=function.lineno,
                                 end_line=function.end_lineno))
    return namespace, metadata


def build_experience(inputs, aligned, scored, variant, official):
    batch_size = len(inputs)
    lengths = [len(row['reconstructed_ids']) for row in aligned]
    positions = [row['alignment']['actor_positions'] for row in aligned]
    width = max(lengths)
    max_steps = max(map(len, positions))
    valid = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
    boundary = torch.zeros((batch_size, width), dtype=torch.bool)
    score_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    step_lengths = torch.zeros((batch_size, max_steps), dtype=torch.long)
    logits = torch.zeros((batch_size, max_steps, 2), dtype=torch.float32)
    for i, ends in enumerate(positions):
        assert ends == sorted(set(ends)) and ends[-1] == lengths[i] - 1
        boundary[i, ends] = True
        score_mask[i, :len(ends)] = True
        previous_end = -1
        for j, end in enumerate(ends):
            # Literal make_experience lengths: first score_id-1; then
            # score_id-current_previous_score_id-1. Separators/EOS excluded.
            step_lengths[i, j] = end - previous_end - 1
            previous_end = end
        if variant:
            saved = scored[i]['variants'][variant]
            assert len(saved['step_logits']) == len(ends)
            logits[i, :len(ends)] = torch.tensor(saved['step_logits'], dtype=torch.float32)
    task_rewards = torch.zeros((batch_size, width), dtype=torch.float32)
    if variant:
        weighted = official['turn_process_reward_logits_to_reward'](
            logits, score_mask, return_outcome_reward=False,
            temperature=0.1, disable_weighted_reward=False)
        task_rewards[boundary] = weighted[score_mask]
    else:
        weighted = torch.zeros_like(step_lengths, dtype=torch.float32)
    info = dict(process_reward=task_rewards, process_reward_mask=boundary,
                num_steps=score_mask.sum(1), response_length=valid.float().sum(1),
                num_tokens_per_step=step_lengths, score_mask=score_mask,
                solved=torch.tensor([row['outcome'] for row in inputs], dtype=torch.float32),
                have_answers=torch.ones(batch_size, dtype=torch.bool))
    return SimpleNamespace(info=info, action_mask=valid), dict(
        lengths=lengths, positions=positions, step_lengths=step_lengths,
        weighted=weighted, logits=logits, valid=valid, boundary=boundary)


def independent_float64(inputs, aligned, scored, variant):
    """Scalar oracle, independently computes soft-min, baseline and raw returns."""
    lengths = [len(row['reconstructed_ids']) for row in aligned]
    positions = [row['alignment']['actor_positions'] for row in aligned]
    weighted = []
    for i, ends in enumerate(positions):
        if variant:
            logits = scored[i]['variants'][variant]['step_logits']
            signed = []
            for pair in logits:
                z0, z1 = [math.exp(float(x) - max(pair)) for x in pair]
                signed.append((z1 - z0) / (z1 + z0))
            exponents = [-r / .1 for r in signed]
            weights = [math.exp(x - max(exponents)) for x in exponents]
            denominator = math.fsum(weights)
            weighted.append([r * w / denominator for r, w in zip(signed, weights)])
        else:
            weighted.append([0.] * len(ends))
    point_rewards = []
    raw_returns = []
    baselines = []
    for i, ends in enumerate(positions):
        peers = [j for j in range(len(inputs))
                 if j != i and inputs[j]['problem_id'] == inputs[i]['problem_id']]
        assert len(peers) == 7
        baseline = math.fsum(math.fsum(weighted[j]) for j in peers) / sum(lengths[j] for j in peers)
        baselines.append(baseline)
        sparse = [0.] * lengths[i]
        previous_end = -1
        for pos, reward in zip(ends, weighted[i]):
            sparse[pos] = reward - baseline * (pos - previous_end - 1)
            previous_end = pos
        if variant is None:
            own_vr = float(inputs[i]['outcome']) - 1.
            peer_vr = math.fsum(float(inputs[j]['outcome']) - 1. for j in peers) / 7.
            sparse[-1] += own_vr - peer_vr
        point_rewards.append(sparse)
        raw_returns.append([math.fsum(sparse[t:]) for t in range(lengths[i])])
    return weighted, point_rewards, raw_returns, baselines


@torch.no_grad()
def replay_batch(inputs, aligned, scored, arm, official):
    variant = ARMS[arm]
    batch_size = len(inputs)
    assert batch_size % 8 == 0
    for start in range(0, batch_size, 8):
        assert len({row['problem_id'] for row in inputs[start:start + 8]}) == 1
    experience, pack = build_experience(inputs, aligned, scored, variant, official)
    args = SimpleNamespace(advantage_estimator='rloo', n_samples_per_prompt=8,
                           micro_rollout_batch_size=batch_size, reward_baseline='token',
                           reward_mode='VR' if variant is None else 'PRM',
                           verifiable_reward_coef=1.0)
    maker = SimpleNamespace(strategy=SimpleNamespace(args=args))
    _, point_rewards = official['process_experiences'](maker, [experience])
    assert len(point_rewards) == 1 and point_rewards[0].shape == experience.action_mask.shape
    raw = point_rewards[0].fliplr().cumsum(dim=1).fliplr()
    # KL=0 counterfactual. No veRL whitening before the boundary adapter.
    normalized, norm_receipt = apply_openrlhf_boundary_norm(raw, pack['valid'], pack['boundary'])
    oracle_weighted, oracle_points, oracle_raw, baselines = independent_float64(inputs, aligned, scored, variant)
    point_error = raw_error = weighted_error = 0.
    for i, length in enumerate(pack['lengths']):
        point_error = max(point_error, max(abs(float(a) - b) for a, b in zip(point_rewards[0][i, :length], oracle_points[i])))
        raw_error = max(raw_error, max(abs(float(a) - b) for a, b in zip(raw[i, :length], oracle_raw[i])))
        weighted_error = max(weighted_error, max(abs(float(a) - b) for a, b in zip(pack['weighted'][i], oracle_weighted[i])))
    assert max(point_error, raw_error, weighted_error) < 2e-5, (point_error, raw_error, weighted_error)
    metrics = []
    for i, length in enumerate(pack['lengths']):
        nonboundary = pack['valid'][i] & ~pack['boundary'][i]
        content_norm = normalized[i][nonboundary]
        metrics.append(dict(
            task_reward_sum=float(pack['weighted'][i].sum()),
            process_peer_actual_length_baseline=baselines[i],
            raw_terminal_before_boundary_mask=float(raw[i, length - 1]),
            raw_first_before_boundary_mask=float(raw[i, 0]),
            normalized_terminal=float(normalized[i, length - 1]),
            normalized_first=float(normalized[i, 0]),
            content_mean_normalized=float(content_norm.mean()) if content_norm.numel() else None,
            content_positive_tokens=int((content_norm > 0).sum()),
            content_negative_tokens=int((content_norm < 0).sum()),
            content_tokens=int(nonboundary.sum()),
            step_positions=pack['positions'][i],
            num_tokens_per_step_excluding_boundaries=pack['step_lengths'][i, :len(pack['positions'][i])].tolist(),
            boundary_normalized_all_zero=bool(torch.all(normalized[i, pack['boundary'][i]] == 0)),
            response_tokens=length,
        ))
    receipt = dict(arm=arm, reward_mode=args.reward_mode,
                   process_reward_coefficient=1.0, softmin_temperature=.1, verifiable_reward_coefficient=1.,
                   reward_baseline='token_actual_peer_response_lengths', group_size=8, kl_coefficient=0.,
                   verifier_mapping='solved - 1 with sequence-level leave-one-out baseline' if variant is None else 'unused in PRM-only mode',
                   oracle_weighted_max_abs_error=weighted_error,
                   oracle_point_reward_max_abs_error=point_error, oracle_raw_advantage_max_abs_error=raw_error,
                   normalization=norm_receipt)
    return metrics, receipt


def self_test(official):
    # One EOS-only empty and seven short valid trajectories; same group.
    inputs = [dict(problem_id='synthetic', outcome=float(i % 2), completion='' if i == 0 else 'x') for i in range(8)]
    aligned = [dict(reconstructed_ids=[1] if i == 0 else [2, 3, 1],
                    alignment=dict(actor_positions=[0] if i == 0 else [2])) for i in range(8)]
    scored = [dict(variants={name:dict(step_logits=[[0., 2. if i == 0 else (-1. if i % 2 else 1.)]])
                            for name in ('native_raw', 'official_rl_prompt')}) for i in range(8)]
    results = {}
    for arm in ARMS:
        metrics, receipt = replay_batch(inputs, aligned, scored, arm, official)
        assert metrics[0]['normalized_terminal'] == 0.
        assert metrics[0]['content_tokens'] == 0
        assert all(m['boundary_normalized_all_zero'] for m in metrics)
        assert receipt['normalization']['normalization_calls'] == 1
        results[arm] = dict(empty=metrics[0], oracle_raw_error=receipt['oracle_raw_advantage_max_abs_error'])
    assert results['OPENRLHF_PURE_NATIVE']['empty']['raw_terminal_before_boundary_mask'] > 0
    return dict(passed=True, synthetic_only=True, results=results)


def main():
    torch.set_num_threads(1)
    targets = {name:OUT / name for name in ('openrlhf_signal_reference_summary.json',
                                           'openrlhf_signal_reference_rows.jsonl',
                                           'openrlhf_signal_reference_receipts.json',
                                           'openrlhf_signal_reference_tests.json')}
    if any(path.exists() for path in targets.values()):
        raise FileExistsError('Reference output already exists; refusing to overwrite')
    official, source_metadata = official_functions()
    tests = self_test(official)
    old = ROOT / 'execution/pure_official_prm_offline_20261022'
    paths = dict(inputs=ROOT / 'execution/pure_empty_boundary_20261022/inputs.jsonl',
                 aligned=old / 'remote/qwen_single/scored.jsonl',
                 scored=old / 'remote/scoring_single/scored.jsonl')
    before = {key:digest(path) for key, path in paths.items()}
    inputs, aligned, scored = [jsonl(paths[key]) for key in ('inputs', 'aligned', 'scored')]
    assert len(inputs) == len(aligned) == len(scored) == 288
    batches = collections.defaultdict(list)
    for i, row in enumerate(inputs):
        assert aligned[i]['index'] == scored[i]['index'] == i
        assert aligned[i]['input_sha256'] == scored[i]['input_sha256']
        batches[row['batch_id']].append(i)
    rows = {}
    receipts = []
    skipped = []
    for batch_id, indices in batches.items():
        if any('alignment_error' in aligned[i] for i in indices):
            skipped.append(batch_id)
            continue
        assert len(indices) == 32
        for arm in ARMS:
            metrics, receipt = replay_batch([inputs[i] for i in indices], [aligned[i] for i in indices],
                                            [scored[i] for i in indices], arm, official)
            receipt['batch_id'] = batch_id
            receipts.append(receipt)
            for i, values in zip(indices, metrics):
                if i not in rows:
                    rows[i] = dict(index=i, batch_id=batch_id, problem_id=inputs[i]['problem_id'],
                                   source_arm=inputs[i]['source_arm'], empty=not inputs[i]['completion'].strip(),
                                   outcome=inputs[i]['outcome'], variants={})
                rows[i]['variants'][arm] = values
    assert skipped == ['B:210'] and len(rows) == 256 and len(receipts) == 24
    empty = [row for row in rows.values() if row['empty']]
    assert len(empty) == 8
    summary = dict(passed=True, commit=SOURCE_COMMIT,
                   scope='Fixed canonical-input OpenRLHF reward/baseline/advantage CPU reference; not end-to-end reproduction',
                   model_forwards=0, new_generations=0, optimizer_steps=0, gpu_access=False,
                   selected_saved_responses=288, usable_responses=256, batches=8, empty_responses=8,
                   skipped_batches=skipped, raw_token_ids_reconstructed=True, kl_coefficient=0.,
                   source_functions=source_metadata,
                   input_provenance={key:dict(path=str(path),sha256=before[key]) for key,path in paths.items()},
                   script_sha256=digest(Path(__file__)), tests_passed=tests['passed'],
                   aggregate={})
    for arm in ARMS:
        values = [row['variants'][arm] for row in empty]
        summary['aggregate'][arm] = dict(
            raw_empty_positive=sum(v['raw_terminal_before_boundary_mask'] > 0 for v in values),
            raw_empty_range=[min(v['raw_terminal_before_boundary_mask'] for v in values), max(v['raw_terminal_before_boundary_mask'] for v in values)],
            normalized_empty_positive=sum(v['normalized_terminal'] > 0 for v in values),
            normalized_empty_zero=sum(v['normalized_terminal'] == 0 for v in values),
            normal_terminal_boundaries_zero=sum(row['variants'][arm]['normalized_terminal'] == 0 for row in rows.values()),
            all_boundary_tokens_zero=all(row['variants'][arm]['boundary_normalized_all_zero'] for row in rows.values()),
            max_raw_oracle_error=max(r['oracle_raw_advantage_max_abs_error'] for r in receipts if r['arm'] == arm),
            max_norm_oracle_error=max(r['normalization']['float64_oracle']['max_abs_error'] for r in receipts if r['arm'] == arm))
    summary['limitations'] = [
        'Selected existing A/B rollouts are not representative new PURE rollouts and cannot establish learning or accuracy gains.',
        'Canonical steps, fixed input formatting and reconstructed positions are reused; official native token splitting and PRM forward are not rerun.',
        'All original verifier outcomes remain fixed; VR uses all available answers, PRM-only uses none; this is not the published PRMVR-10% training recipe.',
        'Actual-other-response-length baseline and coefficient one follow this OpenRLHF branch, differing from the pinned veRL branch.',
        'Normal EOS and all supplied boundaries also receive zero direct advantage; empty scores still enter peers baseline.',
        'Nonempty unhelpful responses and shared-parameter changes in EOS probability are not prevented.',
        'No old A/B reward computation or accuracy result is explained causally by this counterfactual.',
    ]
    assert {key:digest(path) for key,path in paths.items()} == before
    for name, value in [('openrlhf_signal_reference_summary.json', summary),
                        ('openrlhf_signal_reference_receipts.json', receipts),
                        ('openrlhf_signal_reference_tests.json', tests)]:
        with targets[name].open('x') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
    with targets['openrlhf_signal_reference_rows.jsonl'].open('x') as handle:
        for index in sorted(rows):
            handle.write(json.dumps(rows[index], ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=True, aggregate=summary['aggregate'], outputs={key:str(value) for key,value in targets.items()}), indent=2))


if __name__ == '__main__':
    main()
