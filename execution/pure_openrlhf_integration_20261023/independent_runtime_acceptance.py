"""CPU-only independent acceptance of one completed no-update probe.

Reads copied evidence, performs no model loading, generation, remote access or
updates, and writes only the requested new JSON review. The independent signal
oracle begins at the saved GPU task matrix, preserving native BF16 arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signal_oracle(tensors, lengths, ends, outcomes, groups, arm, beta):
    task = tensors['task'].double()
    mask = torch.arange(task.shape[1])[None, :] < torch.tensor(lengths)[:, None]
    boundaries = torch.zeros_like(mask)
    reward = task.clone()
    for i, boundaries_i in enumerate(ends):
        boundaries[i, boundaries_i] = True
        peers = [j for j in range(len(lengths)) if j != i and groups[j] == groups[i]]
        assert len(peers) == 7
        peer_rate = task[peers].sum() / sum(lengths[j] for j in peers)
        previous = -1
        for end in boundaries_i:
            reward[i, end] -= peer_rate * (end - previous - 1)
            previous = end
        if arm == 'P_VR':
            reward[i, lengths[i] - 1] += outcomes[i] - math.fsum(outcomes[j] for j in peers) / len(peers)
    kl = (tensors['old_logps'].double() - tensors['ref_logps'].double()) * mask
    shaped = reward - beta * kl
    raw = shaped.flip(1).cumsum(1).flip(1)
    returns = raw.masked_fill(boundaries, 0)
    # The official normalization preserves EVERY initially-zero advantage,
    # excluding them from count but not the variance numerator.
    # Normalize the separately validated stored FP32 returns, so literal zero
    # detection matches the runtime instead of a float64 cancellation artifact.
    values = tensors['returns'].double()[mask]
    zeros = values == 0
    count = int((~zeros).sum())
    assert count > 0
    mean = values.sum() / count
    variance = ((values - mean) ** 2).sum() / count
    normalized = (values - mean) / torch.sqrt(variance.clamp(min=1e-8))
    normalized[zeros] = 0
    advantages = torch.zeros_like(returns)
    advantages[mask] = normalized
    return dict(mask=mask, boundary=boundaries, baseline_reward=reward, kl=kl,
                reward=shaped, raw_returns=raw, returns=returns, advantages=advantages)


def audit(run):
    run = Path(run)
    required = ['result.json', 'status.json', 'run_config.json', 'initialization.json',
                'generated_exact_ids.jsonl', 'verified_rows.jsonl', 'scored_alignment.jsonl',
                'signals.pt', 'P_VR_signal_receipt.json', 'P_PRM_signal_receipt.json',
                'P_VR_loss_audit.json', 'P_PRM_loss_audit.json', 'preservation.json',
                'outcome_independence.json', 'tokenizer_checks.json', 'prm_loading.json',
                'stages.json', 'nvml_peaks.json']
    missing = [name for name in required if not (run / name).is_file()]
    assert not missing, ('Incomplete result evidence', missing)
    result, status = read(run / 'result.json'), read(run / 'status.json')
    assert result['passed'] and status['status'] == 'completed_no_update'
    config, init = read(run / 'run_config.json'), read(run / 'initialization.json')
    assert config['formal_training'] is False and config['optimizer_constructed'] is False
    assert init['optimizer_constructed'] is False and init['backend_original_reward_and_loss_disabled']
    raw_rows, verified, scored = rows(run / 'generated_exact_ids.jsonl'), rows(run / 'verified_rows.jsonl'), rows(run / 'scored_alignment.jsonl')
    assert len(raw_rows) == len(verified) == len(scored) == 32
    eos = init['native_actor_eos']
    lengths, endpoints, outcomes, groups = [], [], [], []
    changed = 0
    boundary_count = 0
    terminal_counts = {'raw_ended_eos': 0, 'official_inserted_eos': 0}
    for i, (raw, row, score) in enumerate(zip(raw_rows, verified, scored)):
        emitted, processed = raw['emitted_completion_ids'], raw['completion_ids']
        assert emitted and processed
        assert raw == {key: row[key] for key in raw}
        assert raw['problem_id'] == score['problem_id'] == config['questions'][i // 8]
        al = raw['alignment']
        assert score['alignment'] == al and score['index'] == i
        assert al['emitted_completion_ids'] == emitted and al['actor_completion_ids'] == processed
        # Accepted generated rows are guaranteed contiguous and unpadded. The
        # pinned method changes only the last token when there is no raw EOS.
        if emitted[-1] == eos:
            assert processed == emitted
            terminal_counts['raw_ended_eos'] += 1
        else:
            assert eos not in emitted
            assert processed == emitted[:-1] + [eos]
            terminal_counts['official_inserted_eos'] += 1
        assert processed[-1] == eos and eos not in processed[:-1]
        assert row['original_terminated'] == (eos in emitted)
        assert al['process_sequences_changed'] == (processed != emitted)
        changed += processed != emitted
        separators = set(al['separator_token_ids'])
        expected_ends = [j for j, token in enumerate(processed) if token in separators] + [len(processed) - 1]
        assert len(set(expected_ends)) == len(expected_ends)
        assert al['actor_boundary_positions'] == expected_ends
        assert al['score_ids'] == [j + 1 for j in expected_ends]
        prm_ids = list(raw['prompt_ids'])
        token_counts = []
        previous = -1
        for end in expected_ends:
            content = processed[previous + 1:end]
            token_counts.append(len(content))
            prm_ids.extend(content)
            prm_ids.append(al['prm_newline_token_id'])
            previous = end
        prm_positions = [len(raw['prompt_ids']) + j for j in expected_ends]
        assert al['prm_input_ids'] == prm_ids
        assert al['prm_score_positions'] == prm_positions
        assert al['num_tokens_per_step'] == token_counts
        assert len(processed) == sum(token_counts) + len(expected_ends)
        assert all(prm_ids[j] == al['prm_newline_token_id'] for j in prm_positions)
        assert score['prm']['score_positions'] == prm_positions
        assert len(score['prm']['step_logits']) == len(expected_ends)
        assert score['prm']['step_logits_dtype'] == 'torch.bfloat16'
        assert score['prm']['model_frozen'] and score['prm']['optimizer_steps'] == 0
        if i == 0:
            assert score['prm']['literal_head_equal']
        assert al['actor_action_mask'] == [True] * len(processed)
        assert al['actor_attention_mask'] == [True] * (len(raw['prompt_ids']) + len(processed))
        lengths.append(len(processed)); endpoints.append(expected_ends)
        outcomes.append(row['outcome']); groups.append(row['problem_id'])
        boundary_count += len(expected_ends)
    tensors = torch.load(run / 'signals.pt', map_location='cpu', weights_only=True)
    errors, arm_receipts = {}, {}
    for arm in ['P_VR', 'P_PRM']:
        batch = tensors['signals'][arm]
        receipt, loss = read(run / f'{arm}_signal_receipt.json'), read(run / f'{arm}_loss_audit.json')
        assert receipt['normalization_calls'] == 1 and receipt['no_second_normalization']
        assert receipt['kl_estimator'] == 'k1'
        assert receipt['prm_logits_read'] == (arm == 'P_PRM')
        assert receipt['lengths'] == lengths
        assert not receipt['separate_kl_loss']
        if arm == 'P_VR':
            assert torch.count_nonzero(batch['task']) == 0
        else:
            transform = receipt['prm_transform']
            assert transform['input_dtype'] == transform['intermediate_and_output_dtype'] == 'torch.bfloat16'
            assert transform['device'] == 'cuda:0' and transform['cast_after_complete_softmax_signed_softmin']
        expected = signal_oracle(batch, lengths, endpoints, outcomes, groups, arm, receipt['kl_coefficient'])
        errors[arm] = {}
        for name, oracle in expected.items():
            actual = batch[name]
            if oracle.dtype == torch.bool:
                assert torch.equal(actual, oracle), (arm, name)
            else:
                assert torch.isfinite(actual).all(), (arm, name)
                difference = (actual.double() - oracle).abs()
                assert torch.all(difference <= 3e-5 + 3e-5 * oracle.abs()), (arm, name, float(difference.max()))
                errors[arm][name] = float(difference.max())
        assert torch.count_nonzero(batch['advantages'][batch['boundary']]) == 0
        assert torch.count_nonzero(batch['advantages'][~batch['mask']]) == 0
        assert torch.equal(batch['ref_logps'], tensors['reference_logps'])
        assert loss['optimizer_steps'] == 0 and loss['loss_calls'] == 32
        assert loss['advantage_passed_to_loss_unchanged']
        assert math.isfinite(loss['gradient_norm']) and loss['gradient_norm'] > 0
        assert [entry['row'] for entry in loss['loss_trace']] == list(range(32))
        for entry, length in zip(loss['loss_trace'], lengths):
            assert entry['normalization_calls_in_loss'] == 0
            assert entry['valid_tokens'] == length and entry['accumulation_divisor'] == 32
            assert entry['boundary_advantages_zero'] and not entry['separate_kl_loss']
            assert all(math.isfinite(entry[key]) for key in ['avg_ratio', 'avg_advantage', 'scaled_loss'])
        assert len({entry['sealed_input_sha256'] for entry in loss['loss_trace']}) == 1
        arm_receipts[arm] = dict(gradient_norm=loss['gradient_norm'],
                                 old_ref_max_abs=float((batch['old_logps'] - batch['ref_logps']).abs().max()),
                                 normalization_calls=1, loss_normalization_calls=0,
                                 nonzero_boundary_advantages=0, optimizer_steps=0,
                                 avg_ratio_range=[min(e['avg_ratio'] for e in loss['loss_trace']), max(e['avg_ratio'] for e in loss['loss_trace'])])
    for name in ['mask', 'boundary', 'old_logps', 'ref_logps']:
        assert torch.equal(tensors['signals']['P_VR'][name], tensors['signals']['P_PRM'][name])
    preservation = read(run / 'preservation.json')
    assert preservation['passed'] and preservation['actor_and_prm_parameters_unchanged']
    assert preservation['actor_before'] == preservation['actor_after']
    assert preservation['prm_before'] == preservation['prm_after']
    assert preservation['original_AB_and_verl_preserved']
    assert result['optimizer_steps'] == 0 and result['actor_processed_rows_changed'] == changed
    assert result['exact_token_alignment_rows'] == 32 and result['mapped_boundaries'] == boundary_count
    independence = read(run / 'outcome_independence.json')
    assert independence['passed'] and independence['PRM_unchanged'] and independence['VR_changed']
    assert read(run / 'tokenizer_checks.json')['passed']
    assert read(run / 'prm_loading.json')['passed']
    stages = read(run / 'stages.json')
    assert all(math.isfinite(s['seconds']) and s['seconds'] >= 0 for s in stages.values())
    assert math.isclose(result['emitted_tokens_per_second'], sum(len(row['emitted_completion_ids']) for row in raw_rows) / result['generation']['seconds'])
    assert math.isclose(result['prm_responses_per_second'], 32 / stages['official_prm_scoring']['seconds'])
    return dict(passed=True, mode='independent CPU evidence review; no remote access or model loading',
                input_evidence_sha256={name: sha(run / name) for name in required},
                exact_token_rows=32, unique_boundary_count=boundary_count,
                processed_changed_rows=changed, eos=terminal_counts,
                source='literal saved token IDs and independent float64 signal formulas',
                signal_oracle_errors=errors, arms=arm_receipts,
                parameter_hash_equality=True, preservation_receipt_passed=True,
                timing_arithmetic_passed=True,
                limitations=[
                    'PRM task matrix is accepted from saved native-BF16 GPU transform; this script does not rerun the PRM.',
                    'Remote whole-file preservation is supported by the completed runner receipt, not freshly rehashed through this local script.',
                    'No optimizer state memory, multiple updates or training-quality conclusion is certified.',
                    'Displayed throughput is one shared-GPU probe, not a controlled standalone benchmark.'
                ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Refuse to replace a previous review; choose a new output path'
    torch.set_num_threads(1)
    report = audit(args.run)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'input_evidence_sha256'}, indent=2))
