"""CPU-only audit of immutable local evidence. No imports of the training runtime."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json
import math
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASES = {
    17: ROOT / 'execution/abc_seed17_resumed_v1_20261104/completed_evidence',
    42: ROOT / 'execution/ac_seed42_v1_20261106/completed_evidence',
}


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(p * len(values)) - 1)] if values else None


def part_summary(rr):
    tt = [r for r in rr if r['status'] == 'truncated']
    correct = [r for r in rr if r['outcome_reward'] == 1]
    wrong = [r for r in rr if r['outcome_reward'] == 0]
    tokens = sum(r['completion_tokens'] for r in rr)
    return {
        'n': len(rr), 'tokens': tokens,
        'mean_tokens': tokens / len(rr),
        'correct': len(correct),
        'mean_correct_tokens': sum(r['completion_tokens'] for r in correct) / len(correct) if correct else None,
        'mean_wrong_tokens': sum(r['completion_tokens'] for r in wrong) / len(wrong) if wrong else None,
        'truncated': len(tt),
        'truncated_fraction': len(tt) / len(rr),
        'truncated_tokens': sum(r['completion_tokens'] for r in tt),
        'truncated_token_fraction': sum(r['completion_tokens'] for r in tt) / tokens,
    }


def main():
    sources = {}
    training, testing, pairs, samples = [], [], [], []
    for seed, base in BASES.items():
        for arm in ['A', 'C_bounded']:
            p = base / arm / 'compact_signals.jsonl'
            sources[str(p.relative_to(ROOT))] = sha(p)
            rr = rows(p)
            assert len(rr) == 9216
            assert len({(r['tag'], r['optimizer_step_before_update']) for r in rr}) == len(rr)
            correct_lens = [r['completion_tokens'] for r in rr if r['outcome_reward'] == 1]
            trunc = [r for r in rr if r['status'] == 'truncated']
            assert all(r['completion_tokens'] == 2048 and r['outcome_reward'] == 0 for r in trunc)
            # No cached current-policy gradients exist: these are signal coefficients only.
            all_coeff = sum(abs(r['advantage']) for r in rr)
            tr_coeff = sum(abs(r['advantage']) for r in trunc)
            old_l1 = sum(abs(r['advantage']) for r in rr)
            const_l1 = sum(abs(r['advantage']) * r['completion_tokens'] / 2048 for r in rr)
            out = dict(seed=seed, arm=arm, **part_summary(rr))
            out.update(
                status_counts=dict(Counter(r['status'] for r in rr)),
                correct_length_percentiles={str(p): percentile(correct_lens, p) for p in [.5, .9, .95, .99]},
                max_correct_length=max(correct_lens),
                shorter_cap_correct_exceedances={
                    str(cap): {'n': sum(x > cap for x in correct_lens),
                               'fraction': sum(x > cap for x in correct_lens) / len(correct_lens)}
                    for cap in [1024, 1536, 1664, 1792]
                },
                first_48=part_summary([r for r in rr if r['optimizer_step_before_update'] < 48]),
                final_48=part_summary([r for r in rr if r['optimizer_step_before_update'] >= 240]),
                truncated_advantage_signs=dict(Counter('positive' if r['advantage'] > 0 else 'negative' if r['advantage'] < 0 else 'zero' for r in trunc)),
                truncated_old_absolute_coefficient_fraction=tr_coeff / all_coeff,
                constant_length_to_old_absolute_policy_coefficient_ratio=const_l1 / old_l1,
                proposed_extra_minus_half_on_truncated_abs_coefficient=len(trunc) * .5,
                existing_truncated_abs_coefficient=tr_coeff,
                coefficient_replay_assumptions='Policy-only surrogate at ratio=1, without clipping, importance correction, or KL. Not an autograd replay.',
                caveat='Coefficient mass is not a parameter-gradient norm and cannot predict updated accuracy.'
            )
            training.append(out)
            if seed == 17:
                selected = random.Random(20261107 + (arm == 'C_bounded')).sample(trunc, 10)
                for i, r in enumerate(selected):
                    samples.append(dict(sample_id=f'{arm}-{i:02d}', seed=seed, arm=arm,
                                        step=r['optimizer_step_before_update'], tag=r['tag'],
                                        problem_id=r['problem_id']))
            for ds in ['math500', 'gsm8k']:
                p = base / 'evaluation/288' / arm / (ds + '.jsonl')
                sources[str(p.relative_to(ROOT))] = sha(p)
                preds = rows(p)
                assert all(r['reward'] in (0., 1.) for r in preds)
                correct_n = sum(r['reward'] for r in preds)
                wrong = [r for r in preds if r['reward'] == 0]
                statuses = Counter(r['status'] for r in wrong)
                observed_failures = statuses['truncated'] + statuses['missing_or_invalid_box']
                testing.append(dict(seed=seed, arm=arm, dataset=ds, n=len(preds),
                                    correct=int(correct_n), wrong=len(wrong),
                                    accuracy=correct_n / len(preds),
                                    wrong_status_counts=dict(statuses),
                                    truncated_or_invalid_box_wrong=observed_failures,
                                    truncated_or_invalid_box_fraction_of_wrong=observed_failures / len(wrong),
                                    caveat='Disjoint evaluator statuses; not all invalid boxes hide a correct derivation.'))
        for ds in ['math500', 'gsm8k']:
            aa = rows(base / 'evaluation/288/A' / (ds + '.jsonl'))
            cc = rows(base / 'evaluation/288/C_bounded' / (ds + '.jsonl'))
            aa = {r['problem_id']: r for r in aa}
            cc = {r['problem_id']: r for r in cc}
            assert aa.keys() == cc.keys()
            pairs.append(dict(seed=seed, dataset=ds, n=len(aa),
                              both_ended=sum(aa[k]['finish_reason'] == 'stop' and cc[k]['finish_reason'] == 'stop' for k in aa),
                              at_least_one_not_ended=sum(aa[k]['finish_reason'] != 'stop' or cc[k]['finish_reason'] != 'stop' for k in aa),
                              caveat='A union across two policies is not either policy\'s recoverable-error count.'))
    code = ROOT / 'experiments/prefix_protection_20261107/token_loss.py'
    sources[str(code.relative_to(ROOT))] = sha(code)
    build = BASES[17] / 'C_bounded/resume_build_288.json'
    sources[str(build.relative_to(ROOT))] = sha(build)
    settings = json.loads(build.read_text())['settings']
    for k in ['loss_type', 'max_completion_length', 'learning_rate', 'beta', 'per_device_train_batch_size', 'gradient_accumulation_steps', 'mask_truncated_completions']:
        assert k in settings
    out = dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               scope='Offline only; no reward/loss changes, GPU work, or remote writes.',
               source_sha256=sources,
               current_settings={k:settings[k] for k in ['loss_type', 'max_completion_length', 'learning_rate', 'beta', 'per_device_train_batch_size', 'gradient_accumulation_steps', 'mask_truncated_completions']},
               training=training, testing=testing, paired_completion=pairs,
               sampled_training_truncations=samples,
               limitations=['No counterfactual accuracy gain estimated.',
                            'Correct-only length quantiles are censored and survivor-biased.',
                            'Same test sets have already been examined; they are not sealed.'])
    (HERE / 'audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:out[k] for k in ['current_settings','training','testing','paired_completion']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
