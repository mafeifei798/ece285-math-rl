"""CPU paired comparison of the precommitted common-stage pilot checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from support import ROOT, HERE, atomic, read, sha, verify_release


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    midpoint = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [midpoint - half, midpoint + half]


def paired_result(vr, prm, seed=285, replicates=20000):
    x = np.asarray([row['reward'] for row in vr], dtype=np.int8)
    y = np.asarray([row['reward'] for row in prm], dtype=np.int8)
    assert x.shape == y.shape and x.size
    delta = y - x
    counts = np.asarray([(delta == -1).sum(), (delta == 0).sum(), (delta == 1).sum()])
    # Resampling the three observed paired differences is exactly the ordinary
    # question-paired nonparametric bootstrap for binary-accuracy differences.
    draws = np.random.default_rng(seed).multinomial(len(delta), counts / len(delta), size=replicates)
    bootstrap = (draws[:, 2] - draws[:, 0]) / len(delta)
    low, high = np.quantile(bootstrap, [.025, .975])
    discordant = int(counts[0] + counts[2])
    lesser = int(min(counts[0], counts[2]))
    tail = math.fsum(math.exp(math.lgamma(discordant + 1) - math.lgamma(k + 1) -
                             math.lgamma(discordant - k + 1) - discordant * math.log(2))
                     for k in range(lesser + 1)) if discordant else .5
    return dict(count=len(delta), P_VR_correct=int(x.sum()), P_PRM_correct=int(y.sum()),
                P_VR_accuracy=float(x.mean()), P_PRM_accuracy=float(y.mean()),
                P_VR_wilson_95=wilson(int(x.sum()), len(x)), P_PRM_wilson_95=wilson(int(y.sum()), len(y)),
                difference_P_PRM_minus_P_VR=float(delta.mean()),
                paired_bootstrap_95=[float(low), float(high)], bootstrap_replicates=replicates,
                bootstrap_seed=seed, P_VR_only_correct=int(counts[0]), P_PRM_only_correct=int(counts[2]),
                both_correct=int(((x == 1) & (y == 1)).sum()), both_wrong=int(((x == 0) & (y == 0)).sum()),
                mcnemar_exact_two_sided_p=min(1., 2 * tail),
                interpretation='single training seed; intervals quantify question-level resampling, not training-seed uncertainty')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify_release()
    protocol = read(HERE / 'protocol.json')
    assert args.step in protocol['stage_targets']
    assert not args.output.exists(), 'Comparison already exists; inspect before rerunning'
    paths = {arm: ROOT / value / f'evaluation_step_{args.step}' for arm, value in protocol['runs'].items()}
    contracts = {arm: read(path / 'evaluation_config.json') for arm, path in paths.items()}
    summaries = {arm: read(path / 'evaluation_summary.json') for arm, path in paths.items()}
    allowed = {'adapter_sha256', 'adapter_config_sha256'}
    vr_contract = {k: v for k, v in contracts['P_VR'].items() if k not in allowed}
    prm_contract = {k: v for k, v in contracts['P_PRM'].items() if k not in allowed}
    assert vr_contract == prm_contract, 'Evaluation contracts differ beyond adapter files'
    output = dict(passed=True, common_checkpoint_step=args.step, reward_or_verifier_changed=False,
        contract_equal_except_adapter=True, formal_6M_authorized=False,
        release_sha256=sha(HERE / 'release.json'), datasets={}, stages={},
        limitations=['One short, single-seed pilot; not evidence of convergence or a paper-level improvement.',
                     'Equal update/question/response counts, not equal actual generated tokens or GPU time.',
                     'Historical 6M GRPO A/B are not this pilot\'s matched-budget primary comparison.',
                     'Neither a nonsignificant accuracy difference nor a positive PRM reward establishes equivalence or better reasoning.'])
    paired_rows = []
    for arm, relative in protocol['runs'].items():
        stage = read(ROOT / relative / f'stage_{args.step}.json')
        assert stage['batch_step'] == args.step
        output['stages'][arm] = stage
    for dataset in ['math500', 'gsm8k']:
        by_arm = {}
        for arm, path in paths.items():
            file = path / f'{dataset}.jsonl'
            assert sha(file) == summaries[arm]['datasets'][dataset]['predictions_sha256']
            data = read_rows(file)
            by_id = {row['problem_id']: row for row in data}
            assert len(by_id) == len(data) == (500 if dataset == 'math500' else 1319)
            by_arm[arm] = by_id
        assert set(by_arm['P_VR']) == set(by_arm['P_PRM'])
        keys = sorted(by_arm['P_VR'])
        a, b = [by_arm['P_VR'][key] for key in keys], [by_arm['P_PRM'][key] for key in keys]
        for key, vr, prm in zip(keys, a, b):
            assert all(vr[field] == prm[field] for field in ['gold', 'question_sha256', 'prompt_sha256'])
            paired_rows.append(dict(dataset=dataset, problem_id=key,
                P_VR_correct=vr['reward'], P_PRM_correct=prm['reward'],
                P_VR_status=vr['status'], P_PRM_status=prm['status'],
                P_VR_tokens=vr['generated_tokens'], P_PRM_tokens=prm['generated_tokens'],
                P_VR_empty=vr['empty'], P_PRM_empty=prm['empty'],
                P_VR_short=vr['short'], P_PRM_short=prm['short']))
        output['datasets'][dataset] = dict(paired=paired_result(a, b),
            behavior={arm: summaries[arm]['datasets'][dataset] for arm in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paired_path = args.output.with_name('paired_questions.jsonl')
    assert not paired_path.exists(), 'Do not overwrite a previous paired comparison'
    with paired_path.open('x') as stream:
        for row in paired_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    output['paired_questions_sha256'] = sha(paired_path)
    atomic(args.output, output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
