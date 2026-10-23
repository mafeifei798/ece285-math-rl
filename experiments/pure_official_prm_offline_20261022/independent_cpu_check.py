"""Independent stdlib float64 verification; no torch/model/GPU imports.

Reproduce with: python3 -B experiments/pure_official_prm_offline_20261022/independent_cpu_check.py
Existing review is verified rather than overwritten; remove/revise only deliberately.
"""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'execution/pure_official_prm_offline_20261022'
ARMS = ('P_VR', 'QWEN_PRM', 'PURE_NATIVE', 'PURE_RL_PROMPT')


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = dict(
        inputs=ROOT/'execution/pure_empty_boundary_20261022/inputs.jsonl',
        qwen=OUT/'remote/qwen_single/scored.jsonl',
        pure=OUT/'remote/scoring_single/scored.jsonl',
        metrics=OUT/'per_response_metrics.jsonl',
        summary=OUT/'summary.json')
    inputs, qwen, pure, metrics = [lines(paths[k]) for k in ('inputs','qwen','pure','metrics')]
    summary = json.loads(paths['summary'].read_text())
    assert all(len(x) == 288 for x in (inputs, qwen, pure, metrics))
    assert digest(paths['qwen']) == summary['provenance']['qwen_scored_sha256']
    assert digest(paths['pure']) == summary['provenance']['pure_scored_sha256']
    batches = collections.defaultdict(list)
    for i, row in enumerate(inputs):
        assert i == qwen[i]['index'] == pure[i]['index'] == metrics[i]['index']
        h = hashlib.sha256(row['completion'].encode()).hexdigest()
        assert h == qwen[i]['input_sha256'] == pure[i]['input_sha256'] == metrics[i]['raw_sha256']
        batches[row['batch_id']].append(i)
    errors, empty = [], collections.defaultdict(list)
    skipped = []
    for batch_id, ix in batches.items():
        assert len(ix) == 32
        if any('alignment_error' in qwen[i] for i in ix):
            assert all(not metrics[i]['advantage_available'] for i in ix)
            skipped.append(batch_id)
            continue
        assert all(metrics[i]['advantage_available'] for i in ix)
        for arm in ARMS:
            returns, totals = [], []
            for i in ix:
                length = len(qwen[i]['reconstructed_ids'])
                assert length == metrics[i]['reconstructed_length'] and 0 < length <= 2048
                sparse = {}
                if arm == 'P_VR':
                    sparse[length-1] = inputs[i]['outcome']
                else:
                    ps = qwen[i]['native']['step_scores'] if arm == 'QWEN_PRM' else pure[i]['variants'][
                        'native_raw' if arm == 'PURE_NATIVE' else 'official_rl_prompt']['step_scores']
                    signed = [2*p-1 for p in ps]
                    z = [-r/.1 for r in signed]
                    exponentials = [math.exp(x-max(z)) for x in z]
                    denominator = math.fsum(exponentials)
                    positions = qwen[i]['alignment']['actor_positions']
                    assert len(positions) == len(signed)
                    for pos, reward, ex in zip(positions,signed,exponentials):
                        sparse[pos] = .5*reward*ex/denominator
                totals.append(math.fsum(sparse.values()))
                tail, row_returns = 0., [0.]*length
                for token in reversed(range(length)):
                    tail += sparse.get(token,0.)
                    row_returns[token] = tail
                returns.append(row_returns)
            raw = []
            for j,i in enumerate(ix):
                peers = [k for k,p in enumerate(ix) if p != i and inputs[p]['problem_id'] == inputs[i]['problem_id']]
                assert len(peers) == 7
                baseline = math.fsum(math.fsum(returns[k]) for k in peers)/(7*2048)
                raw.append([value-baseline for value in returns[j]])
            flattened = [x for row in raw for x in row]
            mu = math.fsum(flattened)/len(flattened)
            var = math.fsum((x-mu)**2 for x in flattened)/(len(flattened)-1)
            for j,i in enumerate(ix):
                before = raw[j][-1]
                after = (before-mu)/math.sqrt(var+1e-8)
                recorded = metrics[i]['variants'][arm]
                errors.append(max(abs(before-recorded['raw_RLOO_terminal_advantage']),
                                  abs(after-recorded['whitened_terminal_advantage'])))
                assert abs(totals[j]-recorded['task_reward']) < 2e-6
                if not inputs[i]['completion'].strip():
                    correct = [k for k,p in enumerate(ix) if p != i and
                        inputs[p]['problem_id'] == inputs[i]['problem_id'] and inputs[p]['outcome'] == 1]
                    beaten = sum(totals[j] > totals[k]+1e-7 for k in correct)
                    assert beaten == recorded['higher_than_correct_peers']
                    empty[arm].append(dict(index=i,batch_id=batch_id,raw_advantage=before,
                        whitened_advantage=after,correct_peers=len(correct),
                        correct_peers_outranked=beaten))
    assert skipped == ['B:210']
    assert len(errors) == 1024 and max(errors) < 1e-5
    counts = {}
    for arm, rows in empty.items():
        counts[arm] = dict(n=len(rows),raw_positive=sum(r['raw_advantage']>0 for r in rows),
            whitened_positive=sum(r['whitened_advantage']>0 for r in rows),
            correct_peers=sum(r['correct_peers'] for r in rows),
            correct_peers_outranked=sum(r['correct_peers_outranked'] for r in rows))
        assert len(rows) == 8
        assert counts[arm]['raw_positive'] == counts[arm]['whitened_positive'] == (0 if arm=='P_VR' else 8)
    assert counts['QWEN_PRM']['correct_peers_outranked'] == counts['PURE_NATIVE']['correct_peers_outranked'] == 23
    assert counts['PURE_RL_PROMPT']['correct_peers_outranked'] == 22
    historical_empty = [r for r in inputs if not r['completion'].strip()]
    assert len(historical_empty) == 9
    old_zero = all(r['old_reward'] == 0 and not r['old_prm_scores'] for r in historical_empty)
    assert old_zero
    result = dict(passed=True,method='Python standard library float64; no project signal functions reused',
        script=str(Path(__file__).resolve()),script_sha256=digest(Path(__file__)),
        source_files={k:dict(path=str(p),sha256=digest(p)) for k,p in paths.items()},
        cpu_only=True,torch_imported=False,model_loaded=False,gpu_jobs_started=0,
        optimizer_steps=0,new_generations=0,verified_row_arm_pairs=len(errors),
        max_terminal_advantage_abs_error=max(errors),empty_signal_counts=counts,
        historical_empty_rows=9,historical_empty_rewards_zero_and_prm_not_called=old_zero,
        skipped_batches=skipped,empty_case_details=dict(empty),
        interpretation=[
            'Both PRMs produce a positive raw and whitened signal for each of eight replayable empty responses; P-VR does not.',
            'Duplicate normalization is not needed to create this failure: every PRM raw advantage is already positive before the single whitening.',
            '23/23 and 22/23 count comparisons against answer-correct peer responses, not independent questions or process-label error rates.',
            'Historical A/B assigned these empty responses zero reward and did not call PRM, so this finding does not explain the old B versus A accuracy difference.',
            'This is selected historical text with reconstructed tokens and KL=0, not a completed new training experiment or proof of future collapse.'])
    destination = OUT/'independent_review.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == result, 'Existing review differs; preserving it'
    else:
        with destination.open('x') as handle:
            json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.write('\n')
    print(json.dumps(dict(passed=True,max_abs_error=max(errors),empty_signal_counts=counts,
                          saved=str(destination)),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
