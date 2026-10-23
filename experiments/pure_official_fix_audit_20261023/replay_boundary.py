"""CPU frozen-rollout replay; no GPU, generation, training, or reward edits.

Compare existing veRL signals with an explicitly labelled transplant of pinned
OpenRLHF boundary suppression + its single normalization. This is not a complete
OpenRLHF reproduction. Run from any directory with python3 -B <this file>.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pure_integration_20261022'))
sys.path.insert(0, str(ROOT / 'experiments/pure_official_prm_offline_20261022'))
from analyze import checked_signals, sign, spearman
from common import atomic, read, read_jsonl, sha, write_jsonl
from signals import task_rewards
from vendor import pure_core as official
from boundary_adapter import apply_openrlhf_boundary_norm

OUT = ROOT / 'execution/pure_official_fix_audit_20261023'
PREV = ROOT / 'execution/pure_official_prm_offline_20261022'
ARMS = ('P_VR', 'QWEN_PRM', 'PURE_NATIVE', 'PURE_RL_PROMPT')


def local_loss_gradient(adv):
    """Derivative wrt synthetic log-prob at ratio=1; NOT model-weight backward."""
    length = adv.numel()
    logps = torch.zeros((1, length), requires_grad=True)
    old = torch.zeros_like(logps)
    loss, *_ = official.compute_policy_loss(old, logps, adv.reshape(1, length),
                                           torch.ones_like(logps), cliprange=.2)
    derivative, = torch.autograd.grad(loss / 32, logps)
    torch.testing.assert_close(derivative[0], -adv / (32 * length), atol=1e-7, rtol=1e-6)
    return derivative[0].detach()


def main():
    torch.set_num_threads(2)
    paths = dict(inputs=ROOT/'execution/pure_empty_boundary_20261022/inputs.jsonl',
                 qwen=PREV/'remote/qwen_single/scored.jsonl',
                 pure=PREV/'remote/scoring_single/scored.jsonl')
    rows, qwen, pure = [read_jsonl(paths[k]) for k in ('inputs', 'qwen', 'pure')]
    assert len(rows) == len(qwen) == len(pure) == 288
    assert sha(paths['inputs']) == 'e3d564b97c914ab60b56a0911a12fa4bc06ee4adc490e7898ae761fbc1f3400d'
    old_metrics = read_jsonl(PREV/'per_response_metrics.jsonl')
    batches = collections.defaultdict(list)
    for i, (row, q, p) in enumerate(zip(rows, qwen, pure)):
        digest = hashlib.sha256(row['completion'].encode()).hexdigest()
        assert q['index'] == p['index'] == i
        assert digest == q['input_sha256'] == p['input_sha256']
        batches[row['batch_id']].append(i)
    records, receipts, batch_metrics, skipped = [], [], [], []
    for batch_id, indices in batches.items():
        assert len(indices) == 32
        failures = [i for i in indices if 'alignment_error' in qwen[i]]
        if failures:
            skipped.append(dict(batch_id=batch_id, indices=failures,
                                reason='unchanged whole-batch exclusion from previous replay'))
            continue
        lengths = [len(qwen[i]['reconstructed_ids']) for i in indices]
        ends = [qwen[i]['alignment']['actor_positions'] for i in indices]
        outcomes = [rows[i]['outcome'] for i in indices]
        groups = [rows[i]['problem_id'] for i in indices]
        assert sorted(collections.Counter(groups).values()) == [8]*4
        valid = torch.arange(2048)[None, :] < torch.tensor(lengths)[:, None]
        boundary = torch.zeros_like(valid)
        for j, n in enumerate(lengths):
            assert sorted(set(ends[j])) == ends[j] and ends[j][-1] == n-1
            boundary[j, ends[j]] = True
        probs = dict(P_VR=[qwen[i]['native']['step_scores'] for i in indices],
                     QWEN_PRM=[qwen[i]['native']['step_scores'] for i in indices],
                     PURE_NATIVE=[pure[i]['variants']['native_raw']['step_scores'] for i in indices],
                     PURE_RL_PROMPT=[pure[i]['variants']['official_rl_prompt']['step_scores'] for i in indices])
        for arm in ARMS:
            task = task_rewards('P_VR' if arm == 'P_VR' else 'P_PRM', lengths, outcomes, probs[arm], ends)
            task_before = task.clone()
            raw, original, returns = checked_signals(task, valid.float(), groups)
            raw_before = raw.clone()
            # Separate branches: raw -> veRL whitening, raw -> OpenRLHF mask+norm.
            # Never feed original (already normalized) into the adapter.
            changed, receipt = apply_openrlhf_boundary_norm(raw, valid, boundary)
            assert torch.equal(task, task_before) and torch.equal(raw, raw_before)
            assert torch.all(changed[boundary] == 0)
            receipts.append(dict(batch_id=batch_id, arm=arm, **receipt))
            content = valid & ~boundary
            bm = dict(batch_id=batch_id, arm=arm, valid_tokens=int(valid.sum()),
                      boundary_tokens=int(boundary.sum()), content_tokens=int(content.sum()),
                      changed_boundary_nonzero=int((original[boundary].abs() > 1e-7).sum()),
                      boundary_nonzero_after=int((changed[boundary] != 0).sum()),
                      content_sign_changes=int((torch.sign(original[content]) != torch.sign(changed[content])).sum()),
                      mean_abs_content_adv_before=float(original[content].abs().mean()),
                      mean_abs_content_adv_after=float(changed[content].abs().mean()),
                      reward_unchanged=True, raw_baseline_unchanged=True,
                      normalization_calls_per_branch=1)
            batch_metrics.append(bm)
            for j, i in enumerate(indices):
                n = lengths[j]
                old = old_metrics[i]['variants'][arm]
                assert abs(float(task[j].sum()) - old['task_reward']) < 2e-6
                assert abs(float(original[j, n-1]) - old['whitened_terminal_advantage']) < 2e-5
                assert abs(float(raw[j, n-1]) - old['raw_RLOO_terminal_advantage']) < 2e-5
                rec = dict(index=i, batch_id=batch_id, problem_id=groups[j], arm=arm,
                           empty=not rows[i]['completion'].strip(), tokens=n,
                           steps=len(ends[j]), outcome=outcomes[j],
                           reward_before=float(task_before[j].sum()), reward_after=float(task[j].sum()),
                           raw_terminal=float(raw[j, n-1]),
                           normalized_terminal_before=float(original[j, n-1]),
                           normalized_terminal_after=float(changed[j, n-1]),
                           positive_content_before=int((original[j, :n][~boundary[j, :n]] > 1e-7).sum()),
                           positive_content_after=int((changed[j, :n][~boundary[j, :n]] > 1e-7).sum()),
                           mean_adv_before=float(original[j, :n].mean()),
                           mean_adv_after=float(changed[j, :n].mean()),
                           logged_length_matches=qwen[i]['logged_length_matches'])
                if rec['empty']:
                    assert n == 1 and ends[j] == [0]
                    gb = local_loss_gradient(original[j, :n])
                    ga = local_loss_gradient(changed[j, :n])
                    assert torch.all(ga == 0)
                    peers = [k for k in range(32) if groups[k] == groups[j] and k != j]
                    # For a peer baseline, this empty row contributes reward/(7*N).
                    contribution = float(returns[j].sum()) / (len(peers)*2048)
                    rec.update(dloss_dlogp_before=float(gb[0]), dloss_dlogp_after=float(ga[0]),
                               unchanged_empty_contribution_to_each_peer_baseline=contribution)
                records.append(rec)
    assert len(skipped) == 1 and skipped[0]['batch_id'] == 'B:210'
    assert len(records) == 256*4 and len(receipts) == 32
    aggregates = {}
    for arm in ARMS:
        subset = [r for r in records if r['arm'] == arm]
        empty = [r for r in subset if r['empty']]
        normal = [r for r in subset if not r['empty']]
        bm = [r for r in batch_metrics if r['arm'] == arm]
        assert len(empty) == 8
        rankings = []
        for key in {(r['batch_id'], r['problem_id']) for r in subset}:
            g = [r for r in subset if (r['batch_id'], r['problem_id']) == key]
            a, b = [r['reward_before'] for r in g], [r['reward_after'] for r in g]
            assert a == b
            rankings.append(spearman([r['mean_adv_before'] for r in g], [r['mean_adv_after'] for r in g]))
        aggregates[arm] = dict(rows=len(subset), empty=len(empty),
            empty_raw_positive=sum(r['raw_terminal'] > 1e-7 for r in empty),
            empty_normalized_positive_before=sum(r['normalized_terminal_before'] > 1e-7 for r in empty),
            empty_normalized_positive_after=sum(r['normalized_terminal_after'] > 1e-7 for r in empty),
            empty_direct_loss_gradient_nonzero_before=sum(abs(r['dloss_dlogp_before']) > 1e-7 for r in empty),
            empty_direct_loss_gradient_nonzero_after=sum(r['dloss_dlogp_after'] != 0 for r in empty),
            empty_normalized_range_before=[min(r['normalized_terminal_before'] for r in empty),max(r['normalized_terminal_before'] for r in empty)],
            empty_peer_baseline_contribution_range=[min(r['unchanged_empty_contribution_to_each_peer_baseline'] for r in empty),max(r['unchanged_empty_contribution_to_each_peer_baseline'] for r in empty)],
            valid_tokens=sum(b['valid_tokens'] for b in bm), boundary_tokens=sum(b['boundary_tokens'] for b in bm),
            content_tokens=sum(b['content_tokens'] for b in bm), content_sign_changes=sum(b['content_sign_changes'] for b in bm),
            normalized_terminal_nonzero_after=sum(r['normalized_terminal_after'] != 0 for r in subset),
            nonempty_mean_adv_sign_changes=sum(sign(r['mean_adv_before']) != sign(r['mean_adv_after']) for r in normal),
            mean_sequence_adv_rank_spearman=sum(x for x in rankings if x is not None)/sum(x is not None for x in rankings),
            reward_and_reward_rank_unchanged=True)
    # Synthetic two-action counterexample, not an additional generated answer:
    # masking EOS does not remove positive gradient on preceding content.
    syn_raw = torch.tensor([[.6,.6],[-.4,-.4],[.7,.0]])
    syn_mask = torch.ones_like(syn_raw, dtype=torch.bool)
    syn_boundary = torch.tensor([[False,True]]*3)
    syn, syn_receipt = apply_openrlhf_boundary_norm(syn_raw, syn_mask, syn_boundary)
    assert syn[0,0] > 0 and syn[0,1] == 0
    result = dict(passed=True, scope='frozen veRL raw advantage + official OpenRLHF boundary/normalizer transplant',
                  not_full_openrlhf_reproduction=True, arms=aggregates,
                  rows_scored_previously=288, rows_replayed=256, empty_replayed=8,
                  skipped_batches=skipped, input_hashes={k:sha(p) for k,p in paths.items()},
                  historical_tokens_reconstructed=True, width=2048, group_size=8, batch_size=32,
                  process_coefficient=.5, softmin_temperature=.1, offline_kl=0,
                  normalization_calls_per_branch=1, rewards_changed=False, verifier_changed=False,
                  whole_group_selection_unchanged=True,
                  float64_oracle_max_abs_error=max(r['float64_oracle']['max_abs_error'] for r in receipts),
                  matched_previous_replay=True, cpu_only=True, optimizer_steps=0, new_generations=0,
                  synthetic_limit_example=dict(raw=syn_raw.tolist(), normalized=syn.tolist(),
                      conclusion='a nonboundary content token can remain positively reinforced',
                      generated_model_answer=False),
                  limitations=['Selected empty-containing batches, not a representative sample.',
                      'Historical generated token IDs unavailable; reconstruction and canonical boundaries reused.',
                      'KL zero because historical old/ref logprobs unavailable; not a historical gradient replay.',
                      'PRM scores/rankings and empty contribution to peer baselines remain unchanged.',
                      'Every arm also loses normal EOS/step-boundary gradients; this is an algorithm variant.',
                      'Synthetic logprob derivatives do not demonstrate learned accuracy or stability.'])
    atomic(OUT/'boundary_replay.json', result)
    atomic(OUT/'boundary_batch_metrics.json', batch_metrics)
    atomic(OUT/'boundary_normalization_receipts.json', receipts)
    write_jsonl(OUT/'boundary_per_response.jsonl', records)
    atomic(OUT/'boundary_empty_examples.json', [r for r in records if r['empty']])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
