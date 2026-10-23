"""CPU-only exact-response parity and fail-closed checks; no model loading."""
import argparse
import copy
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from guard import GuardedPureIntegration, EmptyResponseBatchError, require_nonempty_batch
from common import atomic, read_jsonl
from trainer import PureIntegration
from signals import micro_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    rows = read_jsonl(a.run / 'generated_exact_ids.jsonl')
    scored = read_jsonl(a.run / 'scored_alignment.jsonl')
    rows = [dict(r, outcome=s['outcome']) for r, s in zip(rows, scored)]
    native = [s['native'] for s in scored]
    saved = torch.load(a.run / 'signals.pt', map_location='cpu', weights_only=True)
    plain = object.__new__(PureIntegration)
    fixed = object.__new__(GuardedPureIntegration)
    plain.tokenizer = fixed.tokenizer = tokenizer
    # No backend, policy weights, or optimizer is constructed.
    checks = {}
    for arm in ['P_VR', 'P_PRM']:
        old = saved['signals'][arm]['old_logps']
        prm = native if arm == 'P_PRM' else None
        before, positions = plain.prepare(arm, rows, prm, old, saved['reference_logps'])
        after, positions2 = fixed.prepare(arm, rows, prm, old, saved['reference_logps'])
        assert positions == positions2
        names = ['mask', 'task_rewards', 'kl', 'rewards', 'returns', 'advantages', 'old_logps']
        for name in names:
            assert torch.equal(getattr(before, name), getattr(after, name)), name
        torch.testing.assert_close(after.advantages, saved['signals'][arm]['advantages'], atol=5e-6, rtol=5e-6)
        assert after.whitening_calls == 1
        max_grad_difference = 0.
        for i, r in enumerate(rows):
            n = len(r['completion_ids'])
            x = old[i:i+1, :n].clone().requires_grad_()
            y = x.detach().clone().requires_grad_()
            lb, _ = micro_loss(before, i, x)
            la, trace = micro_loss(after, i, y)
            lb.backward(); la.backward()
            assert torch.equal(x.grad, y.grad)
            assert trace['normalization_calls'] == 1 and trace['advantage_retained']
            max_grad_difference = max(max_grad_difference, float((x.grad-y.grad).abs().max()))
        rejected = []
        for raw in ['', ' \r\n\t ', '\n\n', '\u2003']:
            bad = copy.deepcopy(rows)
            bad[0]['completion'] = raw
            bad[0]['completion_ids'] = tokenizer.encode(raw, add_special_tokens=False) + [tokenizer.eos_token_id]
            try:
                fixed.prepare(arm, bad, None, old, saved['reference_logps'])
            except EmptyResponseBatchError as exc:
                assert len(exc.incidents) == 1 and exc.incidents[0]['row'] == 0
                rejected.append(repr(raw))
            else:
                raise AssertionError('Empty output reached reward/loss preparation')
        checks[arm] = dict(nonempty_rows=32, all_signal_tensors_bitwise_equal=True,
            token_positions_equal=True, whitening_calls=1,
            logprob_gradient_max_difference=max_grad_difference,
            empty_cases_stopped_before_rewards=rejected)
    # A semantic filter has NOT been added: these remain scored normally.
    for text in ['Thank you.', 'Answer without a box.', '2', 'x=2.', '\\boxed{}']:
        require_nonempty_batch([dict(completion=text)])
    receipt = dict(passed=True, checks=checks, new_generations=0,
        policy_model_loaded=False, optimizer_steps=0, nonempty_nonanswers_are_not_filtered=True,
        boundary_rule='same whole-batch stop in P_VR and P_PRM; no reward substitution or row dropping',
        formal_training_ready=False, PRM_semantic_vulnerability_fixed=False)
    atomic(a.output, receipt)
    print(receipt)


if __name__ == '__main__':
    main()
