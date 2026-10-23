"""PURE sparse task rewards, RLOO, and a loss boundary that rejects renormalization."""
from dataclasses import dataclass
from collections import Counter
import torch
from common import N, GROUP, BATCH
from vendor import pure_core as official

@dataclass
class SignalBatch:
    arm: str
    mask: torch.Tensor
    task_rewards: torch.Tensor
    kl: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    sealed_advantages: torch.Tensor
    old_logps: torch.Tensor
    whitening_calls: int

def task_rewards(arm, lengths, outcomes, scores, positions, device='cpu'):
    reward = torch.zeros((len(lengths), N), dtype=torch.float32, device=device)
    if arm not in ('P_VR', 'P_PRM'):
        raise ValueError(arm)
    for i, length in enumerate(lengths):
        assert 0 < length <= N
        if arm == 'P_VR':
            assert outcomes[i] in (0, 1, 0.0, 1.0)
            reward[i, length - 1] = float(outcomes[i])
        else:
            ps, ends = scores[i], positions[i]
            assert len(ps) == len(ends) > 0 and sorted(set(ends)) == ends
            assert all(0 <= v <= 1 for v in ps) and all(0 <= p < length for p in ends)
            r = 2 * torch.tensor(ps, dtype=torch.float32, device=device) - 1
            weights = torch.softmax(-r / 0.1, dim=-1)
            reward[i, ends] = 0.5 * r * weights
    return reward

def make_signals(arm, lengths, outcomes, scores, positions, groups, old_logps, ref_logps):
    assert len(lengths) == BATCH and all(n == GROUP for n in Counter(groups).values())
    assert len(set(groups)) == 4
    assert old_logps.shape == ref_logps.shape == (BATCH, N)
    assert torch.isfinite(old_logps).all() and torch.isfinite(ref_logps).all()
    device = old_logps.device
    mask = (torch.arange(N, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]).float()
    task = task_rewards(arm, lengths, outcomes, scores, positions, device)
    with torch.no_grad():
        kl = official.kl_penalty(old_logps, ref_logps, 'low_var_kl') * mask
        reward = (task - 0.001 * kl) * mask
        # This is the only normalization in the complete preparation/loss path.
        calls = []
        original = official.verl_F.masked_whiten
        def tracked(*a, **kw):
            calls.append(1)
            return original(*a, **kw)
        official.verl_F.masked_whiten = tracked
        try:
            adv, ret = official.compute_rloo_outcome_advantage(reward, mask, list(groups),
                adv_norm=True, return_aggregate_method='sum')
        finally:
            official.verl_F.masked_whiten = original
        assert len(calls) == 1 and torch.isfinite(adv).all()
        assert torch.count_nonzero(adv * (1 - mask)) == 0
        return SignalBatch(arm, mask, task, kl, reward, ret, adv, adv.clone(), old_logps.detach().clone(), len(calls))

def micro_loss(batch, row, new_logps):
    """Static microbatch of one; caller accumulates each loss / 32."""
    length = int(batch.mask[row].sum())
    assert new_logps.shape == (1, length)
    actual = batch.advantages[row:row + 1, :length]
    expected = batch.sealed_advantages[row:row + 1, :length]
    if not torch.equal(actual, expected):
        raise ValueError('Advantage changed after one global whitening')
    assert batch.whitening_calls == 1
    loss, clipfrac, approx_kl, avg_ratio = official.compute_policy_loss(
        batch.old_logps[row:row + 1, :length], new_logps,
        actual, batch.mask[row:row + 1, :length], cliprange=0.2)
    return loss / BATCH, dict(row=row, tokens=length, loss=float(loss.detach()),
        advantage_retained=True, normalization_calls=batch.whitening_calls,
        clip_fraction=float(clipfrac.detach()), approximate_kl=float(approx_kl.detach()),
        average_ratio=float(avg_ratio.detach()))
