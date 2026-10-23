"""Single-device PURE OpenRLHF signal/loss integration, with no training launcher.

P-VR and P-PRM share the pinned source family, masks, KL treatment and loss.
The PRM transform uses its explicitly selected native dtype/device; baseline,
KL, returns and normalization run on CPU. micro_loss transfers only one frozen
signal row to new log probabilities' device. No optimizer is defined or stepped.
This module does not generate responses or certify tokenizer/step alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

import vendor


HERE = Path(__file__).resolve().parent
PIN = '41f9363b8cc4af044b7fc5f7f9bbbf62139e9553'
VENDOR_HASH = '539479a97a120162d4bc2f26e0098cd305b0907beca495f75996cc031214c223'
TENSOR_FIELDS = ('mask', 'boundary', 'task', 'kl', 'baseline_reward', 'reward',
                 'raw_returns', 'returns', 'advantages', 'old_logps', 'ref_logps')


def _source_receipt():
    actual = hashlib.sha256((HERE / 'vendor.py').read_bytes()).hexdigest()
    if actual != VENDOR_HASH:
        raise RuntimeError(f'Vendored official AST file changed: {actual}')
    manifest = json.loads((HERE / 'vendor_manifest.json').read_text())
    if manifest['commit'] != PIN or manifest['vendor_sha256'] != actual:
        raise RuntimeError('Vendor manifest does not match the fixed pin')
    return manifest


def official_functions():
    """Expose checked fixed AST functions; engine must supply FP32 logits."""
    _source_receipt()
    names = ('log_probs_from_logits', 'compute_approx_kl', 'compute_reward',
             'masked_mean', 'right_padding', 'turn_process_reward_logits_to_reward',
             'process_experiences', 'normalize', 'PolicyLoss')
    return {name:getattr(vendor,name) for name in names}


def _cpu(value, name, dtype=torch.float32):
    if isinstance(value, SignalBatch):
        raise RuntimeError(f'{name} cannot be a sealed SignalBatch; signals must not be normalized twice')
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', dtype=dtype).clone()
    return torch.as_tensor(value, device='cpu', dtype=dtype).clone()


def _fingerprint(tensors):
    result = hashlib.sha256()
    for name in TENSOR_FIELDS:
        tensor = tensors[name]
        if tensor.device.type != 'cpu':
            raise RuntimeError('Sealed signals must remain on CPU')
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(str(tensor.dtype).encode())
        result.update(tensor.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


@dataclass(frozen=True)
class SignalBatch:
    arm: str
    lengths: tuple[int, ...]
    groups: tuple
    mask: torch.Tensor
    boundary: torch.Tensor
    task: torch.Tensor
    kl: torch.Tensor
    baseline_reward: torch.Tensor
    reward: torch.Tensor
    raw_returns: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_logps: torch.Tensor
    ref_logps: torch.Tensor
    receipt: dict
    sealed: bool
    seal_sha256: str

    def assert_sealed(self):
        if not self.sealed or self.receipt.get('normalization_calls') != 1:
            raise RuntimeError('Expected a sealed, once-normalized SignalBatch')
        if _fingerprint({name:getattr(self, name) for name in TENSOR_FIELDS}) != self.seal_sha256:
            raise RuntimeError('Sealed signals were modified after construction')

    def normalize(self):
        raise RuntimeError('SignalBatch is already sealed and normalized once; repeated normalization is forbidden')


class _Buffer:
    mask_separator_adv = True
    def __init__(self, rows):
        self.items = [SimpleNamespace(advantages=row, action_mask=torch.ones_like(row, dtype=torch.bool)) for row in rows]
    def __iter__(self):
        return iter(self.items)


class _IdentityStrategy:
    def __init__(self):
        self.calls = 0
    def all_reduce(self, value, operation):
        if operation != 'sum':
            raise RuntimeError('Unexpected official reduction')
        self.calls += 1
        return value


def _normalize_once(boundary_masked_returns, lengths):
    rows = [boundary_masked_returns[i, :length].clone() for i,length in enumerate(lengths)]
    values64 = torch.cat(rows).double()
    zero = values64 == 0
    count = int((~zero).sum())
    if count == 0:
        raise ValueError('ALL_ZERO_ADVANTAGES: official OpenRLHF normalization denominator is zero; blocked, no substitute signal')
    # Independent float64 oracle preserves the source's unusual zero semantics:
    # exclude zeros from count, include them in the variance numerator, restore.
    mean64 = values64.sum() / count
    variance64 = ((values64 - mean64) ** 2).sum() / count
    expected = (values64 - mean64) / torch.sqrt(variance64.clamp(min=1e-8))
    expected[zero] = 0
    buffer = _Buffer(rows)
    strategy = _IdentityStrategy()
    vendor.normalize(buffer, 'advantages', strategy)
    assert strategy.calls == 2
    result = torch.zeros_like(boundary_masked_returns)
    actual_rows = []
    for i,length in enumerate(lengths):
        result[i, :length] = buffer.items[i].advantages
        actual_rows.append(buffer.items[i].advantages)
    actual = torch.cat(actual_rows).double()
    error = (actual - expected).abs()
    if not torch.isfinite(actual).all() or not torch.all(error <= 2e-5 + 2e-5 * expected.abs()):
        raise RuntimeError('Official normalization disagrees with independent float64 oracle')
    if torch.any(actual[zero] != 0):
        raise RuntimeError('Official normalization failed to preserve initial zeros')
    return result, dict(calls=1, identity_all_reduce_calls=2, nonzero_valid_count=count,
                       preserved_zero_valid_count=int(zero.sum()),
                       zero_slots_in_variance_numerator=True,
                       population_denominator=True, variance_floor=1e-8,
                       float64_oracle_max_abs_error=float(error.max()),
                       all_zero_behavior='block with ValueError; never replace with invented rewards')


@torch.no_grad()
def make_signals(arm, lengths, outcomes, step_logits, positions, groups,
                 old_logps, ref_logps, *, kl_coef=0.01, use_kl_estimator_k3=False,
                 step_logits_dtype='float32', prm_transform_device='cpu',
                 input_stage='old_policy_logps_and_raw_prm_logits'):
    """Create a sealed CPU SignalBatch; accepts GPU old/ref logps by CPU copy.

    ``positions`` are sorted 0-based valid-response boundary indices including
    the last valid action, even on truncation. ``step_logits`` is a row-ragged
    list of [negative, positive] logits, required only for P-PRM. P-VR never reads
    this argument. Group labels must be contiguous, of one common size >= 2.
    ``outcomes`` are unchanged verifier 0/1 results; PRM-only does not use them
    in its reward. Log probabilities have padded shape [B,T], in response order.
    Real BF16 PRM logits must explicitly use step_logits_dtype='bfloat16' and
    prm_transform_device matching the model device, so the native official
    softmax/signed/softmin runs in that dtype/device before the FP32 task copy.
    P-VR ignores both PRM transform arguments without evaluating them.
    """
    if input_stage != 'old_policy_logps_and_raw_prm_logits':
        raise RuntimeError('Only pre-signal inputs are accepted; no repeat normalization')
    canonical = {'P-VR':'P-VR', 'VR':'P-VR', 'P-PRM':'P-PRM', 'PRM':'P-PRM'}
    if arm not in canonical:
        raise ValueError('This controlled integration supports only P-VR and P-PRM')
    arm = canonical[arm]
    sources = _source_receipt()
    lengths = tuple(int(n) for n in lengths)
    groups = tuple(groups)
    if not lengths or any(n < 1 for n in lengths) or len(groups) != len(lengths):
        raise ValueError('Nonempty lengths and matching group labels are required')
    if not math.isfinite(float(kl_coef)) or kl_coef < 0:
        raise ValueError('KL coefficient must be finite and nonnegative')
    old = _cpu(old_logps, 'old_logps')
    ref = _cpu(ref_logps, 'ref_logps')
    if old.ndim != 2 or old.shape != ref.shape or old.shape[0] != len(lengths) or max(lengths) > old.shape[1]:
        raise ValueError('Old/ref log probabilities must share padded [B,T] shape matching lengths')
    size, width = old.shape
    mask = torch.arange(width)[None,:] < torch.tensor(lengths)[:,None]
    if not torch.isfinite(old[mask]).all() or not torch.isfinite(ref[mask]).all():
        raise ValueError('Valid old/ref log probabilities must be finite')
    old[~mask] = 0
    ref[~mask] = 0
    group_sizes = []
    seen = set()
    start = 0
    while start < size:
        label = groups[start]
        if label in seen:
            raise ValueError('Groups must be contiguous')
        seen.add(label)
        end = start + 1
        while end < size and groups[end] == label:
            end += 1
        group_sizes.append(end - start)
        start = end
    if len(set(group_sizes)) != 1 or group_sizes[0] < 2:
        raise ValueError('All groups must have the same size >= 2')
    group_size = group_sizes[0]
    if len(positions) != size:
        raise ValueError('Boundary positions must match batch size')
    max_steps = max(map(len, positions))
    boundary = torch.zeros_like(mask)
    score_mask = torch.zeros((size,max_steps), dtype=torch.bool)
    token_counts = torch.zeros((size,max_steps), dtype=torch.long)
    for i, row in enumerate(positions):
        row = [int(pos) for pos in row]
        if not row or row != sorted(set(row)) or row[0] < 0 or row[-1] != lengths[i] - 1:
            raise ValueError('Sorted unique boundaries must end at the last valid action')
        boundary[i,row] = True
        score_mask[i,:len(row)] = True
        previous = -1
        for j,pos in enumerate(row):
            token_counts[i,j] = pos - previous - 1
            previous = pos
    task = torch.zeros_like(old)
    transform_receipt = dict(used=False)
    if arm == 'P-PRM':
        if step_logits is None or len(step_logits) != size:
            raise ValueError('P-PRM requires one logit row per response')
        dtype_options = {'float32':torch.float32, 'torch.float32':torch.float32,
                         'bfloat16':torch.bfloat16, 'torch.bfloat16':torch.bfloat16}
        if step_logits_dtype not in dtype_options:
            raise ValueError('PRM transform dtype must explicitly be float32 or bfloat16')
        transform_dtype = dtype_options[step_logits_dtype]
        transform_device = torch.device(prm_transform_device)
        if transform_device.type not in ('cpu','cuda'):
            raise ValueError('Supported PRM transform devices are cpu and cuda')
        logits = torch.zeros((size,max_steps,2), dtype=transform_dtype)
        for i,row in enumerate(step_logits):
            row_tensor = _cpu(row, 'step_logits', dtype=transform_dtype)
            if row_tensor.shape != (len(positions[i]),2) or not torch.isfinite(row_tensor).all():
                raise ValueError('Step logits must be finite [steps,2] arrays aligned to boundaries')
            logits[i,:len(positions[i])] = row_tensor
        transform_logits = logits.to(device=transform_device)
        transform_mask = score_mask.to(device=transform_device)
        weighted = vendor.turn_process_reward_logits_to_reward(
            transform_logits, transform_mask, return_outcome_reward=False,
            temperature=.1, disable_weighted_reward=False)
        # Official process reward matrix has the actor's FP32 log-prob dtype.
        # Keep the PRM transform in native dtype/device until this assignment.
        task[boundary] = weighted[transform_mask].float().cpu()
        transform_receipt = dict(used=True, input_dtype=str(transform_logits.dtype),
            intermediate_and_output_dtype=str(weighted.dtype), device=str(transform_device),
            native_helper_no_fp32_promotion=transform_dtype==torch.bfloat16,
            task_matrix_dtype=str(task.dtype), task_matrix_device='cpu',
            cast_after_complete_softmax_signed_softmin=True)
        del transform_logits, transform_mask, weighted
    if outcomes is None:
        if arm == 'P-VR':
            raise ValueError('P-VR requires verifier outcomes')
        solved = torch.zeros(size, dtype=torch.float32)
        have_answers = torch.zeros(size, dtype=torch.bool)
    else:
        solved = _cpu(outcomes, 'outcomes')
        if solved.shape != (size,) or not torch.all((solved == 0) | (solved == 1)):
            raise ValueError('Outcomes must be one unchanged binary verifier value per response')
        have_answers = torch.ones(size, dtype=torch.bool)
    experience = SimpleNamespace(action_mask=mask, info=dict(
        process_reward=task, process_reward_mask=boundary, num_steps=score_mask.sum(1),
        response_length=mask.float().sum(1), num_tokens_per_step=token_counts,
        score_mask=score_mask, solved=solved, have_answers=have_answers))
    args = SimpleNamespace(advantage_estimator='rloo', n_samples_per_prompt=group_size,
        micro_rollout_batch_size=size, reward_baseline='token',
        reward_mode='VR' if arm == 'P-VR' else 'PRM', verifiable_reward_coef=1.)
    maker = SimpleNamespace(strategy=SimpleNamespace(args=args))
    _, point_rewards = vendor.process_experiences(maker, [experience])
    baseline_reward = point_rewards[0]
    kl = vendor.compute_approx_kl(old, ref, action_mask=mask,
                                 use_kl_estimator_k3=use_kl_estimator_k3)
    # PURE calls this expression directly, NOT the imported compute_reward helper.
    # The helper's terminal-scalar placement would change the process signal.
    reward = baseline_reward - float(kl_coef) * kl
    raw_returns = reward.fliplr().cumsum(dim=1).fliplr()
    returns = raw_returns.masked_fill(boundary, 0)
    advantages, norm_receipt = _normalize_once(returns, lengths)
    if not torch.all(advantages[boundary] == 0):
        raise RuntimeError('Boundary advantages must remain exactly zero')
    receipt = dict(pin=PIN, source_family='PURE official OpenRLHF', source=sources,
        arm=arm, group_size=group_size, batch_size=size, lengths=list(lengths),
        process_reward_coefficient=1., softmin_temperature=.1,
        reward_baseline='actual peer response lengths, with step lengths excluding boundary token',
        verifier_coefficient=1., verifier_mapping='solved-1; response-level LOO' if arm=='P-VR' else 'not used in PRM-only reward',
        prm_logits_read=arm=='P-PRM',
        prm_transform=transform_receipt,
        kl_coefficient=float(kl_coef), kl_estimator='k3' if use_kl_estimator_k3 else 'k1',
        kl_logps='frozen old-policy minus frozen reference log probabilities',
        kl_location='subtract beta*kl from baseline-adjusted token reward before reverse cumsum',
        compute_reward_helper_used=False, boundary_mask_before_normalization=True,
        normalization_calls=1, normalization=norm_receipt,
        loss='Pinned PolicyLoss: response token mean then batch mean, clip 0.2 by default',
        micro_loss_scaling='single-response official loss / full SignalBatch batch_size',
        separate_kl_loss=False, no_second_normalization=True,
        source_arrays_unchanged=True, all_boundary_advantages_zero=True,
        processing_device='cpu baseline/KL/returns/norm; PRM transform device explicit above',
        cpu_reduction_is_single_device_resource_adaptation=True, no_optimizer_created=True,
        limits=['Step/token alignment must be independently established by the caller.',
                'Boundary masking also suppresses ordinary EOS and the last action of truncated outputs.',
                'No claim that empty PRM scores or nonempty meaningless responses have been fixed.',
                'This is single-device integration of the signal/loss path, not a reproduction of distributed OpenRLHF runtime.'])
    tensors = dict(mask=mask,boundary=boundary,task=task,kl=kl,
                   baseline_reward=baseline_reward,reward=reward,raw_returns=raw_returns,
                   returns=returns,advantages=advantages,old_logps=old,ref_logps=ref)
    result = SignalBatch(arm=arm,lengths=lengths,groups=groups,receipt=receipt,
                         sealed=True,seal_sha256=_fingerprint(tensors),**tensors)
    result.assert_sealed()
    return result


def micro_loss(batch: SignalBatch, row: int, new_logps: torch.Tensor, *, clip_eps=.2):
    """Pinned PolicyLoss on one response, scaled by 1/B for accumulation.

    Return (differentiable scalar, JSON-serializable receipt). Sum all B returned
    losses once to match per-response reduction when the >10 ratio guard is not
    triggered. The guard acts per microbatch, exactly as an official microbatch1
    setup; this differs from applying that guard once to a full batch.
    """
    batch.assert_sealed()
    if not isinstance(row,int) or not 0 <= row < len(batch.lengths):
        raise ValueError('row must be a valid integer index')
    if not isinstance(new_logps,torch.Tensor):
        raise TypeError('new_logps must be a tensor to preserve its autograd graph')
    if new_logps.ndim == 1:
        new = new_logps.unsqueeze(0)
    elif new_logps.ndim == 2 and new_logps.shape[0] == 1:
        new = new_logps
    else:
        raise ValueError('micro_loss expects one response [T] or [1,T]')
    length = batch.lengths[row]
    if new.shape[1] not in (length,batch.mask.shape[1]):
        raise ValueError('new_logps width must be the valid row length or the padded batch width')
    # Crop padding for CPU/GPU single-row equivalence; preserve autograd.
    new = new[:,:length]
    if not torch.isfinite(new.detach()).all():
        raise ValueError('Valid new log probabilities must be finite')
    old = batch.old_logps[row:row+1,:length].to(device=new.device)
    advantages = batch.advantages[row:row+1,:length].to(device=new.device)
    action_mask = batch.mask[row:row+1,:length].to(device=new.device)
    loss, info = vendor.PolicyLoss(clip_eps=float(clip_eps))(
        new,old,advantages,action_mask=action_mask)
    scaled = loss / len(batch.lengths)
    return scaled, dict(row=row, official_unscaled_loss=float(loss.detach()),
                        scaled_loss=float(scaled.detach()), accumulation_divisor=len(batch.lengths),
                        avg_ratio=float(info['avg_ratio']), avg_advantage=float(info['avg_advantage']),
                        ratio_guard_triggered=bool(info['avg_ratio'] > 10),
                        valid_tokens=length, clip_eps=float(clip_eps),
                        normalization_calls_in_loss=0, separate_kl_loss=False,
                        device=str(new.device), boundary_advantages_zero=True,
                        sealed_input_sha256=batch.seal_sha256)
