"""Fixed PURE OpenRLHF AST extracts; no complete official module is imported.

Pin 41f9363b8cc4af044b7fc5f7f9bbbf62139e9553. Extraction preserves node AST.
The single-process rank-zero shim only supports PolicyLoss's diagnostic print.
No distributed initialization, actor, optimizer or CUDA context is created.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class _SingleProcessRankZero:
    @staticmethod
    def get_rank():
        return 0

dist = _SingleProcessRankZero()

def right_padding(input_ids, pad_token_id):
    max_len = max([i.size(-1) for i in input_ids])
    for i, input_idx in enumerate(input_ids):
        real_idx = input_idx
        if real_idx.ndim == 1:
            real_idx = real_idx.unsqueeze(0)
        assert real_idx.ndim == 2
        input_ids[i] = torch.cat([real_idx, torch.tensor([pad_token_id] * (max_len - real_idx.size(-1)), dtype=real_idx.dtype, device=real_idx.device).unsqueeze(0).repeat(real_idx.size(0), 1)], dim=-1)
    input_ids = torch.cat(input_ids, dim=0)
    return input_ids

def turn_process_reward_logits_to_reward(logits, reward_mask, return_outcome_reward=False, temperature=0.1, disable_weighted_reward=False):
    assert logits.ndim == 3 and logits.size(-1) == 2 and (reward_mask.ndim == 2) and (reward_mask.shape == logits.shape[:2])
    softmax_logits = logits.softmax(dim=-1)
    process_reward = softmax_logits[..., 1] - softmax_logits[..., 0]
    process_reward = process_reward.masked_fill(~reward_mask, 0)
    if not disable_weighted_reward:
        weight = torch.softmax(-process_reward.masked_fill(~reward_mask, float('inf')) / temperature, dim=-1)
    else:
        weight = torch.softmax(torch.ones_like(process_reward).masked_fill(~reward_mask, -float('inf')), dim=-1)
    process_reward = weight * process_reward
    if return_outcome_reward:
        outcome_reward = process_reward.sum(dim=-1)
        return (process_reward, outcome_reward)
    return process_reward

def compute_approx_kl(log_probs: torch.Tensor, log_probs_base: torch.Tensor, action_mask: Optional[torch.Tensor]=None, use_kl_estimator_k3: bool=False) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """
    log_ratio = log_probs.float() - log_probs_base.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    if use_kl_estimator_k3:
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio
    return log_ratio

def compute_reward(r: Union[torch.Tensor, float], kl_coef: float, kl: Union[torch.Tensor, list[torch.Tensor]], action_mask: Optional[torch.Tensor]=None, num_actions: Optional[Union[int, list[int]]]=None, reward_clip_range: Tuple[float, float]=None) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0
    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])
    if action_mask is not None:
        kl_reward = -kl_coef * kl
        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))
        reward = last_reward + kl_reward
    else:
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            kl_reward[action_len - 1] += r[i]
            reward.append(kl_reward)
    return reward

def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int=None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)

@torch.no_grad()
def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
    """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
    args = self.strategy.args
    assert args.advantage_estimator == 'rloo'
    output_process_rewards = []
    process_rewards = []
    original_lengths = []
    have_answers = []
    solveds = []
    reward_masks = []
    num_steps = []
    response_lengths = []
    num_tokens_per_steps = []
    score_masks = []
    action_masks = []
    for experience in experiences:
        process_reward = experience.info['process_reward'].clone()
        original_length = process_reward.size(1)
        have_answer = experience.info['have_answers']
        solved = experience.info['solved']
        reward_mask = experience.info['process_reward_mask']
        num_step = experience.info['num_steps']
        response_length = experience.info['response_length']
        num_tokens_per_step = experience.info['num_tokens_per_step']
        score_mask = experience.info['score_mask']
        action_mask = experience.action_mask
        original_lengths.append(original_length)
        process_rewards.append(process_reward)
        have_answers.append(have_answer)
        solveds.append(solved)
        reward_masks.append(reward_mask)
        num_steps.append(num_step)
        response_lengths.append(response_length)
        num_tokens_per_steps.append(num_tokens_per_step)
        score_masks.append(score_mask)
        action_masks.append(action_mask)
    process_rewards = right_padding(process_rewards, 0)
    have_answers = torch.cat(have_answers)
    solveds = torch.cat(solveds)
    reward_masks = right_padding(reward_masks, False)
    num_steps = torch.cat(num_steps)
    response_lengths = torch.cat(response_lengths)
    num_tokens_per_steps = right_padding(num_tokens_per_steps, 0)
    score_masks = right_padding(score_masks, False)
    action_masks = right_padding(action_masks, False)
    num_data = solveds.numel()
    process_rewards = process_rewards.reshape(-1, args.n_samples_per_prompt, process_rewards.size(1))
    outcome_rewards = process_rewards.sum(-1)
    outcome_rewards_info = outcome_rewards.flatten()
    prediction = outcome_rewards_info.clone().sign()
    prediction[prediction == -1] = 0
    match = torch.full_like(solveds, -1.0, dtype=torch.float)
    match[have_answers] = (prediction[have_answers] == solveds[have_answers]).float()
    if args.reward_baseline == 'step':
        num_steps = num_steps.reshape(-1, args.n_samples_per_prompt)
        reward_masks = reward_masks.reshape(-1, args.n_samples_per_prompt, reward_masks.size(1))
        average_process_reward = outcome_rewards / num_steps
        process_reward_baseline = (average_process_reward.sum(-1, keepdim=True) - average_process_reward) / (args.n_samples_per_prompt - 1)
        process_rewards = (process_rewards - process_reward_baseline.unsqueeze(-1)).masked_fill(~reward_masks, 0)
        process_rewards = process_rewards.reshape(num_data, -1)
    else:
        response_lengths = response_lengths.reshape(-1, args.n_samples_per_prompt)
        average_reward_per_token = (outcome_rewards.sum(-1, keepdim=True) - outcome_rewards) / (response_lengths.sum(-1, keepdim=True) - response_lengths)
        process_rewards = process_rewards.reshape(num_data, -1)
        average_reward_per_token = average_reward_per_token.flatten()
        for i in range(num_data):
            process_rewards[i, reward_masks[i]] -= average_reward_per_token[i] * num_tokens_per_steps[i, score_masks[i]]
    if 'VR' in args.reward_mode:
        verifier_reward = solveds - 1
        verifier_reward = verifier_reward.reshape(-1, args.n_samples_per_prompt)
        verifier_reward_baseline = (verifier_reward.sum(-1, keepdim=True) - verifier_reward) / (args.n_samples_per_prompt - 1)
        verifier_reward = verifier_reward - verifier_reward_baseline
        verifier_reward = verifier_reward.flatten()
        eos_indices = action_masks.size(1) - 1 - action_masks.long().fliplr().argmax(1)
        process_rewards[torch.arange(num_data), eos_indices] += verifier_reward * args.verifiable_reward_coef
    for idx, experience in enumerate(experiences):
        start_idx = idx * args.micro_rollout_batch_size
        end_idx = (idx + 1) * args.micro_rollout_batch_size
        output_process_rewards.append(process_rewards[start_idx:end_idx, :original_lengths[idx]])
        experience.info['match'] = match[start_idx:end_idx]
        experience.info['reward'] = outcome_rewards_info[start_idx:end_idx]
    return (experiences, output_process_rewards)

def normalize(self, attribute: str, strategy) -> None:
    assert attribute == 'advantages'
    items = []
    action_masks = []
    step_separator_mask = []
    for item in self:
        items.append(getattr(item, attribute))
        action_masks.append(item.action_mask)
        step_separator_mask.append(getattr(item, attribute) == 0)
    items_vector = torch.cat(items).float().flatten()
    if action_masks[0] is None:
        action_masks_vector = 1
        num_actions = items_vector.numel()
    else:
        action_masks_vector = torch.cat(action_masks).flatten()
        num_actions = action_masks_vector.sum()
        if self.mask_separator_adv:
            step_separator_mask_vector = torch.cat(step_separator_mask).flatten()
            num_actions -= step_separator_mask_vector.sum()
    sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
    all_sum, all_count = strategy.all_reduce(sum_and_count, 'sum')
    mean = all_sum / all_count
    std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
    all_std = strategy.all_reduce(std, 'sum')
    rstd = (all_std / all_count).clamp(min=1e-08).rsqrt()
    for i, item in enumerate(self):
        target = (items[i] - mean) * rstd
        setattr(item, attribute, target.masked_fill(step_separator_mask[i], 0) if self.mask_separator_adv else target)

class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float=0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, action_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        with torch.no_grad():
            avg_ratio = masked_mean(ratio, action_mask, dim=-1).mean()
            avg_advantage = masked_mean(advantages, action_mask, dim=-1).mean()
        if avg_ratio > 10:
            loss *= 0.0
            if dist.get_rank() == 0:
                print('ratio is too high, skipping update')
        info = {'avg_ratio': avg_ratio, 'avg_advantage': avg_advantage}
        return (loss, info)

def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)
