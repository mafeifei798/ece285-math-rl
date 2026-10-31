# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extracted verbatim from tanzelin430/PAPO, commit
# 7d260f0a7a6e23d289e32bbc96f80408dddfa1d5, core_algos.py:333-481.
# Local modification: add isolated imports and an identity registry decorator;
# the official function body is unchanged. CPU reference only.
from collections import defaultdict
import numpy as np
import torch

def register_adv_est(name):
    return lambda function: function

@register_adv_est("grpo_dual")
def compute_grpo_dual_objective_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    reward_prm: np.ndarray,
    lambda_process: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    acc: np.ndarray = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    PA_GRPO (Process-Aware GRPO): Dual-objective advantage computation.

    Decomposes advantage into two independently normalized components:
        A_total = A_out + lambda_process * A_proc

    - A_out: standard GRPO advantage from binary rule-based rewards
    - A_proc: process advantage from PRM scores, normalized ONLY among correct responses
    - lambda_process: fixed weight for process advantage (default 1.0 = equal weight)

    Args:
        token_level_rewards: (bs, response_length) binary rule-based rewards
        response_mask: (bs, response_length) mask for valid tokens
        index: (bs,) group IDs for each response
        reward_prm: (bs,) PRM scores (0, 0.5, 1) from non_tensor_batch
        lambda_process: weight for process advantage (default 1.0)
        epsilon: small value to avoid division by zero
        norm_adv_by_std_in_grpo: whether to normalize by std (True=GRPO, False=Dr.GRPO)
        acc: (bs,) binary accuracy from non_tensor_batch, used for correct_mask
             when reward shaping (e.g. DAPO overlong penalty) makes scores_rule non-binary.
             If None, falls back to scores_rule == 1.0.

    Returns:
        advantages: (bs, response_length) token-level advantages
        returns: (bs, response_length) same as advantages (outcome-only)
        metrics: dict of logging metrics
    """
    scores_rule = token_level_rewards.sum(dim=-1)  # (bs,)
    scores_prm = torch.tensor(reward_prm, dtype=scores_rule.dtype, device=scores_rule.device)

    # Build global binary correctness mask (unaffected by reward shaping)
    if acc is not None:
        binary_acc = torch.tensor(acc, dtype=torch.bool, device=scores_rule.device)
    else:
        binary_acc = None

    # Group responses by prompt uid
    id2indices = defaultdict(list)
    bsz = scores_rule.shape[0]
    for i in range(bsz):
        id2indices[index[i]].append(i)

    A_out = torch.zeros_like(scores_rule)
    A_proc = torch.zeros_like(scores_rule)
    A_total = torch.zeros_like(scores_rule)

    # Metrics accumulators
    process_active_count = 0
    total_group_count = 0

    with torch.no_grad():
        for uid, indices in id2indices.items():
            total_group_count += 1
            G = len(indices)
            idx_t = torch.tensor(indices, dtype=torch.long, device=scores_rule.device)
            group_rule = scores_rule[idx_t]
            group_prm = scores_prm[idx_t]

            # ─── Step 1: A_out (standard GRPO normalization) ───
            group_A_out = torch.zeros(G, dtype=scores_rule.dtype, device=scores_rule.device)
            if G > 1:
                mu_rule = group_rule.mean()
                std_rule = group_rule.std()
                if std_rule > epsilon:
                    if norm_adv_by_std_in_grpo:
                        group_A_out = (group_rule - mu_rule) / (std_rule + epsilon)
                    else:
                        group_A_out = group_rule - mu_rule

            # ─── Step 2: A_proc (process advantage among correct subset) ───
            group_A_proc = torch.zeros(G, dtype=scores_rule.dtype, device=scores_rule.device)
            if binary_acc is not None:
                correct_mask = binary_acc[idx_t]
            else:
                correct_mask = (group_rule == 1.0)
            num_correct = correct_mask.sum().item()
            process_active = False

            if num_correct >= 2:
                prm_correct = group_prm[correct_mask]
                mu_prm = prm_correct.mean()
                std_prm = prm_correct.std()

                if std_prm > epsilon:
                    process_active = True
                    process_active_count += 1
                    if norm_adv_by_std_in_grpo:
                        group_A_proc[correct_mask] = (group_prm[correct_mask] - mu_prm) / (std_prm + epsilon)
                    else:
                        group_A_proc[correct_mask] = group_prm[correct_mask] - mu_prm

            # ─── Step 3: Combine (fixed lambda, no adaptive clipping) ───
            group_A_total = group_A_out + lambda_process * group_A_proc

            # Write back to global tensors
            for j, idx in enumerate(indices):
                A_out[idx] = group_A_out[j]
                A_proc[idx] = group_A_proc[j]
                A_total[idx] = group_A_total[j]

        # Broadcast scalar advantages to token dimension
        advantages = A_total.unsqueeze(-1) * response_mask

    # Compute metrics
    proc_nonzero = A_proc[A_proc != 0]
    if binary_acc is not None:
        correct_global = binary_acc
        wrong_global = ~binary_acc
    else:
        correct_global = (scores_rule == 1.0)
        wrong_global = (scores_rule == 0.0)

    metrics = {
        "dual_objective/outcome_mean": A_out.mean().item(),
        "dual_objective/outcome_std": A_out.std().item(),
        "dual_objective/process_mean": proc_nonzero.mean().item() if proc_nonzero.numel() > 0 else 0.0,
        "dual_objective/process_std": proc_nonzero.std().item() if proc_nonzero.numel() > 1 else 0.0,
        "dual_objective/lambda_process": lambda_process,
        "dual_objective/process_active_ratio": process_active_count / max(total_group_count, 1),
        "dual_objective/total_advantage_mean": A_total.mean().item(),
        "dual_objective/total_advantage_std": A_total.std().item(),
    }

    # Per-correctness advantage breakdown
    if correct_global.any():
        A_total_correct = A_total[correct_global]
        metrics["dual_objective/correct_adv_mean"] = A_total_correct.mean().item()
        metrics["dual_objective/correct_adv_min"] = A_total_correct.min().item()
        metrics["dual_objective/correct_adv_max"] = A_total_correct.max().item()
        metrics["dual_objective/correct_Aout_mean"] = A_out[correct_global].mean().item()
        metrics["dual_objective/correct_Aproc_mean"] = A_proc[correct_global].mean().item()
    if wrong_global.any():
        metrics["dual_objective/wrong_adv_mean"] = A_total[wrong_global].mean().item()
        metrics["dual_objective/wrong_adv_min"] = A_total[wrong_global].min().item()

    return advantages, advantages, metrics

