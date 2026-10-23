"""Corrected-only B baseline: audit the original TRL tensor without replacing it.

Only the shared reward callback fixes percent units, CRLF and valid unit-status
handling. Mixed reward, group standardization, clipping, KL and loss are original.
Integer tags check tensor preservation through shuffle, split and policy loss.
"""
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]

PINNED_TRL_SHA256 = "f217183c84be9b38b23454920eb1b914726aa3c4b9dd4c076d455bdaa67bf1b8"
PINNED_UTILS_SHA256 = "07cccc9023d60528544390d70efc0a505d2dc1e92473133c045d33f7ed976360"


class CorrectedBMixin:
    def initialize_baseline(self, ledger, *, coefficient=0.2, run_dir=None, budget=6000000):
        if coefficient != 0.2:
            raise ValueError("This version fixes coefficient=0.2; use a new version for ablations")
        self.baseline_ledger, self.baseline_coefficient = ledger, coefficient
        self.run_dir = Path(run_dir) if run_dir else None
        self.budget = budget
        self._baseline_serial = 0
        self._baseline_expected = {}
        self.baseline_loss_calls = 0
        self.baseline_loss_rows = 0
        self.baseline_generation_checks = []
        self.accounting = {k: 0 for k in ["generated_completion_tokens", "generated_completions",
            "generated_groups", "mixed_reward_groups", "correct_completions", "truncated_completions",
            "prm_input_tokens", "prm_padded_tokens", "prm_scored_completions"]}
        self.accounting.update(prm_forward_seconds=0.0, reward_status_counts={}, outcome_group_counts={})

    def _generate_and_score_completions(self, inputs):
        if self.accelerator.num_processes != 1 or self.num_generations != 8:
            raise RuntimeError("Version is pinned to single-process groups of eight")
        output = super()._generate_and_score_completions(inputs)
        rows = self.baseline_ledger.records
        previous = output["advantages"]
        if len(rows) != previous.numel() or len(rows) % 8:
            raise RuntimeError("Reward ledger does not match generation output")
        for start in range(0, len(rows), 8):
            if len({r["problem_id"] for r in rows[start:start + 8]}) != 1:
                raise RuntimeError("Question group was changed before advantage construction")
        device = previous.device
        rewards = torch.tensor([r["reward"] for r in rows], dtype=torch.float32, device=device)
        grouped = rewards.reshape(-1, 8)
        historical = ((grouped - grouped.mean(1, keepdim=True)) /
                      (grouped.std(1, keepdim=True) + 1e-4)).flatten()
        torch.testing.assert_close(previous, historical, atol=2e-5, rtol=2e-5)
        # Audit only: keep the actual pinned TRL tensor; never replace it.
        actual = previous
        assert output["advantages"] is previous
        tags = torch.arange(self._baseline_serial, self._baseline_serial + len(rows),
                            dtype=torch.int64, device=device)
        self._baseline_serial += len(rows)
        self._baseline_expected = dict(zip(tags.cpu().tolist(), actual.cpu().tolist()))
        output["baseline_trace_id"] = tags
        # TRL already logged these exact values; preserve the log unchanged.
        assert list(self._logs["advantages"])[-len(rows):] == actual.cpu().tolist()
        self.baseline_generation_checks.append({"rows": len(rows),
            "advantage_tensor_retained": output["advantages"] is previous,
            "reward_and_grpo_unchanged": True})
        if self.model.training:
            tokens = int(output["completion_mask"].sum())
            if tokens != sum(r["completion_tokens"] for r in rows):
                raise RuntimeError("Stored masks and actual completion token accounting differ")
            batch = dict(generated_completion_tokens=tokens, generated_completions=len(rows),
                generated_groups=len(rows) // 8,
                mixed_reward_groups=int((grouped.max(1).values > grouped.min(1).values).sum()),
                correct_completions=int(sum(r["outcome_reward"] for r in rows)),
                truncated_completions=sum(not r["terminated"] for r in rows), **self.baseline_ledger.cost)
            for key, value in batch.items():
                self.accounting[key] += value
            statuses = Counter(self.accounting["reward_status_counts"])
            statuses.update(r["status"] for r in rows)
            self.accounting["reward_status_counts"] = dict(statuses)
            groups = Counter(self.accounting["outcome_group_counts"])
            for start in range(0, len(rows), 8):
                total = sum(r["outcome_reward"] for r in rows[start:start + 8])
                groups["all_correct" if total == 8 else "all_wrong" if total == 0 else "mixed"] += 1
            self.accounting["outcome_group_counts"] = dict(groups)
            self.accounting["last_batch"] = batch
            if self.run_dir:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                with (self.run_dir / "rollouts.jsonl").open("a") as f:
                    for i, row in enumerate(rows):
                        f.write(json.dumps(dict(row, advantage=float(actual[i]),
                            advantage_rule="mixed_reward_original_group_std"),
                            ensure_ascii=False, allow_nan=False) + "\n")
        return output

    def _assert_baseline_loss_input(self, inputs):
        ids = inputs.get("baseline_trace_id")
        if ids is None or ids.dtype != torch.int64:
            raise RuntimeError("Corrected B provenance tag missing before policy loss")
        try:
            values = [self._baseline_expected[i] for i in ids.detach().cpu().tolist()]
        except KeyError as exc:
            raise RuntimeError("Stale/misaligned Corrected B buffer") from exc
        actual = inputs["advantages"]
        expected = torch.tensor(values, dtype=torch.float32, device=actual.device)
        if actual.dtype != torch.float32 or actual.requires_grad or not torch.equal(actual, expected):
            raise RuntimeError("Corrected B advantage changed before loss: possible second normalization")
        return actual.detach().clone()

    def _compute_loss(self, model, inputs):
        before = self._assert_baseline_loss_input(inputs)
        loss = super()._compute_loss(model, inputs)
        if not torch.equal(before, inputs["advantages"]):
            raise RuntimeError("Policy loss mutated the provided sequence advantages")
        self.baseline_loss_calls += 1
        self.baseline_loss_rows += inputs["advantages"].numel()
        return loss


def trainer_class():
    """Bind to actual pinned TRL only when explicitly constructing a trainer."""
    import importlib.metadata
    from trl import GRPOTrainer
    from trl.trainer.utils import shuffle_sequence_dict
    path = Path(inspect.getsourcefile(GRPOTrainer))
    if importlib.metadata.version("trl") != "0.23.1" or hashlib.sha256(path.read_bytes()).hexdigest() != PINNED_TRL_SHA256:
        raise RuntimeError("Installed TRL differs from audited source; recheck integration first")
    if hashlib.sha256(Path(inspect.getsourcefile(shuffle_sequence_dict)).read_bytes()).hexdigest() != PINNED_UTILS_SHA256:
        raise RuntimeError("Installed TRL buffer helpers differ from audited source")

    class CorrectedBTrainer(CorrectedBMixin, GRPOTrainer):
        def __init__(self, *args, baseline_ledger, run_dir, budget, **kwargs):
            super().__init__(*args, **kwargs)
            self.initialize_baseline(baseline_ledger, run_dir=run_dir, budget=budget)
            if (self.args.loss_type != "grpo" or self.num_generations != 8 or self.scale_rewards != "group"
                    or self.use_liger_loss or self.num_iterations != 1):
                raise RuntimeError("Unaudited objective, group, Liger, or iteration configuration")
            if self.accelerator.num_processes != 1:
                raise RuntimeError("Single-process implementation only")
    return CorrectedBTrainer
