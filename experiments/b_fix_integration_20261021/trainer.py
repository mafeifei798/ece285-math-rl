"""Opt-in independent B-fix trainer. No imports/constructors start training.

TRL computes its historical advantages first; this subclass replaces that
discarded tensor ONCE, before TRL shuffles/splits/buffers the generation batch.
An integer row tag travels with each response to enforce the exact tensor at
the policy loss. No change to ratio clipping, KL, length reduction, or optimizer.
"""
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
from bounded_advantage import compute_advantages

PINNED_TRL_SHA256 = "f217183c84be9b38b23454920eb1b914726aa3c4b9dd4c076d455bdaa67bf1b8"
PINNED_UTILS_SHA256 = "07cccc9023d60528544390d70efc0a505d2dc1e92473133c045d33f7ed976360"


class BFixMixin:
    def initialize_bfix(self, ledger, *, coefficient=0.2, run_dir=None, budget=6000000):
        if coefficient != 0.2:
            raise ValueError("This version fixes coefficient=0.2; use a new version for ablations")
        self.bfix_ledger, self.bfix_coefficient = ledger, coefficient
        self.run_dir = Path(run_dir) if run_dir else None
        self.budget = budget
        self._bfix_serial = 0
        self._bfix_expected = {}
        self.bfix_loss_calls = 0
        self.bfix_loss_rows = 0
        self.bfix_generation_checks = []
        self.accounting = {k: 0 for k in ["generated_completion_tokens", "generated_completions",
            "generated_groups", "mixed_reward_groups", "correct_completions", "truncated_completions",
            "prm_input_tokens", "prm_padded_tokens", "prm_scored_completions"]}
        self.accounting.update(prm_forward_seconds=0.0, reward_status_counts={}, outcome_group_counts={})

    def _generate_and_score_completions(self, inputs):
        if self.accelerator.num_processes != 1 or self.num_generations != 8:
            raise RuntimeError("Version is pinned to single-process groups of eight")
        output = super()._generate_and_score_completions(inputs)
        rows = self.bfix_ledger.records
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
        signal = compute_advantages(
            torch.tensor([r["outcome_reward"] for r in rows], device=device),
            [r["min_score"] for r in rows],
            valid=[r["valid_for_process_reward"] for r in rows], coefficient=self.bfix_coefficient)
        actual = signal["advantages"]
        output["advantages"] = actual
        tags = torch.arange(self._bfix_serial, self._bfix_serial + len(rows),
                            dtype=torch.int64, device=device)
        self._bfix_serial += len(rows)
        self._bfix_expected = dict(zip(tags.cpu().tolist(), actual.cpu().tolist()))
        output["bfix_trace_id"] = tags
        # Replace TRL's just-appended diagnostic advantages too; never leave
        # logs claiming the discarded mixed-normalized tensor was optimized.
        log = self._logs["advantages"]
        prior = list(log)
        log.clear()
        log.extend(prior[:-len(rows)] + actual.cpu().tolist())
        self.bfix_generation_checks.append({"rows": len(rows),
            "replaced_before_shuffle_and_split": True,
            "max_process_component": float(signal["process_component"].abs().max()),
            "max_legacy_difference": float((actual - previous).abs().max())})
        if self.model.training:
            tokens = int(output["completion_mask"].sum())
            if tokens != sum(r["completion_tokens"] for r in rows):
                raise RuntimeError("Stored masks and actual completion token accounting differ")
            batch = dict(generated_completion_tokens=tokens, generated_completions=len(rows),
                generated_groups=len(rows) // 8,
                mixed_reward_groups=int((grouped.max(1).values > grouped.min(1).values).sum()),
                correct_completions=int(sum(r["outcome_reward"] for r in rows)),
                truncated_completions=sum(not r["terminated"] for r in rows), **self.bfix_ledger.cost)
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
                            outcome_advantage=float(signal["outcome_component"][i]),
                            process_advantage=float(signal["process_component"][i]),
                            advantage_rule="outcome_GRPO_plus_bounded_process_v1"),
                            ensure_ascii=False, allow_nan=False) + "\n")
        return output

    def _assert_bfix_loss_input(self, inputs):
        ids = inputs.get("bfix_trace_id")
        if ids is None or ids.dtype != torch.int64:
            raise RuntimeError("B-fix provenance tag missing before policy loss")
        try:
            values = [self._bfix_expected[i] for i in ids.detach().cpu().tolist()]
        except KeyError as exc:
            raise RuntimeError("Stale/misaligned B-fix buffer") from exc
        actual = inputs["advantages"]
        expected = torch.tensor(values, dtype=torch.float32, device=actual.device)
        if actual.dtype != torch.float32 or actual.requires_grad or not torch.equal(actual, expected):
            raise RuntimeError("B-fix advantage changed before loss: possible second normalization")
        return actual.detach().clone()

    def _compute_loss(self, model, inputs):
        before = self._assert_bfix_loss_input(inputs)
        loss = super()._compute_loss(model, inputs)
        if not torch.equal(before, inputs["advantages"]):
            raise RuntimeError("Policy loss mutated the provided sequence advantages")
        self.bfix_loss_calls += 1
        self.bfix_loss_rows += inputs["advantages"].numel()
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

    class BFixTrainer(BFixMixin, GRPOTrainer):
        def __init__(self, *args, bfix_ledger, run_dir, budget, **kwargs):
            super().__init__(*args, **kwargs)
            self.initialize_bfix(bfix_ledger, run_dir=run_dir, budget=budget)
            if (self.args.loss_type != "grpo" or self.num_generations != 8 or self.scale_rewards != "group"
                    or self.use_liger_loss or self.num_iterations != 1):
                raise RuntimeError("Unaudited objective, group, Liger, or iteration configuration")
            if self.accelerator.num_processes != 1:
                raise RuntimeError("Single-process implementation only")
    return BFixTrainer
