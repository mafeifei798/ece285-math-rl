"""Opt-in percent verifier and PRM-only CRLF adapter; no training entry point."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/reward_repair_v2_20261021"))
from v2core import (atomic_json, canonical_view, make_metadata, mixed_reward,
                    outcome_v2, read_jsonl, sha256, valid_for_prm, write_jsonl)

VERSION = "reward-repair-v3-20261021"


def repaired_outcome(text, gold, terminated, *, question, problem_id, unit_metadata):
    """Requested units are frozen from questions, never guessed from an answer."""
    metadata = unit_metadata.get(problem_id, make_metadata(question, gold, "legacy"))
    return outcome_v2(text, gold, terminated, question=question, metadata=metadata)


class CanonicalPRMAdapter:
    """Use around either existing native frozen scorer. Never edit actor outputs.

    Call repaired_outcome first, then score only valid_for_prm responses and
    combine with mixed_reward. The old mixed_reward rejects unit_* statuses.
    """
    def __init__(self, frozen_scorer):
        self.frozen_scorer = frozen_scorer

    def score(self, questions, raw_completions):
        views = [canonical_view(c, True) for c in raw_completions]
        results, cost = self.frozen_scorer.score(questions, [v["text"] for v in views])
        for raw, view, result in zip(raw_completions, views, results):
            assert result["step_count"] == len(view["steps"])
            result.update(raw_no_double_newline="\n\n" not in raw.strip(),
                          canonical_no_double_newline="\n\n" not in view["text"].strip(),
                          raw_sha256=view["raw_sha256"], step_raw_spans=view["spans"],
                          crlf_replacements=view["crlf_replacements"], repair_version=VERSION)
        return results, cost


def proposed_bounded_advantages(outcomes, process_minima, coefficient=0.2, group_size=8):
    """Diagnostic proposal only, NOT enabled in any trainer or bug-fix replay.

    A = ordinary outcome GRPO advantage + lambda * centered process probability.
    The second term is deliberately not normalized by its own tiny variance.
    Invalid responses use process_minima=0, as in the existing protocol.
    """
    import torch
    y = torch.as_tensor(outcomes, dtype=torch.float32).view(-1, group_size)
    p = torch.as_tensor(process_minima, dtype=torch.float32).view(-1, group_size)
    assert torch.isfinite(y).all() and torch.isfinite(p).all()
    assert ((y == 0) | (y == 1)).all() and ((p >= 0) & (p <= 1)).all()
    ay = (y - y.mean(1, keepdim=True)) / (y.std(1, keepdim=True) + 1e-4)
    ap = coefficient * (p - p.mean(1, keepdim=True))
    return (ay + ap).flatten(), ap.flatten()
