"""Reconstruct B's last logged reward batch; tensor algebra only, no training."""
import collections
import json
from pathlib import Path
import sys

from trl_kernel import actual_trl_advantages
from v2core import atomic_json, audit_advantages, sha256


def main():
    root, out, trl_source = map(Path, sys.argv[1:4])
    import torch
    run = root / "runs/B_seed17_24M"
    tail = collections.deque(maxlen=32)
    total = 0
    with (run / "rollouts.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                tail.append(json.loads(line))
                total += 1
    rows = list(tail)
    assert len(rows) == 32 and len({r["optimizer_step_before_update"] for r in rows}) == 1
    assert all(len({r["problem_id"] for r in rows[i:i + 8]}) == 1 for i in range(0, 32, 8))
    recorded_failure = json.loads((run / "failure.json").read_text())
    rewards, actual = actual_trl_advantages([r["reward"] for r in rows], trl_source, "cuda:0")
    cpu = torch.as_tensor([r["reward"] for r in rows], dtype=torch.float32).view(-1, 8)
    old = ((cpu - cpu.mean(1, keepdim=True)) / (cpu.std(1, keepdim=True) + 1e-4)).reshape(-1).to(actual.device)
    old_error = None
    try:
        torch.testing.assert_close(actual, old, atol=2e-5, rtol=2e-5)
    except AssertionError as exc:
        old_error = str(exc)
    before = actual.clone()
    returned = audit_advantages(rewards, actual, receipt_path=out / "actual_failure_batch_new_guard.json",
                                problem_ids=[r["problem_id"] for r in rows])
    assert returned is actual and torch.equal(before, actual)
    close = torch.isclose(actual, old, atol=2e-5, rtol=2e-5)
    atomic_json(out / "actual_failure_batch_replay.json", {
        "source_rollout_sha256": sha256(run / "rollouts.jsonl"), "source_rows": total,
        "last_batch_source_range": [total - 32, total - 1],
        "optimizer_step_before_update": rows[0]["optimizer_step_before_update"],
        "problem_ids": [r["problem_id"] for r in rows], "logged_rewards": [r["reward"] for r in rows],
        "recorded_failure_message": recorded_failure["error"], "reconstructed_old_guard_error": old_error,
        "old_guard_failed": old_error is not None, "old_guard_mismatched_elements": int((~close).sum()),
        "max_old_cross_device_difference": float((actual - old).abs().max()),
        "new_guard_passed": True, "actual_advantage_tensor_unchanged": True,
        "new_training": False, "source_run_resumed": False,
        "limitation": "Reconstruction from saved rewards using installed TRL tensor algebra; original full GPU advantage tensor was not saved, so no claim of bitwise comparison to that missing tensor."})
    print(json.dumps({"old_guard_failed": old_error is not None, "mismatched": int((~close).sum()),
                      "max_old_error": float((actual - old).abs().max()), "new_guard_passed": True}))


if __name__ == "__main__":
    main()
