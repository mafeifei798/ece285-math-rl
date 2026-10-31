"""Run only the approved first baseline, then evaluate its final saved adapter."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, atomic_json

def main():
    run = ROOT / "runs/A_seed17"
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "controller.json").exists():
        raise RuntimeError("Controller already has a receipt; inspect rather than duplicating")
    state = {"pid": os.getpid(), "status": "training", "started_unix": time.time(), "seed": 17,
             "arm": "A_outcome_GRPO", "target_completion_tokens": 6000000, "queue_other_runs": False}
    atomic_json(run / "controller.json", state)
    try:
        with (ROOT / "logs/A_seed17_train.log").open("w") as log:
            child = subprocess.Popen([sys.executable, str(ROOT / "training/train_baseline.py"), "--run-dir", str(run)],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            atomic_json(run / "controller.json", state)
            if child.wait() != 0:
                raise RuntimeError("Training failed or stopped incomplete; inspect saved diagnostics")
        state.update(status="evaluating")
        atomic_json(run / "controller.json", state)
        with (ROOT / "logs/A_seed17_evaluation.log").open("w") as log:
            child = subprocess.Popen([sys.executable, str(ROOT / "training/evaluate_policy.py"),
                "--output", str(run / "evaluation"), "--adapter", str(run / "final_adapter")],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            atomic_json(run / "controller.json", state)
            if child.wait() != 0:
                raise RuntimeError("Final evaluation failed; completed training artifacts are retained")
        state.update(status="completed", finished_unix=time.time())
    except Exception as exc:
        state.update(status="failed", error=str(exc), finished_unix=time.time())
        raise
    finally:
        atomic_json(run / "controller.json", state)

if __name__ == "__main__":
    main()
