"""One-shot locked Bprime pilot, 6M training and unchanged greedy evaluation."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from support import HERE, ROOT, atomic, read, sha, source_hashes, preservation_snapshot, assert_preserved
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "operations"))
from common import read_jsonl
from run_continuation import environment, idle_gpu, verify_evaluation


def gate(pilot, hashes, preserved):
    status = read(pilot / "status.json")
    rows = read_jsonl(pilot / "rollouts.jsonl")
    metrics = read_jsonl(pilot / "metrics.jsonl")
    audits = [read(p) for p in (pilot / "advantage_audits").glob("*.json")]
    checks = {
        "two_optimizer_steps": status["status"] == "pilot_completed" and status["global_step"] == 2,
        "actor_updated": status["actor_adapter_changed"] and status["lora_B_abs_sum"] > 0,
        "frozen_PRM": status["prm_frozen_verified"],
        "finite_nonzero_gradient": any(math.isfinite(m.get("grad_norm", 0)) and m.get("grad_norm", 0) > 0 for m in metrics),
        "all_metrics_finite": all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, (int, float))),
        "same_initial_sampling": all(read(pilot / "initial_sampling_match.json").values()),
        "native_weights_verified": read(pilot / "prm_loading.json")["required_scoring_tensors_exactly_matched"] == 342,
        "advantage_audits_passed": len(audits) == 2 and all(a["passed"] for a in audits),
        "exact_token_accounting": sum(r["completion_tokens"] for r in rows) == status["accounting"]["generated_completion_tokens"],
        "exact_correctness_accounting": sum(r["outcome_reward"] for r in rows) == status["accounting"]["correct_completions"],
        "reward_formula": all(abs(r["reward"] - (r["outcome_reward"] + .2 * min(r["step_scores"]) if r["valid_for_process_reward"] else 0)) < 1e-12 for r in rows),
        "process_reward_exercised": any(r["reward"] > r["outcome_reward"] for r in rows),
        "no_verifier_exceptions": not any(r["status"].startswith("verifier_error") for r in rows),
        "source_hashes_match": hashes == source_hashes(),
        "originals_preserved": assert_preserved(preserved)["passed"],
    }
    receipt = {"passed": all(checks.values()), "checks": checks, "source_hashes": hashes,
        "pilot_extra_tokens": status["accounting"]["generated_completion_tokens"],
        "pilot_progress": read(pilot / "progress.json"), "finished_unix": time.time(),
        "scope": "Engineering/finite-update check only; not an efficacy claim or hyperparameter selection"}
    assert receipt["passed"], json.dumps(checks)
    return receipt


def main():
    plan = read(HERE / "protocol.json")
    preflight = ROOT / "audits/b_qwen7b_6m_20261021_preflight/summary.json"
    assert read(preflight)["passed"], "Pre-launch regression checks have not passed"
    run, pilot = ROOT / plan["run"], ROOT / plan["pilot"]
    run.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in [ROOT / "receipts/pipeline.lock", run / "controller.lock"]:
        lock = path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    assert not (run / "controller.json").exists(), "Controller already exists; inspect before recovery"
    assert not pilot.exists(), "Pilot already exists; inspect before recovery"
    hashes, preserved = source_hashes(), preservation_snapshot()
    assert hashes == read(ROOT / "audits/qwen7b_offline_20261021/ece285_b_qwen7b_expected_sources.json")
    atomic(run / "execution_source_hashes.json", hashes)
    atomic(run / "originals_before.json", preserved)
    atomic(run / "execution_plan.json", plan)
    state = {"status": "preflight", "pid": os.getpid(), "started_unix": time.time(),
        "target_completion_tokens": 6000000, "scope": plan["scope"],
        "preflight_receipt_sha256": sha(preflight)}

    def record(**values):
        state.update(values, heartbeat_unix=time.time())
        atomic(run / "controller.json", state)

    def execute(label, command, gpu):
        assert hashes == source_hashes()
        assert_preserved(preserved)
        started = time.time()
        log_path = ROOT / "logs" / (run.name + "_" + label + ".log")
        with log_path.open("x") as log:
            child = subprocess.Popen([str(ROOT / ".venv/bin/python"), *command], cwd=ROOT,
                env=environment(gpu), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            record(status=label, child_pid=child.pid, gpu=gpu, log=str(log_path))
            while child.poll() is None:
                time.sleep(15)
                record()
            code = child.returncode
        with (run / "stages.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": label, "start": started, "finish": time.time(),
                "wall_seconds": time.time() - started, "gpu": gpu, "exit_code": code}) + "\n")
        record(last_child_exit_code=code, child_pid=None)
        assert code == 0, f"{label} exited {code}; saved evidence retained, no automatic retraining"

    record()
    try:
        gpu = idle_gpu()
        assert gpu is not None, "No idle GPU; do not displace other jobs"
        execute("pilot", [str(HERE / "train.py"), "--pilot"], gpu)
        receipt = gate(pilot, hashes, preserved)
        atomic(run / "pilot_gate.json", receipt)
        gpu = idle_gpu()
        assert gpu is not None, "GPU still occupied; inspect before training"
        execute("training_6M", [str(HERE / "train.py")], gpu)
        status = read(run / "status.json")
        assert status["status"] == "completed" and status["accounting"]["generated_completion_tokens"] >= 6000000
        index = read(run / "milestones.json")
        entry = index["milestones"]["6000000"]
        checkpoint = Path(entry["checkpoint"])
        for filename, digest in entry["checkpoint_sha256"].items():
            assert sha(checkpoint / filename) == digest
        # The training process must have released this GPU before evaluation.
        deadline = time.monotonic() + 120
        while (eval_gpu := idle_gpu()) is None and time.monotonic() < deadline:
            time.sleep(5)
        assert eval_gpu is not None, "Evaluation GPU unavailable; checkpoint preserved"
        execute("evaluation_6M", [str(ROOT / "training/evaluate_policy.py"),
            "--output", entry["evaluation_dir"], "--adapter", str(checkpoint)], eval_gpu)
        entry["evaluation"] = verify_evaluation(entry["evaluation_dir"], checkpoint, read(run / "run_config.json"))
        entry["status"] = "completed"
        index["milestones"]["6000000"] = entry
        atomic(run / "milestones.json", index)
        atomic(checkpoint.parent / "milestone.json", entry)
        atomic(run / "originals_after.json", preservation_snapshot())
        preservation = assert_preserved(preserved)
        assert hashes == source_hashes()
        record(status="completed", finished_unix=time.time(), preservation=preservation,
            all_evaluations_completed=True, no_C_D_launched=True)
    except Exception as exc:
        record(status="failed", error=str(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
