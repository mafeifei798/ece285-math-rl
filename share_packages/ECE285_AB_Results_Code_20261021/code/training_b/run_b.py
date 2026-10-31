"""Serialized B/17 pipeline: wait for A, native check, pilot/resume, train/eval.

This is a bounded one-shot launch, not a recurring scheduler. Existing runs are
never overwritten and failures stop the pipeline with all evidence retained.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "operations"))
from common import atomic_json, file_hash, read_jsonl
from run_continuation import environment, idle_gpu, verify_evaluation


def read(path):
    return json.loads(Path(path).read_text())


def hashes():
    paths = list((ROOT / "training_b").glob("*.py")) + [ROOT / "training_b/config.json"]
    paths += [ROOT / p for p in ["training/train_baseline.py", "training/common.py", "training/reward.py",
                               "training/evaluate_policy.py", "operations/continue_baseline.py",
                               "operations/run_continuation.py", "operations/lora_preflight.py"]]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def pilot_gate(fresh, resumed, expected_hashes, bcfg):
    initial, final = read(fresh / "status.json"), read(resumed / "status.json")
    audit = read(resumed / "resume_audit.json")
    metrics = read_jsonl(fresh / "metrics.jsonl") + read_jsonl(resumed / "metrics.jsonl")
    native = read(ROOT / "receipts/B_native_prm_gate.json")
    lora = read(ROOT / "receipts/B_lora_preflight.json")
    rows = read_jsonl(fresh / "rollouts.jsonl") + read_jsonl(resumed / "rollouts.jsonl")
    checks = {
        "fresh_pilot_complete": initial["status"] == "pilot_completed" and initial["global_step"] == bcfg["pilot_steps"],
        "resume_pilot_complete": final["status"] == "pilot_completed" and final["global_step"] == bcfg["pilot_steps"] + bcfg["pilot_resume_steps"],
        "native_scores_verified": native["passed"] and native["prm_revision"] == bcfg["prm_revision"],
        "frozen_PRM": initial["prm_frozen_verified"] and final["prm_frozen_verified"],
        "real_lora_inference": lora["passed"] and lora["fresh_header_compile_passed"],
        "nonzero_actor_update": final["lora_B_abs_sum"] > 0 and audit["adapter_probe_change_l1"] > 0,
        "finite_nonzero_gradients": any(math.isfinite(m.get("grad_norm", 0)) and m.get("grad_norm", 0) > 0 for m in metrics),
        "all_resume_checks": all(audit.get(k) for k in ["adapter_restored_exactly", "optimizer_restored_exactly",
            "scheduler_restored_exactly", "python_numpy_torch_cpu_cuda_rng_restored", "data_progress_verified", "first_resumed_generation_completed"]),
        "binary_correctness_accounting": final["accounting"]["correct_completions"] == sum(r["outcome_reward"] for r in rows),
        "exact_completion_accounting": final["accounting"]["generated_completion_tokens"] == sum(r["completion_tokens"] for r in rows),
        "process_reward_exercised": any(r["reward"] > r["outcome_reward"] for r in rows),
        "invalid_reward_zero": all(r["reward"] == 0 for r in rows if not r["valid_for_process_reward"]),
        "no_verifier_exceptions": not any(r["status"].startswith("verifier_error") for r in rows),
        "sources_unchanged": hashes() == expected_hashes,
    }
    receipt = {"passed": all(checks.values()), "checks": checks, "source_hashes": expected_hashes,
               "prm_revision": bcfg["prm_revision"], "pilot_generated_tokens_extra": final["accounting"]["generated_completion_tokens"],
               "fresh_pilot": str(fresh), "resumed_pilot": str(resumed), "finished_unix": time.time(),
               "scope": "implementation, finite updates, native scoring, exact state resume; no efficacy claim"}
    atomic_json(ROOT / "receipts/B_pilot_gate.json", receipt)
    assert receipt["passed"], json.dumps(checks)


def main():
    bcfg = read(ROOT / "training_b/config.json")
    run = ROOT / "runs/B_seed17_24M"
    run.mkdir(parents=True, exist_ok=True)
    own_lock = (ROOT / "receipts/B_pipeline.lock").open("a")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (run / "controller.json").exists(), "Already launched; inspect instead of duplicating"
    expected_hashes = hashes()
    atomic_json(run / "execution_source_hashes.json", expected_hashes)
    atomic_json(run / "execution_plan.json", bcfg)
    state = {"status": "preflight", "pid": os.getpid(), "started_unix": time.time(),
             "scope": bcfg["launch_scope"], "target_completion_tokens": 24000000,
             "A_sources_unchanged": True, "replication_gate_executed": False}

    def record(**values):
        state.update(values, heartbeat_unix=time.time())
        atomic_json(run / "controller.json", state)
        atomic_json(ROOT / "receipts/B_pipeline.json", state)

    deadline = time.time() + bcfg["gpu_wait_limit_hours"] * 3600

    def wait_gpu():
        while True:
            gpu = idle_gpu()
            if gpu is not None:
                return gpu
            assert time.time() < deadline, "No idle GPU within bounded queue window"
            record(status="waiting_for_idle_gpu", child_pid=None)
            time.sleep(30)

    def execute(label, command, gpu=None):
        assert hashes() == expected_hashes, "Execution source changed after queueing"
        env = environment(gpu) if gpu is not None else os.environ.copy()
        if gpu is None:
            env["CUDA_VISIBLE_DEVICES"] = ""
        env["B_PRM_PATH"] = read(ROOT / "receipts/B_prm_assets.json")["model_path"]
        started = time.time()
        with (ROOT / "logs" / f"B_seed17_{label}.log").open("x") as log:
            child = subprocess.Popen([str(ROOT / ".venv/bin/python"), *command], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            record(status=label, child_pid=child.pid, gpu=gpu)
            while child.poll() is None:
                time.sleep(20)
                record()
            code = child.returncode
        with (run / "stages.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": label, "started_unix": started, "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started, "assigned_gpu": gpu, "exit_code": code,
                "cost_semantics": "exclusive stage wall time including startup; queue waiting excluded"}) + "\n")
        record(child_pid=None, last_child_exit_code=code)
        assert code == 0, f"{label} exited {code}; pipeline stopped, checkpoint/logs retained"

    def evaluate(label, checkpoint, output, origin):
        execute(label, ["training/evaluate_policy.py", "--output", str(output), "--adapter", str(checkpoint)], wait_gpu())
        return verify_evaluation(output, checkpoint, origin)

    record()
    try:
        # The original controller owns this lock through its 12M/24M evaluations.
        shared_lock = (ROOT / "receipts/pipeline.lock").open("a")
        while True:
            try:
                fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert time.time() < deadline, "A pipeline still owns GPU scheduling after queue deadline"
                record(status="queued_after_A_training_and_evaluation", child_pid=None, gpu=None)
                time.sleep(30)
        a_state = read(ROOT / "runs/A_seed17_24M/controller.json")
        assert a_state["status"] == "completed", "A has not completed cleanly; B will not take its recovery slot"
        assets = read(ROOT / "receipts/B_prm_assets.json")
        while assets["status"] != "completed":
            assert assets["status"] != "failed", "Pinned PRM download failed; inspect asset log"
            assert time.time() < deadline, "Pinned PRM download exceeded queue deadline"
            record(status="waiting_for_verified_prm_assets", child_pid=None, gpu=None)
            time.sleep(30)
            assets = read(ROOT / "receipts/B_prm_assets.json")
        for name, item in assets["files"].items():
            assert file_hash(Path(assets["model_path"]) / name) == item["sha256"]
        # Both branches need seed 17; no future seed/gate decision uses these results.
        execute("unit_tests", ["-m", "pytest", "-q", "training_b/test_b.py"])
        execute("native_prm_check", ["training_b/check_native_prm.py"], wait_gpu())
        pilot, resumed = ROOT / "runs/pilot_B_seed17", ROOT / "runs/pilot_B_seed17_resume"
        execute("pilot", ["training_b/train_b.py", "--run-dir", str(pilot), "--target", "6000000",
                           "--pilot-max-steps", str(bcfg["pilot_steps"])], wait_gpu())
        checkpoint = pilot / f"checkpoint-{bcfg['pilot_steps']}"
        execute("pilot_resume", ["training_b/train_b.py", "--run-dir", str(resumed), "--target", "24000000",
                    "--resume", str(checkpoint), "--prior-run", str(pilot),
                    "--pilot-max-steps", str(bcfg["pilot_steps"] + bcfg["pilot_resume_steps"])], wait_gpu())
        execute("lora_preflight", ["operations/lora_preflight.py", "--adapter", str(resumed / "final_adapter"),
                                    "--receipt", "receipts/B_lora_preflight.json"], wait_gpu())
        pilot_gate(pilot, resumed, expected_hashes, bcfg)
        fresh = ROOT / "runs/B_seed17_6M"
        execute("training_6M", ["training_b/train_b.py", "--run-dir", str(fresh), "--target", "6000000"], wait_gpu())
        first = read(fresh / "milestones.json")["milestones"]["6000000"]
        first_checkpoint = Path(first["checkpoint"])
        for name, digest in first["checkpoint_sha256"].items():
            assert file_hash(first_checkpoint / name) == digest
        first["evaluation"] = evaluate("evaluation_6000000", first_checkpoint, Path(first["evaluation_dir"]), read(fresh / "run_config.json"))
        first["status"] = "completed"
        atomic_json(first_checkpoint.parent / "milestone.json", first)
        fresh_index = read(fresh / "milestones.json")
        fresh_index["milestones"]["6000000"] = first
        atomic_json(fresh / "milestones.json", fresh_index)
        execute("training_24M", ["training_b/train_b.py", "--run-dir", str(run), "--target", "24000000",
                 "--resume", str(first_checkpoint), "--prior-run", str(fresh)], wait_gpu())
        status = read(run / "status.json")
        assert status["status"] == "completed" and status["accounting"]["generated_completion_tokens"] >= 24000000
        index = read(run / "milestones.json")
        index["milestones"]["6000000"] = first
        atomic_json(run / "milestones.json", index)
        for target in [12000000, 24000000]:
            entry = index["milestones"][str(target)]
            checkpoint = Path(entry["checkpoint"])
            for name, digest in entry["checkpoint_sha256"].items():
                assert file_hash(checkpoint / name) == digest
            entry["evaluation"] = evaluate(f"evaluation_{target}", checkpoint, Path(entry["evaluation_dir"]), read(run / "run_config.json"))
            entry["status"] = "completed"
            atomic_json(checkpoint.parent / "milestone.json", entry)
            atomic_json(run / "milestones.json", index)
        for name, digest in first["checkpoint_sha256"].items():
            assert file_hash(first_checkpoint / name) == digest
        assert hashes() == expected_hashes
        record(status="completed", finished_unix=time.time(), all_milestones_evaluated=True,
               actual_completion_tokens=status["accounting"]["generated_completion_tokens"])
    except Exception as exc:
        record(status="failed", error=str(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
