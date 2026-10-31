"""Recover the failed final evaluation without changing or rerunning training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from common import atomic_json, file_hash


def main():
    run = ROOT / "runs/A_seed17"
    locks = []
    for path in [run / "evaluation_recovery.lock", run / "controller.lock", ROOT / "receipts/pipeline.lock"]:
        lock = path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    receipt = run / "evaluation_recovery.json"
    if receipt.exists():
        raise RuntimeError("Recovery receipt exists; inspect before retrying")
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["accounting"]["generated_completion_tokens"] >= status["budget"]
    old_controller = json.loads((run / "controller.json").read_text())
    old_pipeline = json.loads((ROOT / "receipts/pipeline.json").read_text())
    assert old_controller["status"] == old_pipeline["status"] == "failed"
    for pid in [old_controller["pid"], old_controller["child_pid"], old_pipeline["pid"]]:
        cmdline = Path(f"/proc/{pid}/cmdline")
        assert not cmdline.exists() or not cmdline.read_bytes(), f"Prior process alive: {pid}"
    assert not (run / "evaluation/evaluation_summary.json").exists()
    headers = json.loads((ROOT / "receipts/python_dev_headers.json").read_text())
    assert headers["status"] == "matching_runtime_headers_prepared"
    assert headers["runtime_python"] == headers["header_version"] == "3.11.11"
    spec = json.loads((run / "run_config.json").read_text())
    cfg = ROOT / "training/baseline_config.json"
    assert file_hash(cfg) == spec["config_sha256"]
    for name in ["evaluate_policy.py", "reward.py", "common.py"]:
        assert file_hash(ROOT / "training" / name) == spec["code_sha256"][name]
    adapter_hash = file_hash(run / "final_adapter/adapter_model.safetensors")
    previous_eval = json.loads((run / "evaluation/evaluation_config.json").read_text())
    assert previous_eval["adapter_sha256"] == adapter_hash
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    busy = {line.split(",")[0].strip() for line in active.splitlines()}
    available = [int(line.split(",")[0]) for line in devices.splitlines() if line.split(",")[1].strip() not in busy]
    assert available, "No idle GPU; do not interfere with another job"
    gpu = min(available)
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "pip_proxy"}:
            env.pop(key)
    env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu),
               HF_HOME=str(ROOT / "cache/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               VLLM_CACHE_ROOT=str(ROOT / "cache/vllm"), TRITON_CACHE_DIR=str(ROOT / "cache/triton"),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / "cache/torchinductor"),
               VLLM_USE_V1="0", VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_NO_USAGE_STATS="1",
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29585 + gpu),
               DO_NOT_TRACK="1", WANDB_MODE="disabled",
               C_INCLUDE_PATH=os.pathsep.join(headers["include_paths"]))
    py = str(ROOT / ".venv/bin/python")
    state = {"status": "compiler_preflight", "pid": os.getpid(), "started_unix": time.time(),
             "gpu": gpu, "training_restarted": False, "adapter_sha256": adapter_hash,
             "evaluation_code_sha256": file_hash(ROOT / "training/evaluate_policy.py"),
             "header_receipt_sha256": file_hash(ROOT / "receipts/python_dev_headers.json")}
    atomic_json(receipt, state)
    archived = False
    try:
        with (ROOT / "logs/evaluation_compiler_preflight.log").open("w") as log:
            subprocess.run([py, "-c", "import triton; from triton.runtime import driver; print(driver.active.get_current_device())"],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
        atomic_json(run / "controller_before_evaluation_recovery.json", old_controller)
        atomic_json(ROOT / "receipts/pipeline_before_evaluation_recovery.json", old_pipeline)
        archived = True
        with (ROOT / "logs/A_seed17_evaluation_recovery.log").open("w") as log:
            child = subprocess.Popen([py, str(ROOT / "training/evaluate_policy.py"),
                                      "--output", str(run / "evaluation"), "--adapter", str(run / "final_adapter")],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(status="evaluating", child_pid=child.pid)
            atomic_json(receipt, state)
            atomic_json(run / "controller.json", {"status": "evaluating", "pid": os.getpid(),
                        "child_pid": child.pid, "arm": "A_outcome_GRPO", "seed": 17,
                        "recovery": True, "prior_attempt": "controller_before_evaluation_recovery.json"})
            atomic_json(ROOT / "receipts/pipeline.json", {"status": "final_evaluation_recovery",
                        "pid": os.getpid(), "training_status": "completed", "initial_evaluation_exit_code": 0,
                        "prior_attempt": "pipeline_before_evaluation_recovery.json"})
            code = child.wait()
        state["evaluation_exit_code"] = code
        assert code == 0, f"Evaluation exited with code {code}; inspect recovery log"
        summary = json.loads((run / "evaluation/evaluation_summary.json").read_text())
        assert summary["datasets"]["math500"]["count"] == 500
        assert summary["datasets"]["gsm8k"]["count"] == 1319
        assert file_hash(run / "final_adapter/adapter_model.safetensors") == adapter_hash
        state.update(status="completed", finished_unix=time.time())
        atomic_json(receipt, state)
        atomic_json(run / "controller.json", {**state, "arm": "A_outcome_GRPO", "seed": 17,
                    "recovery": True, "prior_attempt": "controller_before_evaluation_recovery.json"})
        atomic_json(ROOT / "receipts/pipeline.json", {"status": "completed", "pid": os.getpid(),
                    "finished_unix": time.time(), "recovered_final_evaluation": True,
                    "training_status": "completed", "initial_evaluation_exit_code": 0,
                    "final_evaluation_exit_code": 0, "prior_attempt": "pipeline_before_evaluation_recovery.json"})
    except Exception as exc:
        state.update(status="failed", error=str(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        atomic_json(receipt, state)
        if archived:
            atomic_json(run / "controller.json", {**state, "recovery": True,
                        "prior_attempt": "controller_before_evaluation_recovery.json"})
            atomic_json(ROOT / "receipts/pipeline.json", {"status": "failed", "error": str(exc),
                        "pid": os.getpid(), "recovery_receipt": str(receipt),
                        "training_status": "completed", "prior_attempt": "pipeline_before_evaluation_recovery.json"})
        raise


if __name__ == "__main__":
    main()
