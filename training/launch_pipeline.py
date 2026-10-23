"""Bounded setup -> checks -> pilot -> one formal A run; stop on any failed gate."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import ROOT, atomic_json

def main():
    lock = (ROOT / "receipts/pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = ROOT / "receipts/pipeline.json"
    if receipt.exists():
        raise RuntimeError("Pipeline receipt exists: inspect previous execution before launching again")
    state = {"pid": os.getpid(), "status": "waiting_for_environment", "started_unix": time.time(), "children": {}}
    def save():
        atomic_json(receipt, state)
    save()
    env = os.environ.copy()
    proxy = os.environ.get("ECE285_HTTP_PROXY", "http://127.0.0.1:7890")
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "pip_proxy"}:
            env.pop(key)
    env.update(PYTHONUNBUFFERED="1", HF_HOME=str(ROOT / "cache/huggingface"),
               HF_ENDPOINT="https://huggingface.co", HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               HTTPS_PROXY=proxy, HTTP_PROXY=proxy, https_proxy=proxy, http_proxy=proxy,
               NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               VLLM_CACHE_ROOT=str(ROOT / "cache/vllm"), TRITON_CACHE_DIR=str(ROOT / "cache/triton"),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / "cache/torchinductor"),
               VLLM_USE_V1="0", VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_NO_USAGE_STATS="1",
               DO_NOT_TRACK="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled")
    py = str(ROOT / ".venv/bin/python")
    def start(name, command, gpu=None):
        child_env = env.copy()
        if gpu is not None:
            child_env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                             MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29585 + gpu))
        with (ROOT / "logs" / f"{name}.log").open("w") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        state["children"][name] = child.pid
        save()
        return child
    def checked(name, command, gpu=None):
        child = start(name, command, gpu=gpu)
        if child.wait() != 0:
            raise RuntimeError(f"{name} failed; see its project log")
    try:
        deadline = time.monotonic() + 3600
        while True:
            install = json.loads((ROOT / "receipts/environment_install.json").read_text())
            if install["status"] == "failed":
                raise RuntimeError("Environment installation failed")
            if install["status"] == "completed":
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Environment setup wait exceeded one hour; no training launched")
            time.sleep(10)
        assert json.loads((ROOT / "receipts/official_asset_verification.json").read_text())["status"] == "verified_against_official_metadata"
        headers = ROOT / "receipts/python_dev_headers.json"
        if headers.exists():
            header_spec = json.loads(headers.read_text())
            assert header_spec["status"] == "matching_runtime_headers_prepared"
            inherited = env.get("C_INCLUDE_PATH", "")
            env["C_INCLUDE_PATH"] = os.pathsep.join(header_spec["include_paths"] + ([inherited] if inherited else []))
        state["status"] = "data_and_reward_checks"; save()
        checked("reward_tests", [py, "-m", "pytest", "-q", "training/test_reward.py"])
        checked("prepare_data", [py, "training/prepare_data.py"])
        # Respect other jobs: both selected cards must still be idle at admission.
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        selected = {row.split(",")[1].strip() for row in devices.splitlines() if row.split(",")[0].strip() in {"0", "1"}}
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if any(row.split(",")[0].strip() in selected for row in active.splitlines()):
            raise RuntimeError("Selected GPU now has another compute process; no pilot launched")
        state["status"] = "pilot_and_initial_evaluation"; save()
        initial = start("initial_policy_evaluation", [py, "training/evaluate_policy.py", "--output", "runs/initial_policy"], gpu=1)
        pilot = start("pilot_A_seed17", [py, "training/train_baseline.py", "--run-dir", "runs/pilot_A_seed17",
                      "--pilot-steps", "3", "--wall-limit-hours", "0.5"], gpu=0)
        if pilot.wait() != 0:
            raise RuntimeError("Pilot failed; formal training not launched")
        checked("pilot_gate", [py, "training/check_pilot.py", "--run-dir", "runs/pilot_A_seed17"])
        state["status"] = "lora_inference_preflight"; save()
        checked("lora_inference_preflight", [py, "operations/lora_preflight.py", "--adapter",
                "runs/pilot_A_seed17/final_adapter"], gpu=0)
        # Wait only for the first evaluation batch, not a favorable accuracy value.
        state["status"] = "waiting_for_initial_evaluation_first_batch"; save()
        deadline = time.monotonic() + 1800
        while not (ROOT / "runs/initial_policy/progress.json").exists():
            if initial.poll() is not None:
                raise RuntimeError("Initial evaluation exited before saving its first batch")
            if time.monotonic() >= deadline:
                raise RuntimeError("Initial evaluation first-batch timeout; formal training not launched")
            time.sleep(5)
        state["status"] = "formal_A_seed17"; save()
        formal = start("A_seed17_controller", [py, "training/run_baseline.py"], gpu=0)
        formal_code = formal.wait()
        initial_code = initial.wait()
        state.update(formal_exit_code=formal_code, initial_evaluation_exit_code=initial_code)
        if formal_code or initial_code:
            raise RuntimeError("A formal stage failed; inspect retained logs and checkpoints")
        state.update(status="completed", finished_unix=time.time()); save()
    except Exception as exc:
        state.update(status="failed", error=str(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        save()
        raise

if __name__ == "__main__":
    main()
