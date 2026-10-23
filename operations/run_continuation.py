"""Run only the authorized A/17 continuation, then evaluate retained milestones."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from common import atomic_json, file_hash, read_jsonl


def read(path):
    return json.loads(Path(path).read_text())


def verify_evaluation(output, adapter, origin):
    output, adapter = Path(output), Path(adapter)
    contract = read(output / "evaluation_config.json")
    assert contract["adapter_sha256"] == file_hash(adapter / "adapter_model.safetensors")
    for key in ["model_revision", "config_sha256", "data_manifest_sha256"]:
        assert contract[key] == origin[key]
    assert contract["reward_code_sha256"] == origin["code_sha256"]["reward.py"]
    summary = read(output / "evaluation_summary.json")
    for name, count in [("math500", 500), ("gsm8k", 1319)]:
        predictions = read_jsonl(output / f"{name}.jsonl")
        expected = {r["problem_id"]: r for r in read_jsonl(ROOT / "data/prepared" / f"{name}.jsonl")}
        assert len(predictions) == len({p["problem_id"] for p in predictions}) == len(expected) == count
        for p in predictions:
            assert p["question_sha256"] == expected[p["problem_id"]]["question_sha256"]
            assert p["gold"] == expected[p["problem_id"]]["answer"]
        score = summary["datasets"][name]
        assert score["count"] == count
        assert score["correct"] == sum(p["reward"] for p in predictions)
        assert score["accuracy"] == score["correct"] / count
        assert score["predictions_sha256"] == file_hash(output / f"{name}.jsonl")
    return summary


def preserved_files(parent, checkpoint):
    files = list(checkpoint.glob("*")) + list((parent / "final_adapter").glob("*"))
    files += list((parent / "evaluation").glob("*.json")) + list((parent / "evaluation").glob("*.jsonl"))
    files += [parent / name for name in ["run_config.json", "status.json", "progress.json", "rollouts.jsonl"]]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files if p.is_file()}


def assert_preserved(hashes):
    for name, digest in hashes.items():
        assert file_hash(ROOT / name) == digest, f"Preserved source changed: {name}"


def idle_gpu():
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    busy = {line.split(",")[0].strip() for line in active.splitlines()}
    available = [int(line.split(",")[0]) for line in devices.splitlines()
                 if line.split(",")[1].strip() not in busy]
    return min(available) if available else None


def environment(gpu):
    headers = read(ROOT / "receipts/python_dev_headers.json")
    assert headers["status"] == "matching_runtime_headers_prepared"
    assert headers["runtime_python"] == headers["header_version"] == "3.11.11"
    assert all(Path(p).is_dir() for p in headers["include_paths"])
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "pip_proxy"}:
            env.pop(key)
    env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HOME=str(ROOT / "cache/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
        VLLM_CACHE_ROOT=str(ROOT / "cache/vllm"), TRITON_CACHE_DIR=str(ROOT / "cache/triton"),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "cache/torchinductor"), VLLM_USE_V1="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_NO_USAGE_STATS="1",
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29587 + gpu),
        DO_NOT_TRACK="1", WANDB_MODE="disabled", C_INCLUDE_PATH=os.pathsep.join(headers["include_paths"]))
    return env


def main():
    plan_path = ROOT / "operations/continuation_24m.json"
    plan = read(plan_path)
    assert plan["target_completion_tokens"] == 24000000
    assert plan["milestones"] == [6000000, 12000000, 24000000]
    parent, source, run = (ROOT / plan[k] for k in ["origin_run", "source_checkpoint", "run_dir"])
    assert run != parent and not run.is_relative_to(parent)
    run.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in [ROOT / "receipts/pipeline.lock", parent / "controller.lock", parent / "run.lock", run / "controller.lock"]:
        lock = path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    assert not (run / "controller.json").exists(), "Continuation already attempted; inspect instead of duplicating"
    state = {"status": "preflight", "pid": os.getpid(), "started_unix": time.time(),
             "origin_run": str(parent), "source_checkpoint": str(source),
             "target_completion_tokens": plan["target_completion_tokens"],
             "scope": plan["launch_scope"], "training_from_base": False}

    def record(**values):
        state.update(values)
        atomic_json(run / "controller.json", state)
        atomic_json(ROOT / "receipts/continuation_24m.json", state)

    def execute(label, script_args, gpu):
        with (ROOT / "logs" / f"A_seed17_24M_{label}.log").open("x") as log:
            child = subprocess.Popen([str(ROOT / ".venv/bin/python"), *script_args],
                cwd=ROOT, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
            record(status=label, child_pid=child.pid, gpu=gpu)
            code = child.wait()
        record(last_child_exit_code=code)
        assert code == 0, f"{label} exited {code}; inspect its log and retained checkpoints"

    record()
    try:
        old_status, origin = read(parent / "status.json"), read(parent / "run_config.json")
        assert old_status["status"] == "completed" and old_status["global_step"] == plan["source_global_step"]
        assert old_status["accounting"]["generated_completion_tokens"] == plan["source_completion_tokens"]
        assert file_hash(source / "adapter_model.safetensors") == file_hash(parent / "final_adapter/adapter_model.safetensors")
        for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                     "completion_accounting.json", "training_args.bin"]:
            assert (source / name).is_file()
        six_summary = verify_evaluation(parent / "evaluation", source, origin)
        preserved = preserved_files(parent, source)
        atomic_json(run / "source_preservation.json", {"files_sha256": preserved, "created_unix": time.time()})
        atomic_json(run / "continuation_plan.json", plan)
        code_paths = [plan_path, ROOT / "operations/continue_baseline.py", Path(__file__),
                      ROOT / "training/evaluate_policy.py", ROOT / "training/train_baseline.py",
                      ROOT / "training/common.py", ROOT / "training/reward.py"]
        code_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in code_paths}
        atomic_json(run / "execution_source_hashes.json", code_hashes)
        atomic_json(run / "milestones.json", {"run": "A_seed17", "cumulative_completion_tokens": True,
            "evaluation_policy": plan["evaluation_policy"], "milestones": {
                "6000000": {"target_completion_tokens": 6000000, "actual_completion_tokens": plan["source_completion_tokens"],
                    "global_step": plan["source_global_step"], "status": "completed_preserved",
                    "checkpoint": str(source), "evaluation_dir": str(parent / "evaluation"),
                    "adapter_sha256": file_hash(source / "adapter_model.safetensors"), "evaluation": six_summary},
                "12000000": {"target_completion_tokens": 12000000, "status": "planned"},
                "24000000": {"target_completion_tokens": 24000000, "status": "planned"}}})
        gpu = idle_gpu()
        assert gpu is not None, "No idle GPU; do not interfere with other workloads"
        execute("training", [str(ROOT / "operations/continue_baseline.py")], gpu)
        training_status = read(run / "status.json")
        assert training_status["status"] == "completed"
        assert training_status["accounting"]["generated_completion_tokens"] >= plan["target_completion_tokens"]
        assert_preserved(preserved)
        for target in [12000000, 24000000]:
            assert_preserved(code_hashes)
            index = read(run / "milestones.json")
            entry = index["milestones"][str(target)]
            checkpoint, output = Path(entry["checkpoint"]), Path(entry["evaluation_dir"])
            for name, digest in entry["checkpoint_sha256"].items():
                assert file_hash(checkpoint / name) == digest
            # Usually the training GPU is free immediately. Never use a busy
            # GPU just because it previously belonged to this run.
            deadline = time.time() + 48 * 3600
            while (gpu := idle_gpu()) is None:
                assert time.time() < deadline, "No idle GPU for evaluation; saved models remain available"
                record(status="waiting_for_evaluation_gpu", next_milestone=target)
                time.sleep(30)
            execute(f"evaluation_{target}", [str(ROOT / "training/evaluate_policy.py"),
                    "--output", str(output), "--adapter", str(checkpoint)], gpu)
            summary = verify_evaluation(output, checkpoint, origin)
            entry.update(status="completed", evaluation=summary)
            index["milestones"][str(target)] = entry
            atomic_json(run / "milestones.json", index)
            atomic_json(checkpoint.parent / "milestone.json", entry)
        assert_preserved(preserved)
        record(status="completed", finished_unix=time.time(), all_milestones_evaluated=True,
               source_6m_files_unchanged=True)
    except Exception as exc:
        record(status="failed", error=str(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
