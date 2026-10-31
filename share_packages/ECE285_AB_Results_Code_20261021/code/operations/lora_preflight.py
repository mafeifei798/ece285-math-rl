"""Check fresh Python-header compilation and real LoRA inference before training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from common import atomic_json, file_hash, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--receipt", default="receipts/lora_inference_preflight.json")
    args = parser.parse_args()
    receipt = ROOT / args.receipt
    adapter = Path(args.adapter).resolve()
    state = {"status": "running", "passed": False, "pid": os.getpid(),
             "started_unix": time.time(), "adapter": str(adapter),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "purpose": "compiler and LoRA inference plumbing only; no accuracy selection"}
    atomic_json(receipt, state)
    try:
        headers = ROOT / "receipts/python_dev_headers.json"
        include = [sysconfig.get_path("include"), sysconfig.get_path("platinclude")]
        if headers.exists():
            spec = json.loads(headers.read_text())
            runtime = ".".join(map(str, sys.version_info[:3]))
            assert spec["runtime_python"] == spec["header_version"] == runtime
            assert spec["status"] == "matching_runtime_headers_prepared"
            include = spec["include_paths"] + include
            state["header_receipt_sha256"] = file_hash(headers)
        include = list(dict.fromkeys(p for p in include if p and Path(p).is_dir()))
        compiler = shutil.which("gcc") or shutil.which("cc")
        assert compiler, "A C compiler is required for LoRA inference"
        # Compile every time: a warm Triton cache must not conceal missing headers.
        source = "#include <Python.h>\nint check_python_headers(void) { return PY_MAJOR_VERSION; }\n"
        probe = subprocess.run([compiler, "-fsyntax-only", "-x", "c", "-", *[f"-I{p}" for p in include]],
                               input=source, text=True, capture_output=True, timeout=30)
        if probe.returncode:
            raise RuntimeError("Python development headers unavailable: " + probe.stderr[-2000:])
        inherited = os.environ.get("C_INCLUDE_PATH", "")
        os.environ["C_INCLUDE_PATH"] = os.pathsep.join(include + ([inherited] if inherited else []))
        state.update(fresh_header_compile_passed=True, include_paths=include)
        cfg = json.loads((ROOT / "training/baseline_config.json").read_text())
        assets = json.loads((ROOT / "receipts/assets.json").read_text())
        manifest = json.loads((ROOT / "data/prepared/manifest.json").read_text())
        validation = ROOT / "data/prepared/validation.jsonl"
        assert file_hash(validation) == manifest["files"][validation.name]["sha256"]
        assert manifest["config_sha256"] == file_hash(ROOT / "training/baseline_config.json")
        rows = read_jsonl(validation)[:2]
        assert len(rows) == 2
        state.update(adapter_sha256=file_hash(adapter / "adapter_model.safetensors"),
                     config_sha256=manifest["config_sha256"], model_revision=assets["model_revision"],
                     validation_question_ids=[row["problem_id"] for row in rows], max_new_tokens=8)
        atomic_json(receipt, state)
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        llm = LLM(model=assets["model_path"], dtype="bfloat16", gpu_memory_utilization=0.6,
                  max_model_len=cfg["max_prompt_tokens"] + cfg["max_new_tokens"],
                  max_num_seqs=64, max_num_batched_tokens=4096, seed=285,
                  enable_lora=True, max_lora_rank=cfg["lora_rank"], enforce_eager=True)
        results = llm.generate([row["prompt"] for row in rows],
                               SamplingParams(temperature=0.0, max_tokens=8, top_p=1.0, top_k=-1),
                               lora_request=LoRARequest("preflight", 1, str(adapter)), use_tqdm=False)
        lengths = [len(result.outputs[0].token_ids) for result in results]
        assert len(lengths) == 2 and all(length > 0 for length in lengths)
        state.update(status="completed", passed=True, real_lora_generation_passed=True,
                     generated_tokens=lengths, finished_unix=time.time())
        atomic_json(receipt, state)
        print(json.dumps(state, indent=2), flush=True)
    except Exception as exc:
        state.update(status="failed", passed=False, error=str(exc),
                     traceback=traceback.format_exc(), finished_unix=time.time())
        atomic_json(receipt, state)
        raise


if __name__ == "__main__":
    main()
