"""Hardware/correctness gate only. Never selects methods using test accuracy."""
import argparse
import json
import math
from pathlib import Path
from common import ROOT, atomic_json, file_hash, read_jsonl

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    run = Path(args.run_dir)
    status = json.loads((run / "status.json").read_text())
    spec = json.loads((run / "run_config.json").read_text())
    metrics = read_jsonl(run / "metrics.jsonl")
    account = status["accounting"]
    checks = {
        "completed_requested_pilot": status["status"] == "pilot_completed" and status["global_step"] == spec["pilot_steps"],
        "nonzero_parameter_update": status["lora_B_abs_sum"] > 0,
        "at_least_one_informative_group": account["mixed_reward_groups"] > 0,
        "generated_tokens_accounted": account["generated_completion_tokens"] > 0,
        "finite_gradient_logged": any(isinstance(row.get("grad_norm"), (float, int)) and math.isfinite(row["grad_norm"]) and row["grad_norm"] > 0 for row in metrics),
        "no_verifier_exceptions": not any(key.startswith("verifier_error") for key in account["reward_status_counts"]),
        "adapter_saved": (run / "final_adapter/adapter_model.safetensors").is_file(),
        "resumable_checkpoint_saved": (run / f"checkpoint-{status['global_step']}" / "optimizer.pt").is_file(),
    }
    result = {"passed": all(checks.values()), "checks": checks, "config_sha256": spec["config_sha256"],
              "critical_code_sha256": {name: spec["code_sha256"][name] for name in ["train_baseline.py", "reward.py", "common.py"]},
              "pilot_run": str(run), "accounting": account, "scope": "plumbing, finite updates, saving, accounting; not efficacy"}
    atomic_json(ROOT / "receipts/pilot_gate.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
