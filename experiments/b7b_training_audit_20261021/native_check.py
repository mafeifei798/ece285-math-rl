"""Replay fixed existing PRM batches; no actor, generation, optimizer or training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
out = root / "audits/b7b_training_audit_20261021"
read = lambda p: json.loads(Path(p).read_text())
read_rows = lambda p: [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
selection = read(out / "sample_manifest.json")
rows = read_rows(root / "runs/B_Qwen7B_seed17_6M/rollouts.jsonl")
questions = {r["problem_id"]: r["problem"] for r in read_rows(root / "data/prepared/train.jsonl")}
devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
busy = {line.split(",")[0].strip() for line in active.splitlines()}
idle = [line.split(",")[0].strip() for line in devices.splitlines() if line.split(",")[1].strip() not in busy]
assert idle, "No idle GPU: do not displace any worker"
os.environ.update(CUDA_VISIBLE_DEVICES=idle[0], HF_HOME=str(root / "cache/huggingface"),
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, str(root / "experiments/b_qwen7b_6m_20261021"))
from frozen_prm import FrozenQwenPRM
from core import encode_qwen
from support import sha
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.mem_get_info()[0] > 55 * 1024**3
started = time.monotonic()
assert sha(root / "runs/B_Qwen7B_seed17_6M/rollouts.jsonl") == selection["source_sha256"]
scorer = FrozenQwenPRM(root / "models/qwen25_math_prm_7b", receipt_dir=out / "replay_loader", microbatch_size=4)
native = {"scored_rows": 0, "native_steps": 0, "count_mismatches": 0, "token_length_mismatches": 0,
          "marker_id": scorer.tokenizer.encode("<extra_0>", add_special_tokens=False), "maximum_tokens": 0}
for row in rows:
    if not row["valid_for_process_reward"]:
        assert row["step_scores"] == [] and row["min_score"] == 0
        continue
    e = encode_qwen(scorer.tokenizer, questions[row["problem_id"]], row["completion"], normalize=False)
    n = len(e["steps"])
    native["scored_rows"] += 1
    native["native_steps"] += n
    native["count_mismatches"] += n != row["step_count"] or n != len(row["step_scores"])
    native["token_length_mismatches"] += len(e["input_ids"]) != row["prm_input_tokens"]
    native["maximum_tokens"] = max(native["maximum_tokens"], len(e["input_ids"]))
    assert len(e["positions"]) == n and len(e["input_ids"]) <= scorer.max_length
prefix_checks = []
for index in selection["rows"]:
    row = rows[index]
    if not row["valid_for_process_reward"]:
        continue
    q = questions[row["problem_id"]]
    full = encode_qwen(scorer.tokenizer, q, row["completion"], normalize=False)
    for i in range(1, len(full["steps"]) + 1):
        prefix = encode_qwen(scorer.tokenizer, q, "\n\n".join(full["steps"][:i]), normalize=False)
        a, b = full["positions"][i - 1], prefix["positions"][-1]
        assert a == b and full["input_ids"][:a + 1] == prefix["input_ids"][:b + 1]
        prefix_checks.append({"source_row": index, "step": i, "position": a, "passed": True})
selected_steps = sorted({rows[i]["optimizer_step_before_update"] for i in selection["rows"]})
by_step = {}
for index, row in enumerate(rows):
    if row["optimizer_step_before_update"] in selected_steps:
        by_step.setdefault(row["optimizer_step_before_update"], []).append((index, row))
replayed, maximum = [], 0.0
repeat_maximum = 0.0
for stage, (step, batch) in enumerate(sorted(by_step.items())):
    assert len(batch) == 32
    valid = [(i, r) for i, r in batch if r["valid_for_process_reward"]]
    q = [questions[r["problem_id"]] for _, r in valid]
    c = [r["completion"] for _, r in valid]
    scores, _ = scorer.score(q, c)
    if stage == 0:
        repeated, _ = scorer.score(q, c)
        repeat_maximum = max(abs(a-b) for r,s in zip(scores,repeated) for a,b in zip(r["step_scores"],s["step_scores"]))
        assert repeat_maximum <= 1e-7
    for (index, old), new in zip(valid, scores):
        diffs = [abs(a-b) for a,b in zip(old["step_scores"], new["step_scores"])]
        maximum = max(maximum, max(diffs))
        replayed.append({"source_row": index, "selected_for_review": index in selection["rows"],
            "max_step_difference": max(diffs), "min_score_difference": abs(new["min_score"]-old["min_score"]),
            "threshold_0p5_flips": sum((a<.5)!=(b<.5) for a,b in zip(old["step_scores"],new["step_scores"])),
            "step_scores": new["step_scores"], "old_step_scores": old["step_scores"]})
    print("replayed_optimizer_step", step, "valid_rows", len(valid), flush=True)
scorer.assert_frozen()
sys.path.insert(0, str(root / "training"))
from reward import outcome
outcome_checks = []
for index in selection["rows"]:
    row = rows[index]
    now = outcome(row["completion"], row["gold"], row["terminated"])
    outcome_checks.append({"source_row": index, "logged_status": row["status"],
        "replayed_status": now["status"], "logged_reward": row["outcome_reward"],
        "replayed_reward": now["reward"], "matches": now["reward"] == row["outcome_reward"] and now["status"] == row["status"]})
manifest = read(out / "input_manifest.json")
for rel, receipt in manifest["files"].items():
    assert sha(root / rel) == receipt["sha256"], rel
receipt = {"completed": True, "scope": "Frozen PRM inference on existing full batches and tokenizer/verifier replay; no actor, new rollouts or training",
    "physical_gpu": idle[0], "native_all_rows": native, "selected_prefix_checks": prefix_checks,
    "replayed_optimizer_batches": len(by_step), "replayed_valid_rows": len(replayed),
    "max_score_difference_from_training": maximum, "identical_batch_repeat_maximum": repeat_maximum,
    "threshold_0p5_flips": sum(r["threshold_0p5_flips"] for r in replayed),
    "replayed_rows": replayed, "outcome_checks": outcome_checks, "model_frozen": True,
    "original_input_files_unchanged": True, "wall_seconds": time.monotonic() - started,
    "interpretation": "Compare original batch composition and microbatch4. Minor BF16 kernel differences are numerical, not automatic proof of a scoring defect."}
(out / "native_replay.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
assert native["count_mismatches"] == native["token_length_mismatches"] == 0
assert all(r["matches"] for r in outcome_checks)
print(json.dumps({k: v for k,v in receipt.items() if k not in ["selected_prefix_checks","replayed_rows","outcome_checks"]},indent=2))
