"""Single-generation greedy evaluation; no PRM, reranking, or voting."""
import argparse
import fcntl
import json
import time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from common import ROOT, atomic_json, file_hash, read_jsonl
import sys
sys.path.insert(0, str(ROOT / "experiments/reward_repair_v3_20261021"))
from repairs import repaired_outcome

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    cli = parser.parse_args()
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "evaluation_summary.json").exists():
        raise RuntimeError("Evaluation already complete; inspect before rerunning")
    cfg = json.loads((ROOT / "training/baseline_config.json").read_text())
    assets = json.loads((ROOT / "receipts/assets.json").read_text())
    manifest = json.loads((ROOT / "data/prepared/manifest.json").read_text())
    assert manifest["config_sha256"] == file_hash(ROOT / "training/baseline_config.json")
    contract = {"model_revision": assets["model_revision"], "config_sha256": manifest["config_sha256"],
                "data_manifest_sha256": file_hash(ROOT / "data/prepared/manifest.json"),
                "reward_code_sha256": file_hash(ROOT / "training/reward.py"),
                "adapter_sha256": file_hash(Path(cli.adapter) / "adapter_model.safetensors") if cli.adapter else None}
    units_path = ROOT / "audits/reward_repair_v3_20261021/unit_metadata.json"
    unit_metadata = json.loads(units_path.read_text())
    expected = json.loads((Path(__file__).resolve().parent / "release.json").read_text())
    assert file_hash(units_path) == expected["unit_metadata_sha256"]
    contract.update(verifier_version="reward-repair-v3-20261021", unit_metadata_sha256=file_hash(units_path),
                    repair_source_sha256=file_hash(ROOT / "experiments/reward_repair_v3_20261021/repairs.py"))
    receipt = output / "evaluation_config.json"
    if receipt.exists():
        assert json.loads(receipt.read_text()) == contract, "Partial evaluation belongs to a different contract"
    else:
        atomic_json(receipt, contract)
    start = time.time()
    llm = LLM(model=assets["model_path"], dtype="bfloat16", gpu_memory_utilization=0.6,
              max_model_len=cfg["max_prompt_tokens"] + cfg["max_new_tokens"],
              max_num_seqs=64, max_num_batched_tokens=4096, seed=285, enable_lora=bool(cli.adapter),
              max_lora_rank=cfg["lora_rank"], enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"], top_p=1.0, top_k=-1)
    adapter = LoRARequest(Path(cli.adapter).parent.name, 1, str(Path(cli.adapter).resolve())) if cli.adapter else None
    summary = {"model_revision": assets["model_revision"], "adapter": cli.adapter,
               "config_sha256": manifest["config_sha256"], "greedy": True, "verifier_version": "reward-repair-v3-20261021", "unit_metadata_sha256": file_hash(units_path), "started_unix": start, "datasets": {},
               "limitation": manifest["limitation"], "prm800k_known_question_overlap": manifest["prm800k_known_question_overlap"]}
    for name in ["math500", "gsm8k"]:
        path = ROOT / "data/prepared" / f"{name}.jsonl"
        assert file_hash(path) == manifest["files"][path.name]["sha256"]
        rows = read_jsonl(path)
        existing = read_jsonl(output / f"{name}.jsonl") if (output / f"{name}.jsonl").exists() else []
        by_id = {row["problem_id"]: row for row in existing}
        todo = [row for row in rows if row["problem_id"] not in by_id]
        with (output / f"{name}.jsonl").open("a") as stream:
            for offset in range(0, len(todo), 64):
                batch = todo[offset:offset + 64]
                answers = llm.generate([row["prompt"] for row in batch], params, lora_request=adapter, use_tqdm=False)
                for row, result in zip(batch, answers):
                    answer = result.outputs[0]
                    judged = repaired_outcome(answer.text, row["answer"], answer.finish_reason == "stop",
                        question=row["problem"], problem_id=row["problem_id"], unit_metadata=unit_metadata)
                    record = {"problem_id": row["problem_id"], "question_sha256": row["question_sha256"],
                              "gold": row["answer"], "completion": answer.text, "generated_tokens": len(answer.token_ids),
                              "finish_reason": answer.finish_reason, **judged}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    by_id[row["problem_id"]] = record
                stream.flush()
                atomic_json(output / "progress.json", {"dataset": name, "done": len(by_id), "total": len(rows),
                            "elapsed_seconds": time.time() - start})
                print(name, len(by_id), "/", len(rows), flush=True)
        assert len(by_id) == len(rows)
        summary["datasets"][name] = {"count": len(rows), "correct": sum(row["reward"] for row in by_id.values()),
            "accuracy": sum(row["reward"] for row in by_id.values()) / len(rows),
            "generated_tokens": sum(row["generated_tokens"] for row in by_id.values()),
            "truncated": sum(row["status"] == "truncated" for row in by_id.values()),
            "predictions_sha256": file_hash(output / f"{name}.jsonl")}
    summary.update(finished_unix=time.time(), elapsed_seconds=time.time() - start)
    atomic_json(output / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
