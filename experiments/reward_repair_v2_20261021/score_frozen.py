"""Paired frozen-PRM inference only, on existing saved responses."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from v2core import atomic_json, encode_native, legacy_module, read_jsonl, sha256, verify_frozen_inputs, write_jsonl


def main():
    root, source, out = map(lambda x: Path(x).resolve(), sys.argv[1:4])
    if (out / "paired_scores.jsonl").exists():
        raise RuntimeError("Paired inference already complete; inspect before repeating")
    verify_frozen_inputs(out, source)
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    plan = json.loads((out / "scoring_plan.json").read_text())
    assets = json.loads((root / "receipts/B_prm_assets.json").read_text())
    assert assets["model_revision"] == "bc44d4a24bf1f7453fc3892dec25c4669e54ede5"
    for name, record in assets["files"].items():
        assert sha256(Path(assets["model_path"]) / name) == record["sha256"], name
    rows = read_jsonl(source)
    questions = {r["problem_id"]: r for r in read_jsonl(out / "question_lookup.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True, padding_side="right")
    model = AutoModelForTokenClassification.from_pretrained(assets["model_path"], local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    versions = {n: p._version for n, p in model.named_parameters()}
    assert model.config.num_labels == 2
    legacy_prm = legacy_module("prm_reward")
    started = time.monotonic()
    costs = {"forward_calls": 0, "unpadded_input_tokens": 0, "padded_input_tokens": 0,
             "repeated_forward_calls": 0}

    @torch.inference_mode()
    def forward(encoded, padded_length, repeat=False):
        inputs = tokenizer.pad([{"input_ids": e["input_ids"]} for e in encoded], padding="max_length",
                               max_length=padded_length, return_tensors="pt").to("cuda:0")
        logits = model(**inputs, use_cache=False).logits
        assert torch.isfinite(logits).all()
        probability = logits.float().softmax(-1)
        assert probability.shape[-1] == 2
        result = [probability[i, e["positions"], 1].cpu().tolist() for i, e in enumerate(encoded)]
        costs["forward_calls"] += 1
        costs["unpadded_input_tokens"] += sum(len(e["input_ids"]) for e in encoded)
        costs["padded_input_tokens"] += inputs["input_ids"].numel()
        costs["repeated_forward_calls"] += int(repeat)
        return result

    completed_path = out / "paired_scores.partial.jsonl"
    done = read_jsonl(completed_path) if completed_path.exists() else []
    assert [r["source_row"] for r in done] == plan["source_rows"][:len(done)], "Invalid partial prefix"
    assert len(done) % plan["microbatch_size"] == 0, "Inspect partial batch before resuming"
    partial_cost = out / "paired_scoring_progress.json"
    if done:
        previous = json.loads(partial_cost.read_text())
        assert previous["completed_rows"] == len(done)
        costs.update(previous["costs"])
    repeat_max = 0.0
    unchanged_input_max = 0.0
    with completed_path.open("a") as stream:
        for offset in range(len(done), len(plan["source_rows"]), plan["microbatch_size"]):
            indices = plan["source_rows"][offset:offset + plan["microbatch_size"]]
            old, new = [], []
            for index in indices:
                r = rows[index]
                question = questions[r["problem_id"]]["problem"]
                a = encode_native(tokenizer, question, r["completion"], model.config.max_position_embeddings, normalize_crlf=False)
                b = encode_native(tokenizer, question, r["completion"], model.config.max_position_embeddings, normalize_crlf=True)
                original = legacy_prm.encode_native(tokenizer, question, r["completion"], model.config.max_position_embeddings)
                assert a["input_ids"] == original["input_ids"] and a["positions"] == original["positions"]
                if r["valid_for_process_reward"]:
                    assert len(a["positions"]) == len(r["step_scores"]) == r["step_count"]
                    assert len(a["input_ids"]) == r["prm_input_tokens"]
                old.append(a)
                new.append(b)
            length = max(len(e["input_ids"]) for e in old + new)
            old_scores = forward(old, length)
            new_scores = forward(new, length)
            if (offset // plan["microbatch_size"]) % plan["repeat_control_every_n_batches"] == 0:
                repeat = forward(old, length, repeat=True)
                for a, b in zip(old_scores, repeat):
                    repeat_max = max(repeat_max, max(abs(x - y) for x, y in zip(a, b)))
                assert repeat_max <= 1e-7, "Repeated identical-input BF16 inference is not stable"
            for j, index in enumerate(indices):
                unchanged = old[j]["input_ids"] == new[j]["input_ids"] and old[j]["positions"] == new[j]["positions"]
                if unchanged:
                    difference = max(abs(a - b) for a, b in zip(old_scores[j], new_scores[j]))
                    unchanged_input_max = max(unchanged_input_max, difference)
                    assert difference <= 1e-7, "Input-identical paired rows differ"
                entry = {"source_row": index, "problem_id": rows[index]["problem_id"],
                         "input_changed": not unchanged,
                         "legacy": {k: v for k, v in old[j].items() if k != "input_ids"},
                         "normalized": {k: v for k, v in new[j].items() if k != "input_ids"}}
                for view, scores, encoding in (("legacy", old_scores[j], old[j]), ("normalized", new_scores[j], new[j])):
                    entry[view].update(step_scores=scores, min_score=min(scores), step_count=len(scores),
                                       prm_input_tokens=len(encoding["input_ids"]))
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            progress = {"complete": False, "completed_rows": offset + len(indices), "total_rows": len(plan["source_rows"]),
                        "elapsed_seconds_current_invocation": time.monotonic() - started, "costs": costs,
                        "repeat_max_difference": repeat_max, "unchanged_input_max_difference": unchanged_input_max}
            atomic_json(partial_cost, progress)
            if offset % 128 == 0:
                print(json.dumps({"stage": "frozen_paired_inference", **progress}), flush=True)
    results = read_jsonl(completed_path)
    assert [r["source_row"] for r in results] == plan["source_rows"]
    assert not model.training and all(not p.requires_grad and p.grad is None and p._version == versions[n] for n, p in model.named_parameters())
    write_jsonl(out / "paired_scores.jsonl", results)
    receipt = {"complete": True, "rows": len(results), "model_revision": assets["model_revision"],
               "weights_sha256": assets["files"]["model.safetensors"]["sha256"], "model_weight_updates": 0,
               "new_rollouts": 0, "dtype": "bfloat16", "device": torch.cuda.get_device_name(0),
               "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers")},
               "repeat_max_difference": repeat_max, "unchanged_input_max_difference": unchanged_input_max,
               "costs": costs, "elapsed_seconds_current_invocation": time.monotonic() - started,
               "all_native_prefix_checks_passed": True, "model_frozen_verified": True,
               "old_new_batch_padding_shape_matched": True,
               "cache_note": "Outside prespecified scored groups, unchanged frozen-model scores are reused from original logs; no new model changes."}
    atomic_json(out / "paired_scoring_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
