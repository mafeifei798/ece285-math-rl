"""Build immutable original-MATH train/validation and evaluation manifests."""
import json
from pathlib import Path
import random
import time
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from common import ROOT, atomic_json, file_hash, last_boxed, prompt_for, question_key, read_jsonl, write_jsonl
from reward import outcome

def main():
    config = json.loads((ROOT / "training/baseline_config.json").read_text())
    assets = json.loads((ROOT / "receipts/assets.json").read_text())
    output = ROOT / "data/prepared"
    if (output / "manifest.json").exists():
        old = json.loads((output / "manifest.json").read_text())
        for name, item in old["files"].items():
            assert file_hash(output / name) == item["sha256"], name
        assert old["config_sha256"] == file_hash(ROOT / "training/baseline_config.json")
        print("Existing prepared data and hashes verified; not rebuilding", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True)
    def load(key, subset=None, split="test"):
        spec = assets["datasets"][key]
        return load_dataset(spec["id"], subset, split=split, revision=spec["revision"],
                            cache_dir=str(ROOT / "cache/datasets"))
    train_raw = []
    for subject in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]:
        ds = load("math", subject, "train")
        for index, row in enumerate(ds):
            train_raw.append({"problem_id": f"math/train/{subject}/{index}", "problem": row["problem"],
                              "answer": last_boxed(row["solution"]), "subject": row["type"], "level": row["level"]})
    assert len(train_raw) == 7500, len(train_raw)
    math500 = []
    for index, row in enumerate(load("math500")):
        math500.append({"problem_id": f"math500/{index}", "source_id": row.get("unique_id"),
                        "problem": row["problem"], "answer": row["answer"], "subject": row.get("subject"), "level": row.get("level")})
    gsm8k = []
    for index, row in enumerate(load("gsm8k", "main")):
        assert "####" in row["answer"]
        gsm8k.append({"problem_id": f"gsm8k/test/{index}", "problem": row["question"],
                      "answer": row["answer"].rsplit("####", 1)[1].strip().replace(",", "")})
    assert len(math500) == 500 and len(gsm8k) == 1319
    pb_path = ROOT / "data/raw/processbench_math.json"
    pb = json.loads(pb_path.read_text())
    test_keys = {question_key(row["problem"]) for row in math500 + gsm8k}
    pb_keys = {question_key(row["problem"]) for row in pb}
    exclusions = Counter()
    unique = {}
    duplicate_audit = []
    for row in train_raw:
        key = question_key(row["problem"])
        if key in test_keys or key in pb_keys:
            exclusions["evaluation_question_overlap"] += 1
        elif row["answer"] is None:
            exclusions["missing_gold_box"] += 1
        elif key in unique:
            prior = unique[key]
            equivalent = prior["answer"] == row["answer"] or outcome(
                "\\boxed{" + row["answer"] + "}", prior["answer"], True)["reward"] == 1
            assert equivalent, f"Conflicting duplicate training answers: {prior['problem_id']} and {row['problem_id']}"
            duplicate_audit.append({"kept": prior["problem_id"], "excluded": row["problem_id"],
                                    "kept_answer": prior["answer"], "excluded_answer": row["answer"],
                                    "mathematically_equivalent": True})
            exclusions["duplicate_question"] += 1
        else:
            unique[key] = row
    rows = [unique[key] for key in sorted(unique)]
    random.Random(config["split_seed"]).shuffle(rows)
    validation, train = rows[:config["validation_size"]], rows[config["validation_size"]:]
    for row in train + validation + math500 + gsm8k:
        row["question_sha256"] = question_key(row["problem"])
        row["prompt"] = prompt_for(row["problem"], config)
        row["prompt_tokens"] = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
    # Never silently truncate an input problem. Prompt length is decided before training.
    all_max = max(row["prompt_tokens"] for row in train + validation + math500 + gsm8k)
    if all_max > config["max_prompt_tokens"]:
        raise RuntimeError(f"Maximum prompt length {all_max} exceeds configured limit; revise uniformly before launch")
    sets = {"train": {r["question_sha256"] for r in train}, "validation": {r["question_sha256"] for r in validation},
            "math500": {r["question_sha256"] for r in math500}, "gsm8k": {r["question_sha256"] for r in gsm8k}}
    for other in ["validation", "math500", "gsm8k"]:
        assert not sets["train"] & sets[other]
    assert not sets["train"] & pb_keys
    exposure = set()
    for row in read_jsonl(assets["prm800k_audit"]["path"]):
        text = row.get("problem") or row.get("question")
        if isinstance(text, dict):
            text = text.get("problem")
        if not isinstance(text, str):
            raise RuntimeError(f"Unexpected PRM800K audit schema: {list(row)}")
        exposure.add(question_key(text))
    overlap = {key: len(values & exposure) for key, values in sets.items()}
    overlap["processbench_unique_questions"] = len(pb_keys & exposure)
    if overlap["math500"]:
        atomic_json(output / "known_overlap_block.json", {"overlap": overlap,
                    "reason": "Resolve primary evaluation exposure before launching the matched comparison"})
        raise RuntimeError("MATH500 overlaps the known PRM training inventory; audit the primary evaluation first")
    manifest = {"status": "prepared", "created_unix": time.time(), "split_seed": config["split_seed"],
                "original_train_count": len(train_raw), "excluded": dict(exclusions), "files": {},
                "duplicate_audit": duplicate_audit,
                "config_sha256": file_hash(ROOT / "training/baseline_config.json"), "assets_sha256": file_hash(ROOT / "receipts/assets.json"),
                "max_prompt_tokens_observed": all_max, "datasets": assets["datasets"],
                "prm800k_known_question_overlap": overlap, "prm800k_audit": assets["prm800k_audit"],
                "processbench_sha256": file_hash(pb_path),
                "limitation": "Exact normalized question audit only; unknown checkpoint pretraining exposure and paraphrases are not ruled out."}
    for name, values in {"train.jsonl": train, "validation.jsonl": validation, "math500.jsonl": math500, "gsm8k.jsonl": gsm8k}.items():
        write_jsonl(output / name, values)
        manifest["files"][name] = {"count": len(values), "sha256": file_hash(output / name)}
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == "__main__":
    main()
