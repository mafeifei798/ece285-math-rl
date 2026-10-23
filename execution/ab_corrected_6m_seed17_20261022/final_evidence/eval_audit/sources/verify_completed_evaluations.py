"""Independent read-only evaluation verification; no inference, torch, or training.

Production CLI requires both fixed-budget arms and evaluations to be complete.
It uses the *frozen original repaired verifier*, not an alternative scorer. The
bookkeeping, joins, aggregations, and paired bootstrap are independently written.
Only --output is created, exclusively. All inputs are hashed before/after use.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata

EXPERIMENT = "ab_corrected_6m_seed17_20261022"
RELEASE_SHA256 = "9eac99ad38cf0ebb389f295655024eff1a09ff530ca05462bd2d9bcfcdbe6907"
VERSION = "reward-repair-v3-20261021"
COUNTS = {"math500": 500, "gsm8k": 1319}
PARSED = {"exact", "symbolic", "unit_exact", "unit_symbolic"}
KNOWN_STATUSES = PARSED | {"truncated", "missing_or_invalid_box", "unparseable"}


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def question_hash(text):
    # Independently implement the manifest's documented normalized question hash.
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return hashlib.sha256(normalized.encode()).hexdigest()


class InputInventory:
    def __init__(self, root):
        self.root = root.resolve()
        self.files = {}

    def observe(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), f"Input escapes project: {path}")
        require(path.is_file(), f"Missing required input: {path}")
        digest = sha(path)
        key = str(path.relative_to(self.root))
        require(key not in self.files or self.files[key] == digest, f"Input changed while reading: {key}")
        if expected is not None:
            require(digest == expected, f"Hash mismatch: {key}")
        self.files[key] = digest
        return digest

    def json(self, path, expected=None):
        self.observe(path, expected)
        return read(path)

    def jsonl(self, path, expected=None):
        self.observe(path, expected)
        return read_rows(path)

    def recheck(self):
        for name, digest in list(self.files.items()):
            require(sha(self.root / name) == digest, f"Input changed during verification: {name}")


def unique_index(rows, label):
    require(all(isinstance(r.get("problem_id"), str) and r["problem_id"] for r in rows),
            f"Invalid problem_id in {label}")
    result = {row["problem_id"]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate problem_id in {label}")
    return result


def category(row):
    if row["reward"] == 1:
        return "correct"
    if row["status"] == "truncated":
        return "truncated"
    if row["status"] in {"missing_or_invalid_box", "unparseable"}:
        return "unparsable"
    return "complete_incorrect"


def descriptive(values):
    import numpy as np
    v = np.asarray(values, dtype=float)
    if not len(v):
        return {"count": 0, "sum": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {"count": len(v), "sum": int(v.sum()), "mean": float(v.mean()),
            "median": float(np.median(v)), "p90": float(np.quantile(v, .9)), "max": int(v.max())}


def verify_rows(prepared, predictions, expected_count, verifier, units, max_tokens=2048):
    """Verify alignment and each persisted verdict; callable with a stub in tests.

    We do not compare extracted answer strings with gold strings. Equivalence is
    judged only by the frozen symbolic verifier, including accepted alternatives.
    Extracted 'answer' must exactly reproduce the deterministic saved extraction.
    """
    require(len(prepared) == expected_count, "Prepared count mismatch")
    require(len(predictions) == expected_count, "Prediction count mismatch")
    target = unique_index(prepared, "prepared dataset")
    pred = unique_index(predictions, "predictions")
    require(target.keys() == pred.keys(), "Prediction IDs do not match prepared dataset")
    mismatches = []
    for problem_id, row in pred.items():
        gold = target[problem_id]
        require(gold["question_sha256"] == question_hash(gold["problem"]),
                f"Prepared question hash mismatch: {problem_id}")
        require(row.get("question_sha256") == gold["question_sha256"], f"Prediction question mismatch: {problem_id}")
        require(row.get("gold") == gold["answer"], f"Gold mismatch: {problem_id}")
        require(isinstance(row.get("completion"), str), f"Invalid completion: {problem_id}")
        require(row.get("finish_reason") in {"stop", "length"}, f"Unknown finish_reason: {problem_id}")
        require(type(row.get("generated_tokens")) is int and 0 <= row["generated_tokens"] <= max_tokens,
                f"Invalid token count: {problem_id}")
        require(isinstance(row.get("reward"), (int, float)) and not isinstance(row["reward"], bool)
                and math.isfinite(row["reward"]) and row["reward"] in (0, 1), f"Non-binary reward: {problem_id}")
        require(row.get("status") in KNOWN_STATUSES, f"Unknown/verifier_error stored status: {problem_id}")
        require((row["finish_reason"] == "length") == (row["status"] == "truncated"),
                f"Termination/status mismatch: {problem_id}")
        require(row["reward"] == 0 or row["status"] in PARSED, f"Unparsed correct verdict: {problem_id}")
        got = verifier(row["completion"], gold["answer"], row["finish_reason"] == "stop",
                       question=gold["problem"], problem_id=problem_id, unit_metadata=units)
        require(got.get("status") in KNOWN_STATUSES, f"Unknown/verifier_error replay status: {problem_id}")
        # Includes optional requested-unit fields when the frozen scorer emits them.
        for key, expected in got.items():
            if key not in row or row[key] != expected:
                mismatches.append({"problem_id": problem_id, "field": key,
                                   "saved": row.get(key), "recomputed": expected})
    require(not mismatches, "Recomputed verdict mismatch: " + json.dumps(mismatches[:10], ensure_ascii=False))
    ordered = [pred[row["problem_id"]] for row in prepared]
    correct = sum(int(row["reward"]) for row in ordered)
    lengths = [row["generated_tokens"] for row in ordered]
    cats = Counter(category(row) for row in ordered)
    return {
        "count": len(ordered), "correct": correct, "accuracy": correct / len(ordered),
        "generated_tokens": sum(lengths), "truncated": cats["truncated"],
        "status_counts": dict(sorted(Counter(r["status"] for r in ordered).items())),
        "failure_types": {key: cats[key] for key in ["correct", "complete_incorrect", "unparsable", "truncated"]},
        "length": descriptive(lengths),
        "length_by_failure_type": {key: descriptive([r["generated_tokens"] for r in ordered if category(r) == key])
                                   for key in ["correct", "complete_incorrect", "unparsable", "truncated"]},
        "all_recomputed_verdicts_match": True,
    }, ordered


def paired_comparison(a, b, draws=10000, seed=285):
    import numpy as np
    aa, bb = unique_index(a, "A predictions"), unique_index(b, "B predictions")
    require(aa.keys() == bb.keys() and len(aa) > 0, "Unpaired prediction IDs")
    ids = list(aa)
    diff = np.asarray([bb[i]["reward"] - aa[i]["reward"] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, draws, 500):
        indices = rng.integers(0, len(ids), size=(min(500, draws - start), len(ids)))
        means.extend(diff[indices].mean(axis=1).tolist())
    changed = [i for i, delta in zip(ids, diff) if delta != 0]
    transitions = Counter(category(aa[i]) + " -> " + category(bb[i]) for i in ids)
    return {
        "n": len(ids), "difference_pp": float(diff.mean() * 100),
        "paired_question_bootstrap_95ci_pp": [float(x * 100) for x in np.quantile(means, [.025, .975])],
        "bootstrap_draws": draws, "bootstrap_seed": seed, "bootstrap_unit": "same question, both arms",
        "bootstrap_order": "A prediction-file order; B joined by problem_id",
        "B_only_correct": int((diff > 0).sum()), "A_only_correct": int((diff < 0).sum()),
        "both_correct": sum(aa[i]["reward"] == bb[i]["reward"] == 1 for i in ids),
        "both_wrong": sum(aa[i]["reward"] == bb[i]["reward"] == 0 for i in ids),
        "failure_type_transitions_A_to_B": dict(sorted(transitions.items())),
        "changed_questions": [{"problem_id": i, "A_reward": aa[i]["reward"], "B_reward": bb[i]["reward"],
                               "A_status": aa[i]["status"], "B_status": bb[i]["status"],
                               "A_category": category(aa[i]), "B_category": category(bb[i]),
                               "A_tokens": aa[i]["generated_tokens"], "B_tokens": bb[i]["generated_tokens"]}
                              for i in changed],
        "limitation": "One training seed. The interval covers sampled questions, not training-seed variability; descriptive failure categories do not establish mathematical causality.",
    }


def frozen_verifier(root, packages):
    """Deferred import of math-verify through source-pinned repairs; never a model."""
    actual = importlib.metadata.version("math-verify")
    require(actual == packages["math-verify"], "math-verify runtime differs from training")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    path = root / "experiments/reward_repair_v3_20261021/repairs.py"
    spec = importlib.util.spec_from_file_location("_independent_eval_frozen_repairs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(sys.modules["v2core"].__file__).resolve() ==
            (root / "experiments/reward_repair_v2_20261021/v2core.py").resolve(), "Wrong v2core module imported")
    return module.repaired_outcome


def stored_project_path(root, value):
    """Resolve remote absolute receipts inside a local project mirror by runs/ suffix."""
    path = Path(value)
    if not path.is_absolute():
        return root / path
    if path.is_relative_to(root):
        return path
    parts = path.parts
    require(parts.count("runs") == 1, f"Cannot safely remap recorded path: {value}")
    return root.joinpath(*parts[parts.index("runs"):])


def verify_adapter_tensors(checkpoint_path, exported_path):
    """Use the sibling stdlib parser; metadata serialization is not model state.

    The actual evaluated checkpoint retains its strict release/status hash checks.
    This extra check only establishes that final_adapter exports the same tensors.
    """
    helper = Path(__file__).with_name("audit_completed_training.py")
    helper_sha256 = sha(helper)
    spec = importlib.util.spec_from_file_location("_eval_adapter_inventory", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = module.safetensors_inventory(checkpoint_path)
    b = module.safetensors_inventory(exported_path)
    require(a == b, "Checkpoint and final_adapter logical tensor content differs")
    require(sha(helper) == helper_sha256, "Adapter audit helper changed during import")
    return {"logical_tensors_match": True, "tensor_count": len(a),
            "stdlib_helper_sha256": helper_sha256,
            "comparison": "tensor name, dtype, shape, byte length, and payload SHA256; ignores header metadata serialization"}


def verify_project(root, *, verifier=None, expected_counts=None, release_sha256=RELEASE_SHA256):
    """Full immutable-input audit. Optional test dependencies are not exposed by CLI."""
    root = Path(root).resolve()
    inv = InputInventory(root)
    counts = COUNTS if expected_counts is None else expected_counts
    source = root / "experiments" / EXPERIMENT
    release = inv.json(source / "release.json", release_sha256)
    require(release["version"] == EXPERIMENT and release["training_authorized"] is True, "Wrong release")
    for name, digest in release["source_hashes"].items():
        inv.observe(root / name, digest)
    for name, digest in release["runtime_inputs"].items():
        inv.observe(root / name, digest)
    plan = inv.json(source / "plan.json")
    require(plan["seed"] == 17 and plan["completion_token_budget_per_arm"] == 6000000, "Wrong paired plan")
    cfg = inv.json(root / "training/baseline_config.json")
    manifest = inv.json(root / "data/prepared/manifest.json")
    assets = inv.json(root / "receipts/assets.json")
    metadata_path = root / "audits/reward_repair_v3_20261021/unit_metadata.json"
    units = inv.json(metadata_path, release["unit_metadata_sha256"])
    common = {
        "model_revision": assets["model_revision"],
        "config_sha256": inv.observe(root / "training/baseline_config.json"),
        "data_manifest_sha256": inv.observe(root / "data/prepared/manifest.json"),
        "unit_metadata_sha256": inv.observe(metadata_path), "verifier_version": VERSION,
    }
    require(manifest["config_sha256"] == common["config_sha256"], "Manifest/config mismatch")
    require(manifest["assets_sha256"] == inv.observe(root / "receipts/assets.json"), "Manifest/assets mismatch")
    prepared = {}
    for name, count in counts.items():
        data_path = root / "data/prepared" / (name + ".jsonl")
        require(manifest["files"][data_path.name]["count"] == count, f"Manifest count mismatch: {name}")
        prepared[name] = inv.jsonl(data_path, manifest["files"][data_path.name]["sha256"])
    results, predictions, specs = {}, {}, {}
    for arm in ["A", "B"]:
        run = root / plan["runs"][arm]
        status = inv.json(run / "status.json")
        spec = inv.json(run / "run_config.json")
        require(status["status"] == "completed" and status["arm"] == arm and status["pilot"] is False,
                f"{arm} formal training not complete")
        require(spec["arm"] == arm and spec["seed"] == 17 and spec["pilot"] is False, f"Wrong {arm} run")
        require(spec["source_release_sha256"] == release_sha256, f"{arm} source release mismatch")
        require(spec["target_completion_tokens"] == status["budget"] == 6000000, f"{arm} budget mismatch")
        tokens = status["accounting"]["generated_completion_tokens"]
        require(6000000 <= tokens < 6000000 + 32 * 2048, f"{arm} final token budget not met")
        for key, expected in common.items():
            require(spec[key] == expected, f"{arm} run contract mismatch: {key}")
        checkpoint = stored_project_path(root, status["checkpoint"])
        require(checkpoint == run / ("checkpoint-" + str(status["global_step"])), f"{arm} nonfinal checkpoint")
        for name, digest in status["checkpoint_sha256"].items():
            inv.observe(checkpoint / name, digest)
        required = {"adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
                    "optimizer.pt", "scheduler.pt", "rng_state.pth", "completion_accounting.json"}
        require(required <= status["checkpoint_sha256"].keys(), f"{arm} incomplete final checkpoint")
        state = inv.json(checkpoint / "trainer_state.json")
        require(state["global_step"] == status["global_step"], f"{arm} checkpoint step mismatch")
        adapter_sha = inv.observe(checkpoint / "adapter_model.safetensors")
        exported_sha = inv.observe(run / "final_adapter/adapter_model.safetensors")
        logical_adapter = verify_adapter_tensors(checkpoint / "adapter_model.safetensors",
                                                run / "final_adapter/adapter_model.safetensors")
        require(inv.json(checkpoint / "adapter_config.json") == inv.json(run / "final_adapter/adapter_config.json"),
                f"{arm} final_adapter config differs from checkpoint")
        contract = inv.json(run / "evaluation/evaluation_config.json")
        expected_contract = dict(common, adapter_sha256=adapter_sha,
                                 reward_code_sha256=inv.observe(root / "training/reward.py"),
                                 repair_source_sha256=inv.observe(root / "experiments/reward_repair_v3_20261021/repairs.py"))
        require(contract == expected_contract, f"{arm} evaluation contract mismatch")
        summary = inv.json(run / "evaluation/evaluation_summary.json")
        require(summary["greedy"] is True and set(summary["datasets"]) == set(counts), f"{arm} evaluation incomplete")
        require(stored_project_path(root, summary["adapter"]) == checkpoint, f"{arm} wrong evaluated adapter")
        for key in ["model_revision", "config_sha256", "verifier_version", "unit_metadata_sha256"]:
            require(summary[key] == common[key], f"{arm} summary contract mismatch: {key}")
        require(summary["finished_unix"] >= summary["started_unix"], f"{arm} incomplete evaluation timestamps")
        if verifier is None:
            verifier = frozen_verifier(root, spec["packages"])
        results[arm] = {"checkpoint": str(checkpoint.relative_to(root)), "adapter_sha256": adapter_sha,
                        "exported_adapter_sha256": exported_sha, "final_adapter_check": logical_adapter,
                        "actor_completion_tokens": tokens, "optimizer_steps": status["global_step"], "datasets": {}}
        predictions[arm] = {}
        for name, count in counts.items():
            path = run / "evaluation" / (name + ".jsonl")
            saved = summary["datasets"][name]
            rows = inv.jsonl(path, saved["predictions_sha256"])
            metrics, ordered = verify_rows(prepared[name], rows, count, verifier, units, cfg["max_new_tokens"])
            for key in ["count", "correct", "accuracy", "generated_tokens", "truncated"]:
                require(metrics[key] == saved[key], f"{arm}/{name} summary mismatch: {key}")
            metrics["predictions_sha256"] = inv.observe(path)
            results[arm]["datasets"][name] = metrics
            # Preserve A file order only for deterministic RNG replay of the
            # controller's bootstrap. Pairing never depends on B row order.
            predictions[arm][name] = rows
        specs[arm] = spec
    for key in ["packages", "shared_training_arguments", "actor_adapter_initial_sha256", *common]:
        require(specs["A"][key] == specs["B"][key], f"A/B common setup mismatch: {key}")
    comparisons = {name: paired_comparison(predictions["A"][name], predictions["B"][name]) for name in counts}
    comparison_path = root / plan["controller"] / "comparison.json"
    controller_comparison = {"available": comparison_path.is_file()}
    if comparison_path.is_file():
        previous = inv.json(comparison_path)
        for name, paired in comparisons.items():
            for key in ["n", "B_only_correct", "A_only_correct"]:
                require(previous["paired_B_minus_A"][name][key] == paired[key], f"Controller comparison mismatch: {name}/{key}")
            require(abs(previous["paired_B_minus_A"][name]["difference_pp"] - paired["difference_pp"]) < 1e-10,
                    f"Controller comparison difference mismatch: {name}")
            saved_ci = previous["paired_B_minus_A"][name]["paired_question_bootstrap_95ci_pp"]
            fresh_ci = paired["paired_question_bootstrap_95ci_pp"]
            require(len(saved_ci) == 2 and all(abs(x-y) < 1e-10 for x, y in zip(saved_ci, fresh_ci)),
                    f"Controller paired bootstrap mismatch: {name}")
        controller_comparison["paired_counts_and_point_estimates_match"] = True
        controller_comparison["paired_bootstrap_intervals_match"] = True
    inv.recheck()
    return {"passed": True, "kind": "independent_completed_evaluation_verification", "experiment": EXPERIMENT,
            "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 17, "arms": results,
            "paired_B_minus_A": comparisons, "controller_comparison": controller_comparison,
            "inputs_sha256": inv.files, "inputs_unchanged_during_verification": True,
            "verifier": "Frozen repaired_outcome, same math-verify version; no replacement scoring rule.",
            "scope": "CPU only, saved outputs only, no inference/training/PRM reranking/checkpoint selection.",
            "limitations": ["Single training seed; paired intervals cover questions only.",
                            "Replay tests implementation consistency, not semantic completeness of the verifier.",
                            "Saved token counts are range/sum checked; tokenizer IDs were not persisted for independent retokenization.",
                            "Final-answer correctness does not certify the reasoning steps."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists; use a new audit receipt path")
    result = verify_project(args.root)
    result["auditor_sha256"] = sha(Path(__file__).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "output": str(args.output), "arms": result["arms"],
                      "paired_B_minus_A": {k: {key: value for key, value in v.items() if key != "changed_questions"}
                                           for k, v in result["paired_B_minus_A"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
