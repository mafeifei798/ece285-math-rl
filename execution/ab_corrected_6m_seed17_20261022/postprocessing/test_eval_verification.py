"""Synthetic negative controls for the independent evaluation verifier (no models)."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("evaluation_auditor", Path(__file__).with_name("verify_completed_evaluations.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def stub_verifier(text, gold, terminated, **kwargs):
    if not terminated:
        return {"reward": 0.0, "status": "truncated", "answer": None}
    if not text.startswith("answer:"):
        return {"reward": 0.0, "status": "missing_or_invalid_box", "answer": None}
    answer = text.split(":", 1)[1]
    equivalent = answer == gold or (answer == "2/2" and gold == "1")
    return {"reward": float(equivalent), "status": "exact" if answer == gold else "symbolic", "answer": answer}


def prepared(n=3, name="toy"):
    result = []
    for i in range(n):
        problem = f"Question {name} {i}: compute one."
        result.append({"problem_id": f"{name}/{i}", "problem": problem, "answer": "1",
                       "question_sha256": audit.question_hash(problem)})
    return result


def predictions(rows, contents=None):
    contents = contents or ["answer:1"] * len(rows)
    result = []
    for item, text in zip(rows, contents):
        stop = text != "unfinished"
        result.append({"problem_id": item["problem_id"], "question_sha256": item["question_sha256"],
                       "gold": item["answer"], "completion": text, "generated_tokens": 12,
                       "finish_reason": "stop" if stop else "length",
                       **stub_verifier(text, item["answer"], stop)})
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value))


def write_safetensors(path, values=(1., 2.), metadata=None, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack("<" + "f" * len(values), *values)
    header = {"lora_weight": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(payload)]}}
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header, indent=indent).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class RowTests(unittest.TestCase):
    def setUp(self):
        self.gold = prepared()
        self.pred = predictions(self.gold, ["answer:2/2", "answer:0", "unfinished"])

    def check(self, pred=None, gold=None):
        return audit.verify_rows(gold or self.gold, pred or self.pred, 3, stub_verifier, {})

    def test_equivalent_alternative_is_not_string_compared_with_gold(self):
        metrics, ordered = self.check()
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(ordered[0]["answer"], "2/2")
        self.assertEqual(metrics["failure_types"], {"correct": 1, "complete_incorrect": 1, "unparsable": 0, "truncated": 1})

    def test_reordered_predictions_join_by_id(self):
        _, ordered = self.check(list(reversed(self.pred)))
        self.assertEqual([r["problem_id"] for r in ordered], [r["problem_id"] for r in self.gold])

    def test_missing_row_rejected(self):
        with self.assertRaisesRegex(audit.VerificationError, "Prediction count"):
            self.check(self.pred[:-1])

    def test_duplicate_id_rejected_even_at_expected_count(self):
        rows = copy.deepcopy(self.pred)
        rows[1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(audit.VerificationError, "Duplicate"):
            self.check(rows)

    def test_question_misalignment_rejected(self):
        rows = copy.deepcopy(self.pred)
        rows[1]["question_sha256"] = rows[0]["question_sha256"]
        with self.assertRaisesRegex(audit.VerificationError, "question mismatch"):
            self.check(rows)

    def test_gold_misalignment_rejected(self):
        rows = copy.deepcopy(self.pred)
        rows[1]["gold"] = "2"
        with self.assertRaisesRegex(audit.VerificationError, "Gold mismatch"):
            self.check(rows)

    def test_reward_corruption_rejected(self):
        rows = copy.deepcopy(self.pred)
        rows[1]["reward"] = 1.0
        with self.assertRaisesRegex(audit.VerificationError, "Recomputed verdict"):
            self.check(rows)

    def test_answer_corruption_rejected_even_if_reward_agrees(self):
        rows = copy.deepcopy(self.pred)
        rows[0]["answer"] = "1"
        with self.assertRaisesRegex(audit.VerificationError, "Recomputed verdict"):
            self.check(rows)

    def test_finish_reason_corruption_rejected(self):
        rows = copy.deepcopy(self.pred)
        rows[2]["finish_reason"] = "stop"
        with self.assertRaisesRegex(audit.VerificationError, "Termination/status"):
            self.check(rows)

    def test_verifier_exception_rejected(self):
        rows = copy.deepcopy(self.pred)
        rows[1]["status"] = "verifier_error:TimeoutError"
        with self.assertRaisesRegex(audit.VerificationError, "verifier_error"):
            self.check(rows)

    def test_nonfinite_or_oversized_tokens_rejected(self):
        for value in [float("nan"), 2049, -1, True]:
            rows = copy.deepcopy(self.pred)
            rows[1]["generated_tokens"] = value
            with self.subTest(value=value), self.assertRaisesRegex(audit.VerificationError, "token count"):
                self.check(rows)

    def test_question_bootstrap_is_paired_and_deterministic(self):
        a = predictions(self.gold, ["answer:1", "answer:0", "unfinished"])
        b = predictions(self.gold, ["answer:1", "answer:1", "answer:0"])
        one = audit.paired_comparison(a, list(reversed(b)))
        two = audit.paired_comparison(a, b)
        self.assertEqual(one, two)
        self.assertEqual(one["B_only_correct"], 1)
        self.assertEqual(one["A_only_correct"], 0)
        self.assertAlmostEqual(one["difference_pp"], 100 / 3)
        self.assertEqual(one["bootstrap_draws"], 10000)
        self.assertEqual(one["failure_type_transitions_A_to_B"]["truncated -> complete_incorrect"], 1)


class ProjectFixture:
    def __init__(self, root):
        self.root = root
        self.source = root / "experiments" / audit.EXPERIMENT
        self.counts = {"math500": 3, "gsm8k": 4}
        self.plan = {"seed": 17, "completion_token_budget_per_arm": 6000000,
                     "runs": {a: f"runs/{a}_corrected" for a in ["A", "B"]}, "controller": "runs/paired"}
        write_json(self.source / "plan.json", self.plan)
        write_json(root / "training/baseline_config.json", {"max_new_tokens": 2048})
        write_json(root / "receipts/assets.json", {"model_revision": "frozen-model"})
        write_json(root / "audits/reward_repair_v3_20261021/unit_metadata.json", {})
        for p in ["training/reward.py", "experiments/reward_repair_v3_20261021/repairs.py"]:
            write_json(root / p, {"frozen-test-stub": p})
        manifest = {"config_sha256": audit.sha(root / "training/baseline_config.json"),
                    "assets_sha256": audit.sha(root / "receipts/assets.json"), "files": {}}
        self.data = {}
        for name, count in self.counts.items():
            self.data[name] = prepared(count, name)
            p = root / "data/prepared" / (name + ".jsonl")
            write_rows(p, self.data[name])
            manifest["files"][p.name] = {"count": count, "sha256": audit.sha(p)}
        write_json(root / "data/prepared/manifest.json", manifest)
        release = {"version": audit.EXPERIMENT, "training_authorized": True,
                   "source_hashes": {str(self.source.relative_to(root) / "plan.json"): audit.sha(self.source / "plan.json")},
                   "runtime_inputs": {}, "unit_metadata_sha256": audit.sha(root / "audits/reward_repair_v3_20261021/unit_metadata.json")}
        write_json(self.source / "release.json", release)
        self.release_hash = audit.sha(self.source / "release.json")
        self.common = {"model_revision": "frozen-model", "config_sha256": manifest["config_sha256"],
                       "data_manifest_sha256": audit.sha(root / "data/prepared/manifest.json"),
                       "unit_metadata_sha256": release["unit_metadata_sha256"], "verifier_version": audit.VERSION}
        for arm in ["A", "B"]:
            run = root / self.plan["runs"][arm]
            checkpoint = run / "checkpoint-280"
            for name in ["adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "completion_accounting.json"]:
                write_json(checkpoint / name, {"fixture": name, "arm": arm})
            write_safetensors(checkpoint / "adapter_model.safetensors", metadata={"format": "pt"})
            write_json(checkpoint / "trainer_state.json", {"global_step": 280})
            write_safetensors(run / "final_adapter/adapter_model.safetensors", metadata={"format": "pt", "export": "same-tensors"}, indent=2)
            write_json(run / "final_adapter/adapter_config.json", {"fixture": "adapter_config.json", "arm": arm})
            status = {"status": "completed", "arm": arm, "pilot": False, "global_step": 280,
                      "budget": 6000000, "accounting": {"generated_completion_tokens": 6000123},
                      "checkpoint": "/remote/project/" + str(checkpoint.relative_to(root)),
                      "checkpoint_sha256": {p.name: audit.sha(p) for p in checkpoint.iterdir()}}
            write_json(run / "status.json", status)
            cfg = dict(self.common, arm=arm, seed=17, pilot=False, source_release_sha256=self.release_hash,
                       target_completion_tokens=6000000, packages={"math-verify": "fixture-only"},
                       shared_training_arguments={"loss_type": "grpo"}, actor_adapter_initial_sha256="same-init")
            write_json(run / "run_config.json", cfg)
            contract = dict(self.common, adapter_sha256=audit.sha(checkpoint / "adapter_model.safetensors"),
                            reward_code_sha256=audit.sha(root / "training/reward.py"),
                            repair_source_sha256=audit.sha(root / "experiments/reward_repair_v3_20261021/repairs.py"))
            write_json(run / "evaluation/evaluation_config.json", contract)
            summary = {**self.common, "greedy": True, "adapter": status["checkpoint"], "datasets": {},
                       "started_unix": 10, "finished_unix": 11}
            for name, count in self.counts.items():
                text = ["answer:1"] * count
                if arm == "B":
                    text[-1] = "unfinished"
                rows = predictions(self.data[name], text)
                path = run / "evaluation" / (name + ".jsonl")
                write_rows(path, rows)
                summary["datasets"][name] = {"count": count, "correct": count - int(arm == "B"),
                                             "accuracy": (count - int(arm == "B")) / count,
                                             "generated_tokens": 12 * count, "truncated": int(arm == "B"),
                                             "predictions_sha256": audit.sha(path)}
            write_json(run / "evaluation/evaluation_summary.json", summary)

    def verify(self):
        return audit.verify_project(self.root, verifier=stub_verifier, expected_counts=self.counts,
                                    release_sha256=self.release_hash)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.fixture = ProjectFixture(Path(self.directory.name))

    def tearDown(self):
        self.directory.cleanup()

    def test_end_to_end_contracts_counts_and_no_input_mutation(self):
        result = self.fixture.verify()
        self.assertTrue(result["passed"])
        self.assertTrue(result["inputs_unchanged_during_verification"])
        self.assertEqual(result["arms"]["A"]["datasets"]["math500"]["correct"], 3)
        self.assertEqual(result["paired_B_minus_A"]["gsm8k"]["difference_pp"], -25.)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm", sys.modules)

    def test_metadata_different_serialization_same_tensors_is_accepted(self):
        result = self.fixture.verify()
        arm = result["arms"]["A"]
        self.assertNotEqual(arm["adapter_sha256"], arm["exported_adapter_sha256"])
        self.assertTrue(arm["final_adapter_check"]["logical_tensors_match"])

    def test_changed_exported_tensor_payload_is_rejected(self):
        path = self.fixture.root / "runs/A_corrected/final_adapter/adapter_model.safetensors"
        write_safetensors(path, values=(1., 3.), metadata={"format": "pt"})
        with self.assertRaisesRegex(audit.VerificationError, "logical tensor content differs"):
            self.fixture.verify()

    def test_incomplete_arm_rejected(self):
        path = self.fixture.root / "runs/B_corrected/status.json"
        value = audit.read(path)
        value["status"] = "training"
        write_json(path, value)
        with self.assertRaisesRegex(audit.VerificationError, "formal training not complete"):
            self.fixture.verify()

    def test_prediction_file_corruption_detected_by_hash(self):
        path = self.fixture.root / "runs/B_corrected/evaluation/math500.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(audit.VerificationError, "Hash mismatch"):
            self.fixture.verify()

    def test_reward_corruption_detected_even_after_summary_hash_refreshed(self):
        root = self.fixture.root
        path = root / "runs/B_corrected/evaluation/math500.jsonl"
        rows = audit.read_rows(path)
        rows[0]["reward"] = 0.
        write_rows(path, rows)
        summary_path = root / "runs/B_corrected/evaluation/evaluation_summary.json"
        summary = audit.read(summary_path)
        summary["datasets"]["math500"]["predictions_sha256"] = audit.sha(path)
        write_json(summary_path, summary)
        with self.assertRaisesRegex(audit.VerificationError, "Recomputed verdict mismatch"):
            self.fixture.verify()

    def test_wrong_evaluated_adapter_rejected(self):
        path = self.fixture.root / "runs/A_corrected/evaluation/evaluation_config.json"
        value = audit.read(path)
        value["adapter_sha256"] = "wrong"
        write_json(path, value)
        with self.assertRaisesRegex(audit.VerificationError, "evaluation contract mismatch"):
            self.fixture.verify()

    def test_wrong_summary_count_rejected(self):
        path = self.fixture.root / "runs/A_corrected/evaluation/evaluation_summary.json"
        value = audit.read(path)
        value["datasets"]["math500"]["correct"] = 0
        write_json(path, value)
        with self.assertRaisesRegex(audit.VerificationError, "summary mismatch: correct"):
            self.fixture.verify()

    def test_wrong_release_rejected(self):
        with self.assertRaisesRegex(audit.VerificationError, "Hash mismatch"):
            audit.verify_project(self.fixture.root, verifier=stub_verifier, expected_counts=self.fixture.counts,
                                 release_sha256="untrusted")

    def controller_comparison(self):
        pair = {}
        for name in self.fixture.counts:
            a = audit.read_rows(self.fixture.root / "runs/A_corrected/evaluation" / (name + ".jsonl"))
            b = audit.read_rows(self.fixture.root / "runs/B_corrected/evaluation" / (name + ".jsonl"))
            pair[name] = audit.paired_comparison(a, b)
        return {"paired_B_minus_A": pair}

    def test_controller_bootstrap_is_independently_reproduced(self):
        write_json(self.fixture.root / "runs/paired/comparison.json", self.controller_comparison())
        result = self.fixture.verify()
        self.assertTrue(result["controller_comparison"]["paired_bootstrap_intervals_match"])

    def test_controller_bootstrap_corruption_rejected(self):
        value = self.controller_comparison()
        value["paired_B_minus_A"]["math500"]["paired_question_bootstrap_95ci_pp"][0] += 1
        write_json(self.fixture.root / "runs/paired/comparison.json", value)
        with self.assertRaisesRegex(audit.VerificationError, "paired bootstrap mismatch"):
            self.fixture.verify()


if __name__ == "__main__":
    unittest.main(verbosity=2)
