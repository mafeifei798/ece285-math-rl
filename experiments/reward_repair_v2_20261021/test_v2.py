"""Behavioral regressions and independent installed-TRL tensor-kernel checks."""
import json
import math
import os
from pathlib import Path
import sys
import unittest

from trl_kernel import actual_trl_advantages, save_provenance
from v2core import (atomic_json, audit_advantages, canonical_view, encode_native,
                    legacy_module, make_metadata, mixed_reward, outcome_v2, valid_for_prm)

ROOT, OUT, TRL_SOURCE = map(Path, sys.argv[1:4])
sys.argv = sys.argv[:1]


class RewardRepairTests(unittest.TestCase):
    def judge(self, prediction, gold="20", kind="percent_value", question="What percent of 100 is 20?", terminated=True):
        return outcome_v2("\\boxed{" + prediction + "}", gold, terminated, question=question,
                          metadata=make_metadata(question, gold, kind))

    def test_real_percent_false_negative(self):
        self.assertEqual(legacy_module("reward").outcome("\\boxed{20\\%}", "20", True)["reward"], 0)
        result = self.judge("20\\%")
        self.assertEqual(result["reward"], 1)
        self.assertTrue(valid_for_prm(result))

    def test_percent_value_and_negative_cases(self):
        for s in ("20", "20%", "20\\%", "(2*10)\\%"):
            self.assertEqual(self.judge(s)["reward"], 1, s)
        for s in ("0.2", "0.2\\%", "2000\\%", "-20\\%", "21\\%"):
            self.assertEqual(self.judge(s)["reward"], 0, s)

    def test_percent_gold_marker_consistency(self):
        for gold in ("20", "20%", "20 \\%"):
            for s in ("20", "20\\%"):
                self.assertEqual(self.judge(s, gold)["reward"], 1)
            self.assertEqual(self.judge("0.2", gold)["reward"], 0)

    def test_probability_is_not_percentage_points(self):
        for s in ("1/5", "0.2", "20\\%"):
            self.assertEqual(self.judge(s, "\\frac{1}{5}", "dimensionless", "What is the probability?")["reward"], 1)
        self.assertEqual(self.judge("20", "\\frac{1}{5}", "dimensionless", "What is the probability?")["reward"], 0)

    def test_mentions_percent_but_asks_count(self):
        q = "Eleven percent of what number is seventy-seven?"
        self.assertEqual(self.judge("700", "700", "legacy", q)["reward"], 1)
        self.assertEqual(self.judge("700\\%", "700", "legacy", q)["reward"], 0)

    def test_legacy_formats_unchanged(self):
        legacy = legacy_module("reward")
        for pred, gold in [("\\frac{2}{4}", "\\frac{1}{2}"), ("\\sqrt{4}", "2"),
                           ("x+1", "1+x"), ("\\{1,2\\}", "\\{1,2\\}"), ("nonsense", "2")]:
            text = "\\boxed{" + pred + "}"
            self.assertEqual(outcome_v2(text, gold, True, question="Q", metadata=make_metadata("Q", gold)),
                             legacy.outcome(text, gold, True))

    def test_metadata_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            outcome_v2("\\boxed{20%}", "20", True, question="different question", metadata=make_metadata("Q", "20", "percent_value"))
        with self.assertRaises(ValueError):
            outcome_v2("\\boxed{20%}", "21", True, question="Q", metadata=make_metadata("Q", "20", "percent_value"))

    def test_invalid_and_truncated_stay_unscored(self):
        meta = make_metadata("Q", "20", "percent_value")
        for text, terminated in [("\\boxed{20%}", False), ("20%", True), ("\\boxed{20", True)]:
            result = outcome_v2(text, "20", terminated, question="Q", metadata=meta)
            self.assertFalse(valid_for_prm(result))
            self.assertEqual(mixed_reward(result, []), 0)
            with self.assertRaises(ValueError):
                mixed_reward(result, [.9])

    def test_no_score_scale_or_aggregation_change(self):
        self.assertAlmostEqual(mixed_reward({"status": "unit_exact", "reward": 1}, [.9, .2, .7]), 1.04)
        for scores in ([], [float("nan")], [-.01], [1.01]):
            with self.assertRaises(ValueError):
                mixed_reward({"status": "symbolic", "reward": 0}, scores)

    def test_crlf_and_raw_span_mapping(self):
        raw = "  A\r\n\r\nB\r\nC \n\n D  "
        old = canonical_view(raw, False)
        new = canonical_view(raw, True)
        self.assertEqual(old["steps"], ["A\r\n\r\nB\r\nC", "D"])
        self.assertEqual(new["steps"], ["A", "B\nC", "D"])
        for step, span in zip(new["steps"], new["spans"]):
            self.assertEqual(raw[span["raw_start"]:span["raw_end"]].replace("\r\n", "\n"), step)
        self.assertEqual(raw, "  A\r\n\r\nB\r\nC \n\n D  ")

    def test_regular_newlines_unchanged(self):
        for raw in ("a\n\nb\n", " a\n\n\n\nb ", "a\nb", "```python\nimport x\n\nx=2\n```", "one block", " "):
            self.assertEqual(canonical_view(raw, True)["steps"], canonical_view(raw, False)["steps"])

    def test_native_prefix_position_compatibility(self):
        from transformers import AutoTokenizer
        assets = json.loads((ROOT / "receipts/B_prm_assets.json").read_text())
        tok = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True)
        legacy = legacy_module("prm_reward")
        texts = ["First $x=1$.\n\nThus $\\boxed{1}$.", "A\r\n\r\nB", "a\n\n\n\nb", "```python\nimport x\n\nx=2\n```", "a single answer"]
        for raw in texts:
            a = encode_native(tok, "Q", raw, 4096, normalize_crlf=False)
            b = legacy.encode_native(tok, "Q", raw, 4096)
            self.assertEqual(a["input_ids"], b["input_ids"])
            self.assertEqual(a["positions"], b["positions"])
            c = encode_native(tok, "Q", raw, 4096, normalize_crlf=True)
            expected = legacy.encode_native(tok, "Q", raw.replace("\r\n", "\n"), 4096)
            self.assertEqual(c["input_ids"], expected["input_ids"])
            self.assertEqual(c["positions"], expected["positions"])
        with self.assertRaises(ValueError):
            encode_native(tok, "Q", "lengthy", 1, normalize_crlf=True)

    def guard_case(self, values, name, device="cpu"):
        import torch
        rewards, actual = actual_trl_advantages(values, TRL_SOURCE, device)
        before = actual.clone()
        pointer = actual.data_ptr()
        returned = audit_advantages(rewards, actual, receipt_path=OUT / "guard_tests" / (name + ".json"),
                                    problem_ids=[f"q{i // 8}" for i in range(len(values))])
        self.assertIs(returned, actual)
        self.assertEqual(pointer, actual.data_ptr())
        self.assertTrue(torch.equal(actual, before))

    def test_actual_installed_trl_kernel_cpu(self):
        cases = [[0.] * 8, [1.] * 8, [1.2] * 8, [.1, .2, .2, .7, 1., 1.1, 1.2, 0.],
                 [1.2] * 7 + [1.1999999], [.2] * 7 + [.20000002]]
        for i, values in enumerate(cases):
            self.guard_case(values, "cpu_" + str(i))

    def test_actual_installed_trl_kernel_gpu(self):
        import torch
        self.assertTrue(torch.cuda.is_available())
        self.guard_case([1.2] * 7 + [1.1999999] + [.2] * 7 + [.20000002], "gpu_near_zero_variance", "cuda:0")

    def test_wrong_formula_is_not_tolerated(self):
        import torch
        values = [0., .1, .3, .4, .5, .6, .8, 1.]
        r, a = actual_trl_advantages(values, TRL_SOURCE)
        wrong = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
        path = OUT / "guard_tests" / "wrong_population_std.json"
        with self.assertRaises(ValueError):
            audit_advantages(r, wrong, receipt_path=path)
        self.assertFalse(json.loads(path.read_text())["passed"])

    def test_nonfinite_shape_and_alignment_evidence_saved(self):
        import torch
        r, a = actual_trl_advantages([0., .1, .3, .4, .5, .6, .8, 1.], TRL_SOURCE)
        bad = a.clone()
        bad[0] = float("nan")
        for name, reward, actual, ids in [("nan", r, bad, None), ("shape", r[:-1], a[:-1], None),
                                          ("alignment", r, a, ["a"] * 7 + ["b"])]:
            path = OUT / "guard_tests" / (name + ".json")
            with self.assertRaises(ValueError):
                audit_advantages(reward, actual, receipt_path=path, problem_ids=ids)
            self.assertTrue(path.exists())
            self.assertFalse(json.loads(path.read_text())["passed"])

    def test_B_patch_preserves_output_and_accounting_without_model(self):
        import ast
        import collections
        import torch
        from types import SimpleNamespace
        from b_audit_patch import patched_b_trainer_class
        from v2core import HERE
        path = HERE / "legacy_snapshot/training_b/train_b.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "BTrainer")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions")
        class FakeGRPO:
            @staticmethod
            def _generate_and_score_completions(self, inputs):
                return self.fake_output
        environment = {"torch": torch, "json": json, "Counter": collections.Counter, "GRPOTrainer": FakeGRPO,
                       "group_diagnostics": legacy_module("prm_reward").group_diagnostics}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), environment)
        original_class = type("StubOriginalB", (), {method.name: environment[method.name]})
        repaired_class = patched_b_trainer_class(original_class, audit_directory=OUT / "stub_B_audit")
        values = [0., .04, .1, .14, 1., 1.04, 1.1, 1.2] * 4
        rewards, advantages = actual_trl_advantages(values, TRL_SOURCE)
        def stub(typ, folder):
            obj = typ()
            obj.run_dir = OUT / folder
            obj.run_dir.mkdir(parents=True, exist_ok=True)
            obj.accelerator = SimpleNamespace(num_processes=1)
            obj.model = SimpleNamespace(training=True)  # Only a flag; no model exists.
            obj.state = SimpleNamespace(global_step=0)
            obj.num_generations = 8
            obj.last_reward_values = values
            obj.last_records = [{"outcome_reward": int(v >= 1), "valid_for_process_reward": True,
                "min_score": (v - int(v >= 1)) / .2, "completion_tokens": 1, "terminated": True,
                "step_count": 1, "raw_no_double_newline": False, "status": "symbolic", "problem_id": "q" + str(i // 8)}
                for i, v in enumerate(values)]
            obj.last_prm_cost = {"prm_forward_seconds": 0., "prm_input_tokens": 32, "prm_padded_tokens": 32, "prm_scored_completions": 32}
            obj.accounting = collections.defaultdict(int)
            obj.accounting.update(reward_status_counts={}, outcome_group_counts={})
            obj._metrics = {"train": {}}
            obj.fake_output = {"advantages": advantages.clone(), "completion_mask": torch.ones((32, 1), dtype=torch.int64)}
            return obj
        a, b = stub(original_class, "stub_original_method"), stub(repaired_class, "stub_repaired_method")
        old_result = a._generate_and_score_completions([])
        new_result = b._generate_and_score_completions([])
        self.assertTrue(torch.equal(old_result["advantages"], new_result["advantages"]))
        self.assertEqual(dict(a.accounting), dict(b.accounting))
        self.assertEqual(a._metrics, b._metrics)
        self.assertTrue(repaired_class.numeric_patch_receipt["remaining_method_statements_identical"])
        self.assertEqual((a.run_dir / "group_diagnostics.jsonl").read_text(), (b.run_dir / "group_diagnostics.jsonl").read_text())
        atomic_json(OUT / "B_method_patch_test.json", repaired_class.numeric_patch_receipt)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    save_provenance(TRL_SOURCE, OUT / "installed_trl_tensor_kernel.json")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RewardRepairTests))
    atomic_json(OUT / "test_results.json", {"tests": result.testsRun, "failures": len(result.failures),
                "errors": len(result.errors), "passed": result.wasSuccessful(), "new_training": False})
    sys.exit(0 if result.wasSuccessful() else 1)
