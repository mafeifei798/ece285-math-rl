"""Independent small fixtures for saved-signal diagnostics; no project imports."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pair_signals import (analyze_rows, canonical, group_decomposition, load_arm,
                                  prefix_comparison, SignalError)


def fixture(y, p, i, arm="B", invalid=False):
    raw = "First step.\r\n\r\nSecond step."
    view = canonical(raw)
    scored = arm == "B" and not invalid
    row = dict(problem_id="q", optimizer_step_before_update=i//32,
               completion=raw, completion_tokens=10, terminated=not invalid,
               status="truncated" if invalid else "exact", outcome_reward=y,
               reward=(y + .2*p) if scored else y if arm == "A" else 0.,
               advantage=0., advantage_rule="mixed_reward_original_group_std",
               valid_for_process_reward=scored, raw_no_double_newline=True,
               step_scores=[p, p] if scored else [], step_count=2 if scored else 0,
               min_score=p if scored else 0., prm_input_tokens=20 if scored else 0)
    if scored:
        row.update(step_raw_spans=view["spans"], raw_sha256=view["raw_sha256"],
                   crlf_replacements=2, canonical_no_double_newline=False,
                   no_internal_separator=False, repair_version="reward-repair-v3-20261021")
    return row


def fill(rows):
    for start in range(0, len(rows), 8):
        expected = group_decomposition(rows[start:start+8])["expected"]
        for row, advantage in zip(rows[start:start+8], expected):
            row["advantage"] = advantage
    return rows


class Tests(unittest.TestCase):
    def test_sample_std_not_population(self):
        rows = [fixture(int(i < 4), 0, i, "A") for i in range(8)]
        d = group_decomposition(rows)
        expected_std = math.sqrt(2/7)
        self.assertAlmostEqual(d["reward_sample_std"], expected_std)
        self.assertAlmostEqual(d["expected"][0], .5/(expected_std+1e-4))

    def test_decomposition_and_residual_are_distinct(self):
        rows = fill([fixture(i % 2, .1*i, i) for i in range(8)])
        d = group_decomposition(rows)
        for actual, y, p in zip(d["actual"], d["outcome_term"], d["process_term"]):
            self.assertAlmostEqual(actual, y+p)
        self.assertTrue(any(abs(a-b)>1e-4 for a,b in zip(d["process_term"], d["residual_vs_outcome_only"])))

    def test_all_wrong_signal_is_entirely_process(self):
        rows = fill([fixture(0, .800 + .001*i, i) for i in range(8)])
        d = group_decomposition(rows)
        self.assertEqual(d["kind"], "all_wrong")
        self.assertTrue(all(v == 0 for v in d["outcome_term"]))
        self.assertGreater(d["effective_process_coefficient"], 100.)
        self.assertTrue(any(a>0 for a in d["actual"]))

    def test_invalid_placeholder_excluded_from_prm_distributions(self):
        rows = fill([fixture(0, .8, i, invalid=(i%8 == 0)) for i in range(32)])
        result, _ = analyze_rows(rows, "B")
        score = result["metrics"]["PRM_min_score_scored_only"]
        self.assertEqual(score["n"], 28)
        self.assertAlmostEqual(score["mean"], .8)
        self.assertEqual(result["counts"]["truncated"], 4)
        self.assertEqual(result["groups"]["all_wrong"]["counts"]["groups"], 4)

    def test_shared_prefix_unequal_length_is_expected(self):
        result = prefix_comparison(["a","b"], ["a","b","c"])
        self.assertTrue(result["overlap_prompt_order_equal"])
        self.assertEqual(result["unpaired_b_tail_groups"], 1)
        self.assertEqual(result["identical_initial_prefix_groups"], 2)
        changed = prefix_comparison(["a","x"], ["a","b","c"])
        self.assertFalse(changed["overlap_prompt_order_equal"])
        self.assertEqual(changed["first_mismatch_group"], 1)

    def test_crlf_spans_preserve_original_text(self):
        raw = "  a\r\n\r\n b\r\nc  "
        view = canonical(raw)
        self.assertEqual(view["steps"], ["a", "b\nc"])
        for text, span in zip(view["steps"], view["spans"]):
            self.assertEqual(raw[span["raw_start"]:span["raw_end"]].replace("\r\n", "\n"), text)

    def test_bad_alignment_is_rejected(self):
        rows = fill([fixture(0, .1 + .01*i, i) for i in range(32)])
        rows[0]["step_raw_spans"][0]["raw_end"] += 1
        with self.assertRaisesRegex(SignalError, "raw/canonical spans"):
            analyze_rows(rows, "B")

    def test_zero_variance_advantages(self):
        rows = fill([fixture(1, 0, i, "A") for i in range(32)])
        result, _ = analyze_rows(rows, "A")
        self.assertEqual(result["metrics"]["absolute_actual_advantage"]["mean"], 0)
        self.assertEqual(result["groups"]["all_correct"]["process_share_of_sum_absolute_components"], None)

    def test_formula_change_is_rejected(self):
        rows = fill([fixture(i%2, .2, i) for i in range(32)])
        rows[0]["reward"] += .01
        with self.assertRaisesRegex(SignalError, "original formula"):
            analyze_rows(rows, "B")

    def test_pending_input_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            run = root / "pending"
            run.mkdir()
            for name, value in [("status.json", {"status":"training"}),
                                ("run_config.json", {}), ("live_audits.json", {}),
                                ("initial_sampling.json", [])]:
                (run/name).write_text(json.dumps(value))
            (run/"rollouts.jsonl").write_text("")
            with self.assertRaisesRegex(SignalError, "still pending"):
                load_arm(root, "pending", "A")


if __name__ == "__main__":
    unittest.main()
