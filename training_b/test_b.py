"""CPU checks for reward semantics, group normalization and native boundaries."""
import math
import os
import pytest
import torch
from prm_reward import encode_native, group_diagnostics, mixed_reward, split_steps


def test_correct_wrong_and_invalid_rewards():
    assert mixed_reward({"status": "exact", "reward": 1}, [0.8, 0.4]) == pytest.approx(1.08)
    assert mixed_reward({"status": "symbolic", "reward": 0}, [0.9, 0.3]) == pytest.approx(0.06)
    for status in ["truncated", "unparseable", "missing_or_invalid_box", "verifier_error:Timeout"]:
        assert mixed_reward({"status": status, "reward": 0}, []) == 0
    with pytest.raises(AssertionError):
        mixed_reward({"status": "truncated", "reward": 0}, [1])


@pytest.mark.parametrize("scores", [[], [math.nan], [math.inf], [-0.01], [1.01]])
def test_bad_prm_scores_fail_closed(scores):
    with pytest.raises(ValueError):
        mixed_reward({"status": "exact", "reward": 1}, scores)


def test_group_types_use_outcomes_not_combined_reward():
    y = [1] * 8 + [0] * 8 + [1, 0] * 4
    m = [i / 10 for i in range(8)] * 3
    r = [a + .2 * b for a, b in zip(y, m)]
    adv, groups = group_diagnostics(y, r, [True] * 24, m)
    assert [g["outcome_type"] for g in groups] == ["all_correct", "all_wrong", "mixed"]
    assert all(not g["zero_reward_variance"] for g in groups)
    expected = torch.tensor(m[:8])
    expected = (expected - expected.mean()) / (expected.std() + 5e-4)
    torch.testing.assert_close(adv[:8], expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(adv[8:16], expected, atol=2e-5, rtol=2e-5)


def test_all_invalid_has_zero_advantages_and_separate_flag():
    adv, groups = group_diagnostics([0] * 8, [0] * 8, [False] * 8, [0] * 8)
    assert bool((adv == 0).all())
    assert groups[0]["outcome_type"] == "all_wrong" and not groups[0]["all_valid"]


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(text.encode())}


def test_native_splitter_one_step_empty_segments_and_terminal_separator():
    assert split_steps("  a\n\n\n\n b \n\n") == ["a", "b"]
    e = encode_native(CharacterTokenizer(), "Q", " a\n\n b ", 100)
    assert e["positions"] == [5, 8] and bytes(e["input_ids"]) == b"Q\n\na\n\nb\n\n"
    assert encode_native(CharacterTokenizer(), "Q", "one step", 100)["no_internal_separator"]
    with pytest.raises(ValueError):
        encode_native(CharacterTokenizer(), "Q", "\n\n", 100)
    with pytest.raises(ValueError):
        encode_native(CharacterTokenizer(), "Q", "123456", 5)


def test_actual_pinned_tokenizer_boundaries():
    model_path = os.environ.get("B_PRM_PATH")
    if not model_path:
        pytest.skip("Set B_PRM_PATH after verified asset download")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    for text in ["2+2=4.\n\nTherefore \\boxed{4}.", "The answer is \\boxed{4}",
                 "1. Let x=2.\n\n 2. x^2=4.\n\n\\boxed{4}\n\n", "\\[x=2\\]\n\n\\boxed{2}"]:
        e = encode_native(tok, "Compute two plus two.", text, 4096)
        assert len(e["positions"]) == len(split_steps(text))


def test_continuation_method_resolution_uses_b_accounting():
    from train_b import BContinuation, BTrainer
    from continue_baseline import VerifiedContinuation
    mro = BContinuation.mro()
    assert mro.index(VerifiedContinuation) < mro.index(BTrainer)
