from common import last_boxed
from reward import outcome

def test_nested_fraction():
    assert last_boxed(r"Thus \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"
    assert outcome(r"Thus \boxed{\frac{1}{2}}.", "0.5", True)["reward"] == 1

def test_last_answer_and_incomplete_answer():
    assert outcome(r"Initially \boxed{3}; corrected: \boxed{4}.", "3", True)["reward"] == 0
    assert outcome(r"Initially \boxed{3}; then \boxed{4", "3", True)["reward"] == 0

def test_incomplete_rollout_gets_zero():
    assert outcome(r"\boxed{42}", "42", False)["reward"] == 0

def test_no_credit_for_unboxed_or_wrong_answer():
    assert outcome("The answer is 42.", "42", True)["reward"] == 0
    assert outcome(r"\boxed{43}", "42", True)["reward"] == 0

def test_symbolic_equivalence():
    assert outcome(r"\boxed{\sqrt{8}}", r"2\sqrt{2}", True)["reward"] == 1
    assert outcome(r"\boxed{(1,2)}", "(2,1)", True)["reward"] == 0

def test_correct_plain_text_answer():
    assert outcome(r"\boxed{\text{no solution}}", r"\text{no solution}", True)["reward"] == 1
