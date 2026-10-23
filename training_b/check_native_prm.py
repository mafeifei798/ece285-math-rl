"""Native-scoring hardware check, with no test-set or training selection."""
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from common import atomic_json
from prm_reward import FrozenPRM, encode_native


def main():
    assets = json.loads((ROOT / "receipts/B_prm_assets.json").read_text())
    prm = FrozenPRM(assets["model_path"])
    question = "Let $a,$ $b,$ and $c$ be positive real numbers.  Find the set of all possible values of\n\\[\\frac{c}{a} + \\frac{a}{b + c} + \\frac{b}{c}.\\]"
    steps = ["This problem involves finding the range of an expression involving three variables.",
             "One possible strategy is to try to eliminate some variables and write the expression in terms of one variable only.",
             "To do this, I might look for some common factors or symmetries in the expression.",
             "I notice that the first and last terms have $c$ in the denominator, so I can factor out $c$ from the whole expression and get\n\\[\\frac{1}{c}\\left(c + \\frac{a^2}{b + c} + b\\right).\\]"]
    completion = "\n\n".join(steps)
    full, costs = prm.score([question], [completion])
    repeated, _ = prm.score([question], [completion])
    scores = full[0]["step_scores"]
    repeat_error = max(abs(a - b) for a, b in zip(scores, repeated[0]["step_scores"]))
    assert repeat_error <= 1e-6
    native = []
    with torch.inference_mode():
        for i in range(1, len(steps) + 1):
            text = "\n\n".join((question, *steps[:i])) + "\n\n"
            inputs = prm.tokenizer(text, return_tensors="pt").to(prm.device)
            native.append(float(prm.model(**inputs, use_cache=False).logits[0, -1].float().softmax(-1)[1]))
    error = max(abs(a - b) for a, b in zip(scores, native))
    assert error <= .025, "Full-pass boundary scores differ from official prefix-at-a-time procedure"
    assert [p >= .5 for p in scores] == [True, True, True, False], "Native model-card label check failed"
    prm.assert_frozen()
    value = {"passed": True, "prm_revision": assets["model_revision"], "native_model_card_scores": native,
             "full_pass_scores": scores, "max_probability_difference": error,
             "repeat_max_difference": repeat_error, "frozen_verified": True, "costs": costs,
             "purpose": "scoring plumbing only; no benchmark-based setting selection", "finished_unix": time.time()}
    atomic_json(ROOT / "receipts/B_native_prm_gate.json", value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
