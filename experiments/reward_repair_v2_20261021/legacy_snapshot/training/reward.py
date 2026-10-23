"""Outcome-only rule verifier. No process model, format bonus, or length bonus."""
from functools import lru_cache
import re
import unicodedata
from math_verify import LatexExtractionConfig, parse, verify
from common import last_boxed

@lru_cache(maxsize=20000)
def parsed_answer(answer):
    return parse("\\boxed{" + answer + "}",
                 extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
                 fallback_mode="no_fallback", extraction_mode="first_match", parsing_timeout=2)

def equivalent(prediction, gold):
    # Exact matching is useful for non-symbolic dataset answers; no guessed numeric extraction.
    normalize = lambda s: re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))
    if normalize(prediction) == normalize(gold):
        return True, "exact"
    try:
        target, candidate = parsed_answer(gold), parsed_answer(prediction)
        if not target or not candidate:
            return False, "unparseable"
        return bool(verify(target, candidate, timeout_seconds=2)), "symbolic"
    except Exception as exc:
        return False, "verifier_error:" + type(exc).__name__

def outcome(text, gold, terminated):
    if not terminated:
        return {"reward": 0.0, "status": "truncated", "answer": None}
    answer = last_boxed(text)
    if answer is None or len(answer) > 2048:
        return {"reward": 0.0, "status": "missing_or_invalid_box", "answer": answer}
    correct, mode = equivalent(answer, gold)
    return {"reward": float(correct), "status": mode, "answer": answer}
