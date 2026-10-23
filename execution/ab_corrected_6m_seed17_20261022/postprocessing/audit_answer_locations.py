"""Read-only structural diagnostic; does not rescore or relabel answers."""
import collections
import hashlib
import json
from pathlib import Path
import re


def boxes(text):
    result = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth = 1
        for i in range(match.end(), len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                result.append({"start": match.start(), "end": i + 1,
                               "text": text[match.end():i].strip()})
                break
        else:
            result.append({"start": match.start(), "end": None, "text": None})
    return result


def category(row):
    if row["reward"] == 1:
        return "correct"
    if row["finish_reason"] == "length" or row["status"] == "truncated":
        return "truncated"
    if row["status"] in ("missing_or_invalid_box", "unparseable", "verifier_error"):
        return "unparsable"
    return "complete_incorrect"


def run(base):
    result = {"scope": "Structural box-location flags only; never new accuracy or mathematical ground truth",
              "definitions": {
                  "early_last_box": "last complete nonempty box ends within first25% of character length, with at least100 characters after it",
                  "starts_with_box": "first box starts within first80 characters",
                  "different_box_texts": "at least2 complete nonempty box strings differ; equivalent mathematical expressions can differ",
              }, "datasets": {}, "inputs_sha256": {}}
    for ds in ("math500", "gsm8k"):
        result["datasets"][ds] = {}
        for arm in ("A", "B"):
            path = base / "raw" / f"runs/{arm}_corrected_seed17_6M_20261022/evaluation/{ds}.jsonl"
            raw = path.read_bytes()
            result["inputs_sha256"][str(path)] = hashlib.sha256(raw).hexdigest()
            counts = collections.defaultdict(collections.Counter)
            flagged = []
            for row in map(json.loads, raw.decode().splitlines()):
                text = row["completion"]
                bs = boxes(text)
                last = bs[-1] if bs else None
                early = bool(last and last["text"] and last["end"] / max(1, len(text)) <= .25 and len(text) - last["end"] >= 100)
                flags = {"starts_with_box": bool(bs and bs[0]["start"] < 80),
                         "early_last_box": early,
                         "different_box_texts": len({b["text"] for b in bs if b["text"]}) > 1,
                         "contains_python_block": "```python" in text,
                         "contains_output_block": "```output" in text}
                for cat in ("all", category(row)):
                    counts[cat]["n"] += 1
                    counts[cat].update({k: int(v) for k, v in flags.items()})
                if early:
                    flagged.append({"problem_id": row["problem_id"], "category": category(row),
                                    "box": last, "length_chars": len(text), "tail": text[-500:]})
            result["datasets"][ds][arm] = {"counts": dict(counts), "early_last_box_cases": flagged}
    return result


if __name__ == "__main__":
    assert boxes(r"x \boxed{\frac{1}{2}} y")[-1]["text"] == r"\frac{1}{2}"
    assert boxes(r"\boxed{1} then \boxed{")[-1]["text"] is None
    assert boxes(r"\boxed{}")[-1]["text"] == ""
    base = Path(__file__).resolve().parents[1] / "final_evidence"
    result = run(base)
    output = base / "case_review" / "answer_locations.json"
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({ds: {arm: v["counts"] for arm, v in data.items()}
                      for ds, data in result["datasets"].items()}, indent=2))
