"""Prospective shared question admission, with raw-question/gold hash binding."""
import json
from pathlib import Path
import sys

from v2core import atomic_json, make_metadata, read_jsonl, sha256, verify_frozen_inputs


def admit_question(problem_id, question, gold, metadata, quarantine):
    expected = metadata[problem_id]
    actual = make_metadata(question, gold, expected["answer_type"])
    for field in ("question_text_sha256", "gold_text_sha256"):
        if actual[field] != expected[field]:
            raise ValueError("Question/answer differs from frozen admission metadata")
    if problem_id in quarantine:
        for field in ("question_text_sha256", "gold_text_sha256"):
            if actual[field] != quarantine[problem_id][field]:
                raise ValueError("Stale quarantine entry; inspect revised question")
        return {"eligible": False, "reason": quarantine[problem_id]["reason"]}
    return {"eligible": True, "reason": None}


def main():
    out, source = map(Path, sys.argv[1:3])
    verify_frozen_inputs(out, source)
    rows = read_jsonl(out / "question_lookup.jsonl")
    metadata = json.loads((out / "unit_metadata.json").read_text())
    quarantine = json.loads((out / "quarantine.json").read_text())
    kept, excluded = [], []
    for q in rows:
        result = admit_question(q["problem_id"], q["problem"], q["answer"], metadata, quarantine)
        (kept if result["eligible"] else excluded).append(q["problem_id"])
    assert len(kept) + len(excluded) == len(rows) and len(excluded) == 1
    # Reusing an ID with edited text must not silently inherit a quarantine rule.
    q = next(q for q in rows if q["problem_id"] == excluded[0])
    rejected_edit = False
    try:
        admit_question(q["problem_id"], q["problem"].replace("largest", "smallest"), q["answer"], metadata, quarantine)
    except ValueError:
        rejected_edit = True
    assert rejected_edit
    atomic_json(out / "training_admission_manifest.json", {"original_training_questions": len(rows),
                "prospective_eligible_questions": len(kept), "eligible_ids": kept, "excluded_ids": excluded,
                "unit_metadata_sha256": sha256(out / "unit_metadata.json"), "quarantine_sha256": sha256(out / "quarantine.json"),
                "same_for_arms": ["A", "B", "C", "D"], "applied_to_original_training": False,
                "training_launched": False, "old_question_data_overwritten": False,
                "stale_question_rejected_test": rejected_edit})
    print(json.dumps({"eligible": len(kept), "excluded": excluded, "stale_question_rejected": rejected_edit, "training": False}))


if __name__ == "__main__":
    main()
