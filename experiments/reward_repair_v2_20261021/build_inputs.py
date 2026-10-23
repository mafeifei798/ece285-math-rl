"""Freeze unit metadata, source pins and replay rows before viewing repaired scores."""
import json
from pathlib import Path
import re
import sys

from v2core import HERE, VERSION, atomic_json, canonical_view, make_metadata, read_jsonl, sha256, write_jsonl

# Reviewed from question/gold only, without policy scores. Explicit percent-output
# questions; merely mentioning a discount, a probability or '%' is insufficient.
PERCENT_IDS = {
    "prealgebra/1087", "prealgebra/407", "counting_and_probability/485", "prealgebra/980",
    "counting_and_probability/539", "algebra/1013", "geometry/596", "prealgebra/100",
    "prealgebra/307", "counting_and_probability/634", "geometry/389", "geometry/336",
    "algebra/1374", "prealgebra/123", "algebra/1648", "algebra/1475",
    "counting_and_probability/578", "prealgebra/95", "prealgebra/841", "prealgebra/433",
    "prealgebra/118", "prealgebra/425", "geometry/700", "prealgebra/641", "prealgebra/424",
    "number_theory/738", "number_theory/303", "prealgebra/580", "algebra/1503",
    "prealgebra/244", "prealgebra/951", "algebra/1290", "prealgebra/1119", "geometry/795",
    "prealgebra/68", "prealgebra/900", "geometry/867", "counting_and_probability/741",
    "algebra/1591", "prealgebra/717", "prealgebra/597", "geometry/789", "prealgebra/813", "algebra/1464",
}
POINT_IDS = {"algebra/994"}
QUARANTINE_ID = "math/train/number_theory/551"


def main():
    root, source, out = map(Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "input_manifest.json").exists():
        raise RuntimeError("Inputs already frozen; inspect instead of regenerating")
    train = read_jsonl(source / "train.jsonl")
    rows = read_jsonl(source / "rollouts.jsonl")
    assert len(rows) % 8 == 0
    questions = {r["problem_id"]: r for r in train}
    metadata = {}
    for q in train:
        suffix = q["problem_id"].removeprefix("math/train/")
        kind = "percent_value" if suffix in PERCENT_IDS else "percentage_points" if suffix in POINT_IDS else "legacy"
        metadata[q["problem_id"]] = {**make_metadata(q["problem"], q["answer"], kind),
                                    "review_basis": "explicit requested output unit; question/gold reviewed" if kind != "legacy" else "unchanged legacy verifier"}
    assert len([m for m in metadata.values() if m["answer_type"] != "legacy"]) == len(PERCENT_IDS | POINT_IDS)
    quarantine = {QUARANTINE_ID: {**make_metadata(questions[QUARANTINE_ID]["problem"], questions[QUARANTINE_ID]["answer"]),
        "reason": "No largest positive multiple exists: multiplying any valid n by 10 preserves the 0/8 digits and divisibility. Dataset solution instead finds a smallest n.",
        "action": "exclude whole question group prospectively in every arm; do not rewrite original data"}}
    units_candidates = [{"problem_id": q["problem_id"], "problem": q["problem"], "gold": q["answer"],
                         "answer_type": metadata[q["problem_id"]]["answer_type"]}
                        for q in train if re.search(r"percent|percentage|%", q["problem"], re.I)]
    # Composite answer needs a separate gold-canonicalization decision and is
    # deliberately unchanged, rather than accepting an unreviewed transformation.
    atomic_json(out / "unit_metadata.json", metadata)
    atomic_json(out / "unit_candidate_inventory.json", units_candidates)
    atomic_json(out / "quarantine.json", quarantine)
    write_jsonl(out / "question_lookup.jsonl", train)
    changed = []
    for i, r in enumerate(rows):
        assert r["problem_id"] in questions and r["gold"] == questions[r["problem_id"]]["answer"]
        if i % 8 == 0:
            assert len({x["problem_id"] for x in rows[i:i + 8]}) == 1
        old = canonical_view(r["completion"], False)
        new = canonical_view(r["completion"], True)
        if old["steps"] != new["steps"]:
            changed.append({"source_row": i, "problem_id": r["problem_id"],
                            "legacy_scored": r["valid_for_process_reward"], "old_steps": len(old["steps"]),
                            "new_steps": len(new["steps"]), "crlf_count": new["crlf_replacements"]})
    atomic_json(out / "changed_input_inventory.json", changed)
    manifest = {"version": VERSION, "source": {"rollouts": sha256(source / "rollouts.jsonl"), "train": sha256(source / "train.jsonl")},
                "rows": len(rows), "groups": len(rows) // 8, "training_questions": len(train),
                "group_size": 8, "epsilon": 1e-4, "process_coefficient": 0.2,
                "metadata_sha256": sha256(out / "unit_metadata.json"), "quarantine_sha256": sha256(out / "quarantine.json"),
                "changed_inventory_sha256": sha256(out / "changed_input_inventory.json"),
                "typed_questions": sum(m["answer_type"] != "legacy" for m in metadata.values()),
                "known_deferred_metadata": {"math/train/counting_and_probability/484": "composite gold 0.56 = 56%; keep legacy; not observed in B-6M"},
                "ranking_policy": "within original 8-response groups; descending midranks; all partners retained",
                "sign_deadband": 1e-6, "material_advantage_change": 0.01,
                "new_training": False, "new_rollouts": False, "source_root": str(root),
                "inference_scope": "all complete PRM-eligible partners in any group whose native PRM input changes; legacy and normalized encodings scored on same frozen model/backend and padding shape"}
    atomic_json(out / "input_manifest.json", manifest)
    print(json.dumps({**manifest, "changed_native_input_rows": len(changed),
                      "changed_native_input_scored_rows": sum(x["legacy_scored"] for x in changed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
