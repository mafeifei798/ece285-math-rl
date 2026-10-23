"""CPU-only deterministic outcome replay; no model generation or updates."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

from v2core import atomic_json, canonical_view, legacy_module, outcome_v2, read_jsonl, sha256, valid_for_prm, verify_frozen_inputs, write_jsonl


def evaluate(task):
    index, row, question, metadata = task
    old = legacy_module("reward").outcome(row["completion"], row["gold"], row["terminated"])
    # Exact memoized reuse of the same legacy function result on unchanged input.
    new = old if metadata["answer_type"] == "legacy" else outcome_v2(
        row["completion"], row["gold"], row["terminated"], question=question, metadata=metadata)
    return {"source_row": index, "problem_id": row["problem_id"], "legacy_recomputed": old,
            "v2_outcome": new, "typed_unit": metadata["answer_type"],
            "logged_label_reproduced": old["reward"] == row["outcome_reward"],
            "logged_validity_reproduced": valid_for_prm(old) == row["valid_for_process_reward"],
            "logged_status_reproduced": old["status"] == row["status"]}


def main():
    source, out, review_file = map(Path, sys.argv[1:4])
    if (out / "outcomes.jsonl").exists():
        raise RuntimeError("Outcome replay already exists; inspect instead of repeating")
    verify_frozen_inputs(out, source)
    started = time.monotonic()
    rows = read_jsonl(source)
    questions = {r["problem_id"]: r for r in read_jsonl(out / "question_lookup.jsonl")}
    metadata = json.loads((out / "unit_metadata.json").read_text())
    tasks = [(i, r, questions[r["problem_id"]]["problem"], metadata[r["problem_id"]]) for i, r in enumerate(rows)]
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(evaluate, tasks, chunksize=8):
            results.append(result)
            if len(results) % 512 == 0:
                atomic_json(out / "outcome_progress.json", {"completed": len(results), "total": len(rows),
                                                            "elapsed_seconds": time.monotonic() - started})
                print(json.dumps({"stage": "outcome_replay", "completed": len(results), "total": len(rows)}), flush=True)
    assert [r["source_row"] for r in results] == list(range(len(rows)))
    write_jsonl(out / "outcomes.jsonl", results)
    changed = json.loads((out / "changed_input_inventory.json").read_text())
    groups = {r["source_row"] // 8 for r in changed
              if rows[r["source_row"]]["valid_for_process_reward"]
              or valid_for_prm(results[r["source_row"]]["legacy_recomputed"])
              or valid_for_prm(results[r["source_row"]]["v2_outcome"])}
    # Newly eligible records have no old cached scores and require fresh scoring.
    groups |= {i // 8 for i, r in enumerate(results) if
               (valid_for_prm(r["legacy_recomputed"]) or valid_for_prm(r["v2_outcome"]))
               and not rows[i]["valid_for_process_reward"]}
    reviews = json.loads(review_file.read_text())
    groups |= {r["source_row"] // 8 for r in reviews}
    groups |= {i // 8 for i in (78, 2095, 4975)}
    score_rows = [i for i, r in enumerate(rows) if i // 8 in groups and
                  (r["valid_for_process_reward"] or valid_for_prm(results[i]["legacy_recomputed"])
                   or valid_for_prm(results[i]["v2_outcome"]))]
    atomic_json(out / "scoring_plan.json", {"groups": sorted(groups), "source_rows": score_rows,
               "scored_rows": len(score_rows), "rows": len(rows),
               "selection_basis": "all input-change groups, newly eligible groups, fixed prior 50-review groups and three prior known cases; every eligible partner retained",
               "outcomes_sha256": sha256(out / "outcomes.jsonl"), "new_training": False,
               "new_rollouts": False, "dtype": "bfloat16", "device": "cuda:0",
               "microbatch_size": 4, "paired_fixed_padding": True, "repeat_control_every_n_batches": 32})
    summary = {"complete": True, "rows": len(rows), "elapsed_seconds": time.monotonic() - started,
               "logged_label_differences": sum(not r["logged_label_reproduced"] for r in results),
               "logged_validity_differences": sum(not r["logged_validity_reproduced"] for r in results),
               "logged_status_differences": sum(not r["logged_status_reproduced"] for r in results),
               "unit_label_changes": sum(r["legacy_recomputed"]["reward"] != r["v2_outcome"]["reward"] for r in results),
               "newly_eligible": sum(valid_for_prm(r["v2_outcome"]) and not rows[i]["valid_for_process_reward"] for i, r in enumerate(results)),
               "scoring_rows": len(score_rows), "scoring_groups": len(groups)}
    atomic_json(out / "outcome_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
