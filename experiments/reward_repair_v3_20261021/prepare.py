"""Freeze affected native microbatches and independently replay unit rules."""
import argparse
from collections import Counter
import json
from pathlib import Path
from repairs import *


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--inputs", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--metadata", type=Path, required=True)
    cli.add_argument("--original-manifest", type=Path, required=True)
    args = cli.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "preparation.json").exists(), "Already prepared: do not repeat"
    protected = json.loads(args.original_manifest.read_text())["files"]
    for name, receipt in protected.items():
        assert sha256(args.inputs / name) == receipt["sha256"], name
    metadata = json.loads(args.metadata.read_text())
    atomic_json(out / "unit_metadata.json", metadata)
    questions = {}
    for dataset in ["train", "math500", "gsm8k"]:
        questions.update({r["problem_id"]: r for r in read_jsonl(args.inputs / "data/prepared" / (dataset + ".jsonl"))})
    arms = {"A": "A_seed17", "B_old": "B_seed17_6M", "B7": "B_Qwen7B_seed17_6M"}
    summaries, sources = {}, {}
    for arm, name in arms.items():
        paths = [("train", args.inputs / "runs" / name / "rollouts.jsonl")]
        for dataset in ["math500", "gsm8k"]:
            matches = list((args.inputs / "runs" / name).rglob(dataset + ".jsonl"))
            assert len(matches) == 1, (arm, dataset, matches)
            paths.append((dataset, matches[0]))
        for dataset, path in paths:
            data, judged, changes = read_jsonl(path), [], []
            for i, r in enumerate(data):
                old = {"reward": r.get("outcome_reward", r["reward"]), "status": r["status"], "answer": r["answer"]}
                if r["problem_id"] in metadata:
                    q = questions[r["problem_id"]]
                    assert q["answer"] == r["gold"]
                    terminated = r.get("terminated", r.get("finish_reason") == "stop")
                    new = repaired_outcome(r["completion"], r["gold"], terminated,
                        question=q["problem"], problem_id=r["problem_id"], unit_metadata=metadata)
                else:
                    new = dict(old)  # Frozen legacy-label cache; non-unit code path is unchanged.
                judged.append({"source_row": i, "problem_id": r["problem_id"], "old": old, "new": new})
                if old["reward"] != new["reward"]:
                    changes.append(judged[-1])
                assert valid_for_prm(old) == valid_for_prm(new), "Eligibility changed: fresh missing scores need review"
            key = arm + "_" + dataset
            write_jsonl(out / "outcomes" / (key + ".jsonl"), judged)
            sources[key] = {"relative": str(path.relative_to(args.inputs)), "sha256": sha256(path)}
            summaries[key] = {"n": len(data), "old_correct": sum(r["old"]["reward"] for r in judged),
                "new_correct": sum(r["new"]["reward"] for r in judged), "changes": changes}
            print(key, "changed", len(changes), flush=True)
    data = read_jsonl(args.inputs / sources["B7_train"]["relative"])
    changed, inventory, affected_batches = [], [], []
    for i, r in enumerate(data):
        old, new = canonical_view(r["completion"], False), canonical_view(r["completion"], True)
        for text, span in zip(new["steps"], new["spans"]):
            assert r["completion"][span["raw_start"]:span["raw_end"]].replace("\r\n", "\n") == text
        different = old["steps"] != new["steps"]
        if different:
            inventory.append({"source_row": i, "process_eligible": r["valid_for_process_reward"],
                "old_steps": len(old["steps"]), "new_steps": len(new["steps"]),
                "replacements": new["crlf_replacements"], "raw_sha256": new["raw_sha256"]})
        if different and r["valid_for_process_reward"]:
            changed.append(i)
    changed_set = set(changed)
    for start in range(0, len(data), 32):
        valid = [i for i in range(start, start + 32) if data[i]["valid_for_process_reward"]]
        for offset in range(0, len(valid), 4):
            indices = valid[offset:offset + 4]
            if any(i in changed_set for i in indices):
                affected_batches.append({"optimizer_step": data[start]["optimizer_step_before_update"],
                    "valid_microbatch_index": offset // 4, "indices": indices})
    atomic_json(out / "crlf_inventory.json", inventory)
    atomic_json(out / "unit_summary.json", summaries)
    atomic_json(out / "protection_manifest.json", protected)
    manifest = {"version": VERSION, "inputs": str(args.inputs), "sources": sources,
        "unit_metadata_sha256": sha256(out / "unit_metadata.json"), "new_training": False,
        "new_actor_samples": 0, "PRM_parameter_updates": 0, "group_size": 8,
        "unchanged_reward": "y+0.2*min(p); invalid=0; original GRPO sample std +1e-4",
        "changed_process_rows": changed, "changed_process_count": len(changed),
        "affected_microbatches": affected_batches,
        "paired_scoring_rows": sum(len(b["indices"]) for b in affected_batches),
        "changes_are": "Only explicit percent-output units and PRM-only CRLF normalization. No data quarantine or selection."}
    atomic_json(out / "preparation.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ["affected_microbatches","changed_process_rows","sources"]}, indent=2))


if __name__ == "__main__": main()
