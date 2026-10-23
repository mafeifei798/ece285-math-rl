"""Verify frozen input copy and select a score-blinded uniform 20-record review."""
import hashlib
import json
from pathlib import Path
import random
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "execution/b7b_training_audit_20261021"
INPUTS = OUT / "inputs"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    manifest = read(OUT / "input_manifest.json")
    archive = OUT / "frozen_inputs.tar.gz"
    assert sha(archive) == manifest["archive_sha256"]
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            p = Path(member.name)
            assert not p.is_absolute() and ".." not in p.parts and member.isfile()
            assert member.name in manifest["files"]
        tar.extractall(INPUTS, filter="data")
    for name, receipt in manifest["files"].items():
        p = INPUTS / name
        assert sha(p) == receipt["sha256"] and p.stat().st_size == receipt["bytes"], name
    population = rows(INPUTS / "runs/B_Qwen7B_seed17_6M/rollouts.jsonl")
    questions = {r["problem_id"]: r for r in rows(INPUTS / "data/prepared/train.jsonl")}
    indices = sorted(random.Random(285).sample(range(len(population)), 20))
    selection = {"population": "All 9056 new Bprime training completions, including invalid/truncated outputs",
        "population_n": len(population), "sampling": "uniform without replacement, Python random.Random(285)",
        "seed": 285, "source_sha256": sha(INPUTS / "runs/B_Qwen7B_seed17_6M/rollouts.jsonl"),
        "rows": indices, "reviewer": "Codex mathematical review; not independent human ground truth",
        "blind_fields": ["PRM scores", "logged outcome label", "group rank", "advantage"]}
    save(OUT / "sample_manifest.json", selection)
    selected, text = [], []
    for order, index in enumerate(indices, 1):
        record = population[index]
        question = questions[record["problem_id"]]
        steps = [s.strip() for s in record["completion"].split("\n\n") if s.strip()]
        sample = {"sample_id": f"B{order:02d}", "source_row": index, "question": question["problem"],
            "problem_id": record["problem_id"], "gold": record["gold"], "steps": steps, "record": record}
        selected.append(sample)
        text += [f"===== {sample['sample_id']} row {index} step {record['optimizer_step_before_update']} =====",
                 "QUESTION: " + question["problem"], "GOLD: " + str(record["gold"])]
        text += [f"STEP {i}: {step}" for i, step in enumerate(steps, 1)]
        text += [""]
    save(OUT / "selected_records.json", selected)
    (OUT / "blind_review.txt").write_text("\n".join(text))
    save(OUT / "input_verification.json", {"passed": True, "files": len(manifest["files"]),
        "archive_sha256": manifest["archive_sha256"], "new_training": False})
    print(json.dumps({"verified_files": len(manifest["files"]), "sample_rows": indices,
        "question_fields": list(next(iter(questions.values()))), "review_characters": len("\n".join(text))}))


if __name__ == "__main__":
    main()
