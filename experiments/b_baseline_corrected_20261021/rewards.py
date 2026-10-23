"""Shared live/replay reward path. Imports and construction never load a model."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/reward_repair_v3_20261021"))
from repairs import CanonicalPRMAdapter, repaired_outcome, valid_for_prm, mixed_reward


class RewardLedger:
    def __init__(self, scorer, unit_metadata):
        self.scorer = CanonicalPRMAdapter(scorer)
        self.unit_metadata = unit_metadata
        self.records = []
        self.cost = {}

    def score_records(self, *, completions, completion_counts, terminated,
                      answer, problem_id, problem, step=0):
        n = len(completions)
        if not n or any(len(x) != n for x in
                        [completion_counts, terminated, answer, problem_id, problem]):
            raise ValueError("Reward inputs must preserve every original response")
        records, selected = [], []
        for i in range(n):
            result = repaired_outcome(completions[i], answer[i], terminated[i],
                question=problem[i], problem_id=problem_id[i], unit_metadata=self.unit_metadata)
            valid = valid_for_prm(result)
            records.append(dict(optimizer_step_before_update=step, problem_id=problem_id[i],
                gold=answer[i], completion=completions[i], completion_tokens=completion_counts[i],
                terminated=terminated[i], status=result["status"], answer=result["answer"],
                outcome_reward=result["reward"], valid_for_process_reward=valid,
                step_scores=[], min_score=0.0, step_count=0, prm_input_tokens=0,
                raw_no_double_newline="\n\n" not in completions[i].strip()))
            if valid:
                selected.append(i)
        scores, self.cost = self.scorer.score([problem[i] for i in selected],
                                              [completions[i] for i in selected])
        if len(scores) != len(selected):
            raise ValueError("Scorer returned a different number of responses")
        for i, result in zip(selected, scores):
            records[i].update(result)
        for row in records:
            row["reward"] = mixed_reward({"status": row["status"],
                "reward": row["outcome_reward"]}, row["step_scores"])
        self.records = records
        return [row["reward"] for row in records]

    def callback(self, eos_token_id, owner):
        def outcome_and_process(prompts, completions, completion_ids, answer,
                                problem_id, problem, **unused):
            return self.score_records(completions=completions,
                completion_counts=[len(ids) for ids in completion_ids],
                terminated=[eos_token_id in ids for ids in completion_ids],
                answer=answer, problem_id=problem_id, problem=problem,
                step=owner["trainer"].state.global_step)
        return outcome_and_process
