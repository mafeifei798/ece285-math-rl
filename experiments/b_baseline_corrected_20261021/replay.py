"""CPU-only integration replay; real stored rewards, synthetic log-probabilities.

Execute the exact pinned TRL preparation, buffer, compute_loss and _compute_loss
method bodies. Only generation and model log-probabilities are fixtures. No
model is loaded, no answer or PRM score is generated, no backward/step occurs.
"""
import argparse
import ast
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Union, Optional, Sequence
import numpy as np
import torch
from rewards import RewardLedger
from trainer import CorrectedBMixin, PINNED_TRL_SHA256, PINNED_UTILS_SHA256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).open() if s.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def exact_methods():
    source = HERE / "vendor/grpo_trainer_0_23_1.py"
    if sha(source) != PINNED_TRL_SHA256:
        raise ValueError("Pinned trainer source changed")
    if sha(HERE / "vendor/trl_utils_0_23_1.py") != PINNED_UTILS_SHA256:
        raise ValueError("Pinned buffer helper source changed")
    env = dict(torch=torch, Any=Any, Union=Union, Optional=Optional, Sequence=Sequence)
    wanted = {"split_tensor_dict", "shuffle_sequence_dict", "nanmin", "nanmax",
              "split_pixel_values_by_grid", "unsplit_pixel_values_by_grid"}
    utiltree = ast.parse((HERE / "vendor/trl_utils_0_23_1.py").read_text())
    functions = [n for n in utiltree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(functions) != len(wanted):
        raise ValueError("Missing pinned preparation helpers")
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 "pinned_TRL_utils", "exec"), env)
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GRPOTrainer")
    wanted = {"_prepare_inputs", "compute_loss", "_compute_loss"}
    bodies = {}
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name in wanted:
            # Only profiling wrappers are omitted; every executable statement
            # in the TRL method body remains identical.
            bodies[n.name] = hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
            n.decorator_list = []
            exec(compile(ast.fix_missing_locations(ast.Module(body=[n], type_ignores=[])),
                         "pinned_TRL_" + n.name, "exec"), env)
    generate = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_generate_and_score_completions")
    start = next(i for i, n in enumerate(generate.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "rewards" for t in n.targets)
                 and "nansum" in ast.unparse(n))
    end = next(i for i in range(start, len(generate.body)) if isinstance(generate.body[i], ast.If)
               and ast.unparse(generate.body[i].test) == "self.scale_rewards != 'none'")
    code = compile(ast.fix_missing_locations(ast.Module(body=generate.body[start:end + 1], type_ignores=[])),
                   "pinned_TRL_reward_kernel", "exec")
    return env, code, bodies


class CachedScorer:
    """Replay native cached probabilities, strictly in original batch order."""
    def set_batch(self, batch, scores):
        self.batch, self.scores = batch, scores

    def score(self, questions, completions):
        if questions != [r["problem"] for r in self.batch]:
            raise ValueError("PRM question ordering changed")
        if completions != [r["completion"].replace("\r\n", "\n") for r in self.batch]:
            raise ValueError("Canonical PRM input does not match cached native score")
        result = []
        for r, probabilities in zip(self.batch, self.scores):
            steps = [s.strip() for s in r["completion"].replace("\r\n", "\n").split("\n\n") if s.strip()]
            if len(steps) != len(probabilities):
                raise ValueError("Step-to-score alignment changed")
            result.append(dict(step_count=len(steps), step_scores=probabilities,
                               min_score=min(probabilities), prm_input_tokens=0))
        return result, dict(prm_forward_seconds=0.0, prm_input_tokens=0,
                           prm_padded_tokens=0, prm_scored_completions=len(result))


class StoredTokenIDs:
    """Only stored count/EOS membership is available; never invent actor tokens."""
    def __init__(self, row):
        self.row = row

    def __len__(self):
        return self.row["completion_tokens"]

    def __contains__(self, token):
        if token != 42:
            raise ValueError("Replay only knows EOS membership")
        return self.row["terminated"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    cli = p.parse_args()
    out = cli.out.resolve()
    if out.exists() and (out / "complete.json").exists():
        raise ValueError("Completed replay exists; do not overwrite")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(285)
    frozen = ROOT / "execution/reward_repair_v3_20261021"
    if not frozen.exists():
        frozen = ROOT / "audits/reward_repair_v3_20261021"
    original = ROOT / "execution/b7b_training_audit_20261021/inputs"
    if not original.exists():
        original = ROOT
    source = original / "runs/B_Qwen7B_seed17_6M/rollouts.jsonl"
    train = original / "data/prepared/train.jsonl"
    inputs = [source, train, frozen / "variants/combined.jsonl", frozen / "paired_scores.jsonl", frozen / "unit_metadata.json"]
    before = {str(p): sha(p) for p in inputs}
    expected_hashes = json.loads((HERE / "release.json").read_text())["replay_input_hashes"]
    for path in inputs:
        if before[str(path)] != expected_hashes[path.name]:
            raise ValueError("Frozen replay input changed: " + str(path))
    data, fixed = rows(source), rows(frozen / "variants/combined.jsonl")
    paired = {r["source_row"]: r for r in rows(frozen / "paired_scores.jsonl")}
    questions = {r["problem_id"]: r for r in rows(train)}
    if len(data) != 9056 or len(fixed) != len(data):
        raise ValueError("Expected all 9056 original responses")
    for i, r in enumerate(data):
        r["source_row"] = i
        r["problem"] = questions[r["problem_id"]]["problem"]
        assert fixed[i]["source_row"] == i and r["problem_id"] == fixed[i]["problem_id"]
    env, kernel, methods = exact_methods()
    cached = CachedScorer()
    ledger = RewardLedger(cached, json.loads((frozen / "unit_metadata.json").read_text()))

    class Base:
        _prepare_inputs = env["_prepare_inputs"]
        compute_loss = env["compute_loss"]
        _compute_loss = env["_compute_loss"]

        def _generate_and_score_completions(self, batch):
            eligible = [r for r in batch if fixed[r["source_row"]]["process_scored"]]
            cached.set_batch(eligible, [paired[r["source_row"]]["normalized"]["step_scores"]
                if r["source_row"] in paired else r["step_scores"] for r in eligible])
            values = self.callback(prompts=[questions[r["problem_id"]]["prompt"] for r in batch],
                completions=[r["completion"] for r in batch], completion_ids=[StoredTokenIDs(r) for r in batch],
                answer=[r["gold"] for r in batch], problem_id=[r["problem_id"] for r in batch],
                problem=[r["problem"] for r in batch])
            for raw, actual in zip(batch, ledger.records):
                expected = fixed[raw["source_row"]]
                assert actual["completion"] == raw["completion"]
                assert actual["status"] == expected["status"]
                assert actual["outcome_reward"] == expected["outcome_reward"]
                assert actual["min_score"] == (expected["min_score"] or 0.0)
                assert actual["step_count"] == expected["step_count"]
                assert actual["reward"] == expected["reward"]
            e = dict(torch=torch, self=self, device=torch.device("cpu"),
                     rewards_per_func=torch.tensor(values, dtype=torch.float32).reshape(-1, 1))
            exec(kernel, e)
            self._logs["advantages"].extend(e["advantages"].tolist())
            counts = torch.tensor([r["completion_tokens"] for r in batch])
            width = int(counts.max())
            mask = (torch.arange(width).unsqueeze(0) < counts.unsqueeze(1)).long()
            # Token values/log-probabilities are algebra fixtures; masks and
            # lengths come from real saved completions, including truncations.
            return dict(prompt_ids=torch.zeros(len(batch), 1, dtype=torch.long),
                prompt_mask=torch.ones(len(batch), 1, dtype=torch.long),
                completion_ids=torch.zeros(len(batch), width, dtype=torch.long), completion_mask=mask,
                advantages=e["advantages"], num_items_in_batch=mask.sum(),
                old_per_token_logps=torch.full(mask.shape, -2.0),
                ref_per_token_logps=torch.full(mask.shape, -2.1))

        def _get_per_token_logps_and_entropies(self, model, ids, mask, width, **unused):
            shape = (ids.shape[0], width)
            return torch.full(shape, -2.0 + self.log_ratio), torch.full(shape, .7)

    cls = type("ReplayCorrectedB", (CorrectedBMixin, Base), {})
    instance = cls()
    instance.model = SimpleNamespace(training=True)
    instance.accelerator = SimpleNamespace(num_processes=1, gather=lambda x: x)
    instance.args = SimpleNamespace(steps_per_generation=32, delta=None)
    instance.num_generations, instance.num_iterations = 8, 1
    instance.scale_rewards, instance.reward_weights = "group", torch.ones(1)
    instance.use_liger_loss = False
    instance.use_vllm, instance.vllm_importance_sampling_correction = True, False
    instance.top_entropy_quantile, instance.beta = 1.0, .04
    instance.importance_sampling_level, instance.loss_type = "token", "grpo"
    instance.epsilon_low = instance.epsilon_high = .2
    instance.current_gradient_accumulation_steps = 32
    instance._metrics = {"train": defaultdict(list)}
    instance._logs = {"advantages": deque(maxlen=32)}
    instance._step, instance._buffered_inputs = 0, None
    instance.state = SimpleNamespace(global_step=0)
    instance.initialize_baseline(ledger)
    owner = {"trainer": instance}
    instance.callback = ledger.callback(42, owner)
    received, loss_errors, negatives = {}, [], []
    outcome_groups = Counter()
    for start in range(0, len(data), 32):
        batch = data[start:start + 32]
        for _ in range(32):
            prepared = instance._prepare_inputs(batch)
            tag = int(prepared["baseline_trace_id"].item())
            a = float(prepared["advantages"].item())
            assert tag not in received
            received[tag] = a
            for ratio in [0.0, -.7, .5]:
                instance.log_ratio = ratio
                loss = instance.compute_loss(instance.model, prepared)
                now = torch.tensor(-2.0 + ratio)
                old = torch.tensor(-2.0)
                reference = torch.tensor(-2.1)
                coefficient = (now - old).exp()
                clipped = coefficient.clamp(.8, 1.2)
                policy = -torch.minimum(coefficient * prepared["advantages"], clipped * prepared["advantages"])
                kl = (reference - now).exp() - (reference - now) - 1
                expected = (policy + .04 * kl).mean() / 32
                loss_errors.append(float((loss - expected).abs()))
                torch.testing.assert_close(loss, expected, atol=2e-7, rtol=2e-6)
            if start == 0 and len(received) == 1:
                tags = sorted(instance._baseline_expected)[:8]
                original_a = torch.tensor([instance._baseline_expected[i] for i in tags])
                cases = {"second_group_normalization": (original_a - original_a.mean()) / (original_a.std() + 1e-4),
                         "rescale_after_buffer": original_a * 2,
                         "shuffle_advantage_without_tag": original_a.flip(0)}
                for name, changed in cases.items():
                    try:
                        instance._assert_baseline_loss_input(dict(advantages=changed, baseline_trace_id=torch.tensor(tags)))
                    except RuntimeError:
                        negatives.append(name)
                    else:
                        raise AssertionError("Guard missed " + name)
                try:
                    instance._assert_baseline_loss_input(dict(advantages=original_a))
                except RuntimeError:
                    negatives.append("missing_provenance_tag")
        assert list(instance._logs["advantages"]) == [received[i] for i in range(start, start + 32)]
        instance.state.global_step += 1
        if instance.state.global_step % 50 == 0:
            print(json.dumps(dict(batches=instance.state.global_step, rows=len(received))), flush=True)
    assert set(received) == set(range(9056))
    yy = np.array([r["outcome_reward"] for r in fixed], dtype=np.float64).reshape(-1, 8)
    mm = np.array([r["min_score"] or 0 for r in fixed], dtype=np.float64).reshape(-1, 8)
    # Quantize mixed rewards exactly as TRL, then independently reduce in FP64.
    rr = np.array([r["reward"] for r in fixed], dtype=np.float32).astype(np.float64).reshape(-1, 8)
    expected = (rr - rr.mean(1, keepdims=True)) / (rr.std(1, ddof=1, keepdims=True) + 1e-4)
    actual = np.array([received[i] for i in range(9056)]).reshape(-1, 8)
    max_error = float(np.abs(actual - expected).max())
    assert max_error < 2e-5
    stats = {}
    for kind in ["mixed", "all_correct", "all_wrong"]:
        ids = [g for g in range(len(yy)) if
            ("all_correct" if yy[g].sum() == 8 else "all_wrong" if yy[g].sum() == 0 else "mixed") == kind]
        stats[kind] = dict(groups=len(ids), mean_abs_advantage=float(np.abs(actual[ids]).mean()))
        if kind == "mixed":
            stats[kind]["correct_negative_rows"] = int(((actual[ids] < -1e-6) & (yy[ids] == 1)).sum())
            stats[kind]["wrong_positive_rows"] = int(((actual[ids] > 1e-6) & (yy[ids] == 0)).sum())
    assert len(negatives) == 4
    assert instance.accounting["generated_completion_tokens"] == 6014788
    assert instance.baseline_loss_rows == 9056 * 3
    assert before == {str(p): sha(p) for p in inputs}
    with (out / "loss_input_trace.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["source_row", "problem_id", "outcome", "min_score", "advantage_at_loss", "independent_fp64", "abs_error"])
        for i, row in enumerate(fixed):
            writer.writerow([i, row["problem_id"], row["outcome_reward"], row["min_score"], received[i],
                             expected.flatten()[i], abs(received[i] - expected.flatten()[i])])
    result = dict(passed=True, rows=9056, groups=1132, generation_batches=283,
        loss_calls=instance.baseline_loss_calls, tested_ratio_profiles=[0, -.7, .5],
        original_TRL_advantage_tensor_preserved=True, advantages_at_loss_bitwise_equal_to_TRL=True, second_normalization_detected=True,
        negative_controls_detected=negatives, max_vs_independent_fp64=max_error,
        max_loss_error=max(loss_errors), group_stats=stats, accounting=instance.accounting,
        exact_TRL_method_ast_hashes=methods, pinned_TRL_sha256=PINNED_TRL_SHA256,
        synthetic_only=["token identity tensors", "current/old/reference log probabilities", "entropy"],
        real_inputs=["raw completion text", "question/gold", "EOS/length", "native cached step scores", "original 8-way groups"],
        model_loads=0, actor_generation=0, PRM_model_forward_calls=0, backward_calls=0,
        optimizer_steps=0, new_training=False, hashes_unchanged=True,
        limitation="This verifies reward/advantage/buffer/loss integration, not live model loading, GPU numerical behavior, or accuracy improvements.")
    saved = np.array([r["advantage"] for r in fixed]).reshape(-1, 8)
    result["max_vs_saved_v3_advantages"] = float(np.abs(actual - saved).max())
    assert result["max_vs_saved_v3_advantages"] < 2e-5
    assert all(r["advantage_tensor_retained"] for r in instance.baseline_generation_checks)
    result["method_changes"] = False
    result["advantage_rule"] = "original_mixed_reward_group_standardization"
    save(out / "input_manifest.json", before)
    save(out / "complete.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
