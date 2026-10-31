"""Frozen native H4 token-classifier scoring; no learned component is added."""
import math
import time
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

SEPARATOR = "\n\n"
VALID_STATUSES = {"exact", "symbolic"}


def split_steps(text):
    return [s.strip() for s in text.split(SEPARATOR) if s.strip()]


def mixed_reward(result, step_scores, coefficient=0.2):
    if result["status"] not in VALID_STATUSES:
        assert not step_scores, "Invalid responses must not receive process reward"
        return 0.0
    if not step_scores or not all(math.isfinite(p) and 0 <= p <= 1 for p in step_scores):
        raise ValueError("A valid response requires finite native-step probabilities")
    return float(result["reward"]) + coefficient * min(step_scores)


def encode_native(tokenizer, question, completion, max_length):
    """Exactly match model-card prefix tokenization; causal full-pass gathering.

    Explicitly compare every prefix with the full sequence, so whitespace/BPE
    merging cannot silently move the scoring position into the next step.
    """
    steps = split_steps(completion)
    if not steps:
        raise ValueError("Empty native step sequence")
    prefixes = [SEPARATOR.join((question, *steps[:i])) + SEPARATOR for i in range(1, len(steps) + 1)]
    ids = tokenizer(prefixes[-1], add_special_tokens=True)["input_ids"]
    if len(ids) > max_length:
        raise ValueError(f"PRM input {len(ids)} exceeds native context {max_length}; no truncation")
    positions = []
    for prefix in prefixes:
        prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Native prefix is not a token prefix; boundary requires investigation")
        positions.append(len(prefix_ids) - 1)
    assert len(set(positions)) == len(steps) and positions[-1] == len(ids) - 1
    return {"input_ids": ids, "positions": positions, "steps": steps,
            "no_internal_separator": len(steps) == 1, "raw_no_double_newline": SEPARATOR not in completion.strip()}


class FrozenPRM:
    def __init__(self, model_path, device="cuda:0", microbatch_size=4):
        self.device = torch.device(device)
        self.microbatch_size = microbatch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="right")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        assert self.model.config.num_labels == 2
        self.model.requires_grad_(False).eval()
        self.model.config.use_cache = False
        self.max_length = self.model.config.max_position_embeddings
        self.versions = {n: p._version for n, p in self.model.named_parameters()}
        self.assert_frozen()

    def assert_frozen(self):
        assert not self.model.training
        for n, p in self.model.named_parameters():
            assert not p.requires_grad and p.grad is None and p._version == self.versions[n]

    @torch.inference_mode()
    def score(self, questions, completions):
        assert len(questions) == len(completions)
        self.assert_frozen()
        start = time.perf_counter()
        encoded = [encode_native(self.tokenizer, q, c, self.max_length) for q, c in zip(questions, completions)]
        results = []
        padded_tokens = 0
        for offset in range(0, len(encoded), self.microbatch_size):
            batch = encoded[offset:offset + self.microbatch_size]
            inputs = self.tokenizer.pad([{"input_ids": b["input_ids"]} for b in batch], padding=True,
                                        return_tensors="pt").to(self.device)
            logits = self.model(**inputs, use_cache=False).logits
            assert torch.isfinite(logits).all()
            probabilities = logits.float().softmax(dim=-1)[..., 1]
            padded_tokens += inputs["input_ids"].numel()
            for i, row in enumerate(batch):
                scores = probabilities[i, row["positions"]].cpu().tolist()
                results.append({"step_scores": scores, "min_score": min(scores), "step_count": len(scores),
                                "prm_input_tokens": len(row["input_ids"]),
                                "no_internal_separator": row["no_internal_separator"],
                                "raw_no_double_newline": row["raw_no_double_newline"]})
        self.assert_frozen()
        return results, {"prm_forward_seconds": time.perf_counter() - start,
                         "prm_input_tokens": sum(len(r["input_ids"]) for r in encoded),
                         "prm_padded_tokens": padded_tokens, "prm_scored_completions": len(encoded)}


def group_diagnostics(outcomes, rewards, valid, process_scores, group_size=8):
    y = torch.as_tensor(outcomes, dtype=torch.float32).reshape(-1, group_size)
    r = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, group_size)
    m = torch.as_tensor(process_scores, dtype=torch.float32).reshape(-1, group_size)
    v = torch.as_tensor(valid, dtype=torch.bool).reshape(-1, group_size)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
    result = []
    for i in range(len(y)):
        count = int(y[i].sum())
        kind = "all_correct" if count == group_size else ("all_wrong" if count == 0 else "mixed")
        result.append({"outcome_type": kind, "all_valid": bool(v[i].all()), "correct": count,
                       "mean_abs_advantage": float(adv[i].abs().mean()),
                       "advantage_std": float(adv[i].std()), "zero_reward_variance": bool(r[i].std() == 0),
                       "outcome_mean": float(y[i].mean()), "outcome_std": float(y[i].std()),
                       "reward_mean": float(r[i].mean()), "reward_std": float(r[i].std()),
                       "process_mean": float(m[i].mean()), "process_std": float(m[i].std()),
                       "process_score_semantics": "zero for invalid response; use all_valid strata for PRM-only interpretation"})
    return adv.reshape(-1), result
