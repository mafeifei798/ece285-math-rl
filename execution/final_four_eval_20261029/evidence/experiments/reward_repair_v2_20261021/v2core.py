"""Versioned reward/input/audit repairs. No optimizer, actor, or training entry point."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
VERSION = "reward-repair-v2-20261021"
GROUP_SIZE = 8
EPSILON = 1e-4
PROCESS_COEFFICIENT = 0.2
VALID_STATUSES = {"exact", "symbolic", "unit_exact", "unit_symbolic"}
_MODULES = {}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_frozen_inputs(output, source_rollouts):
    output = Path(output)
    manifest = json.loads((output / "input_manifest.json").read_text())
    expected = {"unit_metadata.json": manifest["metadata_sha256"],
                "quarantine.json": manifest["quarantine_sha256"],
                "changed_input_inventory.json": manifest["changed_inventory_sha256"],
                "question_lookup.jsonl": manifest["source"]["train"]}
    for name, value in expected.items():
        if sha256(output / name) != value:
            raise ValueError("Frozen replay input changed: " + name)
    if sha256(source_rollouts) != manifest["source"]["rollouts"]:
        raise ValueError("Saved rollout source changed")
    return manifest


def legacy_module(name):
    """Import exact archived code, without depending on mutable original sources."""
    if name in _MODULES:
        return _MODULES[name]
    relative = {"common": "training/common.py", "reward": "training/reward.py",
                "prm_reward": "training_b/prm_reward.py"}[name]
    path = HERE / "legacy_snapshot" / relative
    manifest = json.loads((HERE / "legacy_snapshot/source_manifest.json").read_text())
    if sha256(path) != manifest[relative]:
        raise RuntimeError("Archived original changed: " + relative)
    spec = importlib.util.spec_from_file_location("_repair_v2_original_" + name, path)
    module = importlib.util.module_from_spec(spec)
    if name == "reward":
        sentinel = object()
        prior = sys.modules.get("common", sentinel)
        sys.modules["common"] = legacy_module("common")
        try:
            spec.loader.exec_module(module)
        finally:
            if prior is sentinel:
                del sys.modules["common"]
            else:
                sys.modules["common"] = prior
    else:
        spec.loader.exec_module(module)
    _MODULES[name] = module
    return module


def make_metadata(question, gold, kind="legacy"):
    if kind not in {"legacy", "percent_value", "percentage_points", "dimensionless"}:
        raise ValueError("Unknown answer type")
    return {"answer_type": kind, "question_text_sha256": text_sha256(question),
            "gold_text_sha256": text_sha256(gold)}


def outcome_v2(text, gold, terminated, *, question, metadata):
    """Use frozen question units; never infer units from this model response."""
    kind = metadata["answer_type"]
    if metadata["question_text_sha256"] != text_sha256(question):
        raise ValueError("Question does not match frozen answer-unit metadata")
    if metadata["gold_text_sha256"] != text_sha256(gold):
        raise ValueError("Gold does not match frozen answer-unit metadata")
    legacy = legacy_module("reward")
    if kind in {"legacy", "dimensionless"}:
        return legacy.outcome(text, gold, terminated)
    if kind not in {"percent_value", "percentage_points"}:
        raise ValueError("Unsupported answer type: " + kind)
    if not terminated:
        return {"reward": 0.0, "status": "truncated", "answer": None}
    answer = legacy_module("common").last_boxed(text)
    if answer is None or len(answer) > 2048:
        return {"reward": 0.0, "status": "missing_or_invalid_box", "answer": answer}

    def in_requested_unit(value):
        # Only an explicit trailing percentage marker is removed, within a
        # question whose requested output unit is already frozen as percent.
        cleaned = value.strip()
        cleaned = re.sub(r"(?:\\%|%)\s*$", "", cleaned).strip()
        return cleaned

    predicted_unit = in_requested_unit(answer)
    gold_unit = in_requested_unit(gold)
    correct, mode = legacy.equivalent(predicted_unit, gold_unit)
    status = {"exact": "unit_exact", "symbolic": "unit_symbolic"}.get(mode, mode)
    return {"reward": float(correct), "status": status, "answer": answer,
            "answer_type": kind, "prediction_in_requested_unit": predicted_unit,
            "gold_in_requested_unit": gold_unit}


def valid_for_prm(result):
    return result["status"] in VALID_STATUSES


def mixed_reward(result, step_scores):
    if not valid_for_prm(result):
        if step_scores:
            raise ValueError("Unscored/invalid output must not have process scores")
        return 0.0
    if not step_scores or not all(math.isfinite(p) and 0 <= p <= 1 for p in step_scores):
        raise ValueError("Valid response needs finite [0,1] native-step scores")
    return float(result["reward"]) + PROCESS_COEFFICIENT * min(step_scores)


def canonical_view(raw, normalize_crlf=True):
    """Return PRM-only view with a character-to-raw offset map, not edited CoT."""
    characters, offsets = [], []
    i = 0
    replacements = 0
    while i < len(raw):
        if normalize_crlf and raw[i:i + 2] == "\r\n":
            characters.append("\n")
            offsets.append(i)
            i += 2
            replacements += 1
        else:
            characters.append(raw[i])
            offsets.append(i)
            i += 1
    offsets.append(len(raw))
    view = "".join(characters)
    spans, steps = [], []
    start = 0
    boundaries = [m.start() for m in re.finditer("\n\n", view)] + [len(view)]
    for end in boundaries:
        chunk = view[start:end]
        stripped = chunk.strip()
        if stripped:
            left = start + len(chunk) - len(chunk.lstrip())
            right = end - (len(chunk) - len(chunk.rstrip()))
            steps.append(stripped)
            spans.append({"canonical_start": left, "canonical_end": right,
                          "raw_start": offsets[left], "raw_end": offsets[right]})
        start = end + 2
    assert steps == [s.strip() for s in view.split("\n\n") if s.strip()]
    return {"text": view, "steps": steps, "spans": spans, "crlf_replacements": replacements,
            "raw_sha256": text_sha256(raw), "canonical_sha256": text_sha256(view),
            "splitter": "native-double-newline-crlf-v2" if normalize_crlf else "native-double-newline-v1"}


def encode_native(tokenizer, question, completion, max_length, *, normalize_crlf):
    view = canonical_view(completion, normalize_crlf)
    steps = view["steps"]
    if not steps:
        raise ValueError("Empty native-step sequence")
    prefixes = ["\n\n".join((question, *steps[:i])) + "\n\n" for i in range(1, len(steps) + 1)]
    ids = tokenizer(prefixes[-1], add_special_tokens=True)["input_ids"]
    if len(ids) > max_length:
        raise ValueError(f"PRM input {len(ids)} exceeds {max_length}; no silent truncation")
    positions = []
    for prefix in prefixes:
        prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Native prefix is not a full-input token prefix")
        positions.append(len(prefix_ids) - 1)
    if len(set(positions)) != len(steps) or positions[-1] != len(ids) - 1:
        raise ValueError("Invalid native scoring positions")
    return {"input_ids": ids, "positions": positions, "steps": steps, "spans": view["spans"],
            "crlf_replacements": view["crlf_replacements"], "splitter": view["splitter"],
            "raw_sha256": view["raw_sha256"], "canonical_sha256": view["canonical_sha256"]}


def advantages_like_trl(rewards, group_size=GROUP_SIZE):
    """Keep input device/dtype and pinned TRL flatten/repeat reduction layout."""
    grouped_mean = rewards.view(-1, group_size).mean(dim=1)
    grouped_mean = grouped_mean.repeat_interleave(group_size, dim=0)
    advantages = rewards - grouped_mean
    grouped_std = rewards.view(-1, group_size).std(dim=1)
    grouped_std = grouped_std.repeat_interleave(group_size, dim=0)
    return advantages / (grouped_std + EPSILON)


def audit_advantages(rewards, actual, *, receipt_path, group_size=GROUP_SIZE, problem_ids=None):
    """Read-only audit of actual TRL tensors; save evidence before raising.

    CPU float64 is a diagnostic, not a brittle cross-device hard assertion.
    Hard comparison uses the actual reward tensor's device/dtype and TRL layout.
    The fixed 8*machine-epsilon comparison allowance is verified offline.
    """
    import torch
    receipt = {"version": VERSION, "changes_actual_advantages": False,
               "group_size": group_size, "epsilon": EPSILON,
               "reward_device": str(rewards.device), "reward_dtype": str(rewards.dtype),
               "actual_device": str(actual.device), "actual_dtype": str(actual.dtype)}

    def safe_values(tensor):
        values = tensor.detach().cpu().flatten().tolist()
        return [v if math.isfinite(v) else str(v) for v in values]

    receipt.update(rewards=safe_values(rewards), actual_advantages=safe_values(actual))
    errors = []
    if rewards.ndim != 1 or actual.shape != rewards.shape or rewards.numel() % group_size:
        errors.append("shape_or_group_size")
    if rewards.dtype != torch.float32 or rewards.dtype != actual.dtype or rewards.device != actual.device:
        errors.append("dtype_or_device")
    if not torch.isfinite(rewards).all() or not torch.isfinite(actual).all():
        errors.append("nonfinite")
    if problem_ids is not None:
        if len(problem_ids) != rewards.numel() or any(
            len(set(problem_ids[i:i + group_size])) != 1 for i in range(0, len(problem_ids), group_size)
        ):
            errors.append("question_group_alignment")
        receipt["problem_ids"] = list(problem_ids)
    if not errors:
        with torch.no_grad():
            expected = advantages_like_trl(rewards.detach(), group_size)
            tolerance = 8 * torch.finfo(rewards.dtype).eps
            close = torch.isclose(actual.detach(), expected, atol=tolerance, rtol=tolerance)
            cpu64 = advantages_like_trl(rewards.detach().cpu().double(), group_size)
            receipt.update(expected_advantages=safe_values(expected),
                           cpu_float64_advantages=safe_values(cpu64),
                           max_same_device_error=float((actual.detach() - expected).abs().max()),
                           max_float64_reference_difference=float((actual.detach().cpu().double() - cpu64).abs().max()),
                           hard_atol=tolerance, hard_rtol=tolerance,
                           reference_mean=rewards.detach().view(-1, group_size).mean(1).cpu().tolist(),
                           reference_sample_std=rewards.detach().view(-1, group_size).std(1).cpu().tolist())
            if not close.all():
                errors.append("actual_does_not_match_pinned_TRL_formula")
    receipt.update(passed=not errors, errors=errors)
    atomic_json(receipt_path, receipt)
    if errors:
        raise ValueError("Advantage audit failed; receipt saved: " + ",".join(errors))
    return actual  # Return the identical tensor object; never a replacement.


def midranks(values):
    return [1 + sum(b > a for b in values) + 0.5 * (sum(b == a for b in values) - 1) for a in values]
