"""Frozen-score audit utilities. No actor, optimizer or training entry point."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reward_repair_v2_20261021"))
from v2core import atomic_json, canonical_view, encode_native, read_jsonl, sha256, write_jsonl

PIN = "0610740060112df12585d00a1c5f4624d2f59051"
SOURCE_SHA = "2d5309f259bf454d9e446dea750a9bcbea6e3c1349ae483c9862e0beac540228"
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
GROUP = 8
EPS = 1e-4
LAMBDA = 0.2


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def encode_qwen(tokenizer, question, completion, normalize=False):
    view = canonical_view(completion, normalize)
    if not view["steps"]:
        raise ValueError("No steps")
    if "<extra_0>" in question or "<extra_0>" in completion:
        raise ValueError("Literal model scoring token in source text")
    marker = tokenizer.encode("<extra_0>", add_special_tokens=False)
    assert len(marker) == 1
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question},
                {"role": "assistant", "content": "<extra_0>".join(view["steps"]) + "<extra_0>"}]
    serialized = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(serialized, add_special_tokens=False)
    positions = [i for i, v in enumerate(ids) if v == marker[0]]
    assert len(positions) == len(view["steps"])
    return {"input_ids": ids, "positions": positions, "steps": view["steps"], "spans": view["spans"],
            "raw_sha256": view["raw_sha256"], "input_sha256": text_sha(serialized),
            "crlf_replacements": view["crlf_replacements"]}


def midranks(values, descending=True):
    result = []
    for value in values:
        higher = sum((v > value) if descending else (v < value) for v in values)
        equal = sum(v == value for v in values)
        result.append(1 + higher + (equal - 1) / 2)
    return result


def cpu_advantages(values):
    """Independent Python reference, after matching TRL's FP32 reward quantization."""
    import struct
    values = [struct.unpack('f', struct.pack('f', float(v)))[0] for v in values]
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return [(v - mean) / (std + EPS) for v in values]


def auc(positive, negative):
    if not positive or not negative:
        return None
    return sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def verified_inputs(root, out):
    source = root / "runs/B_seed17_6M/rollouts.jsonl"
    assert sha256(source) == SOURCE_SHA
    manifest = json.loads((out / "input_manifest.json").read_text())
    for rel, checksum in manifest["inputs"].items():
        assert sha256(root / rel) == checksum, rel
    return manifest
