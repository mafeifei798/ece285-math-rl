"""CPU-only OpenRLHF boundary + normalization transplant for offline replay.

This is NOT a complete OpenRLHF baseline reproduction. The caller supplies raw,
pre-whitening advantages (for example, the existing veRL RLOO output). This
module changes only boundary suppression and the single normalization operation.
It neither changes rewards nor recomputes an OpenRLHF reward baseline.

API: apply_openrlhf_boundary_norm(raw_advantages, valid_mask, boundary_mask)
     -> (CPU float32 tensor with the input shape, JSON-serializable receipt).

The unmodified normalize method is AST-extracted from pinned official source.
No imports from that source, GPU initialization, optimizer, or launcher execute.
Use ``python boundary_adapter.py --self-test`` for CPU-only synthetic checks.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "41f9363b8cc4af044b7fc5f7f9bbbf62139e9553"
SOURCE_PATH = (
    PROJECT_ROOT / "execution/pure_official_fix_audit_20261023/source_openrlhf"
    / "openrlhf/trainer/ppo_utils/replay_buffer.py"
)
SOURCE_SHA256 = "421b73119fd800fb503abe107cc6549ad418d5908cfae42b776224fde715667d"
ORACLE_ATOL = 2e-5
ORACLE_RTOL = 2e-5


class _IdentityStrategy:
    """Single-process equivalent of the pinned method's distributed sum."""

    def __init__(self):
        self.calls = []

    def all_reduce(self, value, operation):
        if operation != "sum":
            raise ValueError(f"Unexpected reduction: {operation!r}")
        self.calls.append({"operation": operation, "numel": value.numel()})
        return value


class _OfflineBuffer:
    def __init__(self, items):
        self.items = items
        self.mask_separator_adv = True

    def __iter__(self):
        return iter(self.items)


def _official_normalize():
    source_bytes = SOURCE_PATH.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Pinned official source hash mismatch: {digest}")
    tree = ast.parse(source_bytes.decode("utf-8"), filename=str(SOURCE_PATH))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NaiveReplayBuffer"]
    if len(classes) != 1:
        raise ValueError("Expected exactly one official NaiveReplayBuffer class")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "normalize"]
    if len(methods) != 1:
        raise ValueError("Expected exactly one official normalize method")
    method = copy.deepcopy(methods[0])
    # Compile only this function. Never execute the source module or constructor.
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE_PATH), "exec"), namespace)
    return namespace["normalize"], {
        "commit": SOURCE_COMMIT,
        "file": str(SOURCE_PATH),
        "sha256": digest,
        "method": "NaiveReplayBuffer.normalize",
        "line_start": method.lineno,
        "line_end": method.end_lineno,
        "ast_extracted_method_only": True,
    }


def _cpu_tensor(value, name, *, mask=False):
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise ValueError(f"{name} must already be on CPU; no GPU access is allowed")
    tensor = torch.as_tensor(value, device="cpu")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional [batch, token] array")
    if mask:
        if not torch.all((tensor == 0) | (tensor == 1)).item():
            raise ValueError(f"{name} must contain only boolean / 0 / 1 entries")
        return tensor.to(dtype=torch.bool).clone()
    return tensor.detach().to(dtype=torch.float32).clone()


def _float64_oracle(rows):
    """Independent scalar formula; faithfully includes zero slots in variance.

    Official statistics exclude original zeros from the count, but still include
    their squared deviation from the mean in the numerator. Afterwards those
    original zero slots are restored to zero. This is intentionally NOT ordinary
    whitening over nonboundary tokens.
    """
    numbers = [[float(x) for x in row] for row in rows]
    flat = [x for row in numbers for x in row]
    count = sum(x != 0.0 for x in flat)
    if count == 0:
        raise ValueError("All valid advantages are zero after boundary masking; official normalizer denominator is zero")
    mean = math.fsum(flat) / count
    variance = math.fsum((x - mean) ** 2 for x in flat) / count
    reciprocal_std = 1.0 / math.sqrt(max(variance, 1e-8))
    normalized = [
        [0.0 if x == 0.0 else (x - mean) * reciprocal_std for x in row]
        for row in numbers
    ]
    return normalized, {"nonzero_count": count, "mean": mean, "variance": variance,
                        "reciprocal_std": reciprocal_std,
                        "variance_includes_initial_zero_slots": True,
                        "variance_denominator_excludes_initial_zero_slots": True,
                        "variance_floor": 1e-8, "bessel_correction": False}


@torch.no_grad()
def apply_openrlhf_boundary_norm(raw_advantages, valid_mask, boundary_mask, *,
                                input_stage="raw_pre_whitening"):
    """Suppress boundary advantages, then execute official normalization once.

    Caller MUST pass raw advantages, never an already-whitened tensor. Content
    alone cannot prove that provenance; input_stage and receipt make it explicit.
    Masks must use the nonpacked, right-padding convention in the official path.
    All rows must have at least one valid action. At least one nonzero advantage
    must remain globally after suppression, otherwise ValueError is raised.
    """
    if input_stage != "raw_pre_whitening":
        raise ValueError("Only raw_pre_whitening input is accepted; no second normalization is allowed")
    raw = _cpu_tensor(raw_advantages, "raw_advantages")
    valid = _cpu_tensor(valid_mask, "valid_mask", mask=True)
    boundary = _cpu_tensor(boundary_mask, "boundary_mask", mask=True)
    if raw.shape != valid.shape or raw.shape != boundary.shape:
        raise ValueError("All three input shapes must match")
    if raw.numel() == 0:
        raise ValueError("Empty batches or zero-width tensors are unsupported")
    if not torch.isfinite(raw[valid]).all().item():
        raise ValueError("Valid raw advantages must be finite")
    if torch.any(boundary & ~valid).item():
        raise ValueError("Boundary mask cannot select padding")
    lengths = valid.sum(dim=1).tolist()
    expected_valid = torch.arange(raw.shape[1])[None, :] < torch.tensor(lengths)[:, None]
    if not torch.equal(valid, expected_valid) or any(n == 0 for n in lengths):
        raise ValueError("Each row must contain a nonempty contiguous valid prefix followed by right padding")
    # Work only on a detached copy. The caller's original tensor is never changed.
    before = raw.clone()
    masked = raw.clone()
    masked[boundary] = 0.0
    masked[~valid] = 0.0
    rows = [masked[i, :n].clone() for i, n in enumerate(lengths)]
    oracle_rows, oracle_statistics = _float64_oracle([row.tolist() for row in rows])
    initial_zero = (masked == 0) & valid
    items = [SimpleNamespace(advantages=row, action_mask=torch.ones_like(row, dtype=torch.bool))
             for row in rows]
    buffer = _OfflineBuffer(items)
    strategy = _IdentityStrategy()
    official_normalize, source_receipt = _official_normalize()
    official_normalize(buffer, "advantages", strategy)
    if len(strategy.calls) != 2:
        raise AssertionError("Expected exactly two official all_reduce calls from one normalization")
    output = torch.zeros_like(raw)
    oracle = torch.zeros_like(raw, dtype=torch.float64)
    for i, n in enumerate(lengths):
        output[i, :n] = items[i].advantages
        oracle[i, :n] = torch.tensor(oracle_rows[i], dtype=torch.float64)
    if not torch.isfinite(output).all().item():
        raise AssertionError("Official normalization produced nonfinite values")
    absolute_error = (output.double() - oracle).abs()
    tolerance = ORACLE_ATOL + ORACLE_RTOL * oracle.abs()
    if not torch.all(absolute_error <= tolerance).item():
        raise AssertionError(f"Official float32 vs independent float64 oracle disagreement: max_abs={absolute_error.max().item()}")
    if torch.any(output[boundary] != 0).item() or torch.any(output[initial_zero] != 0).item():
        raise AssertionError("Boundary / original-zero restoration failed")
    if not torch.allclose(raw, before, rtol=0, atol=0, equal_nan=True):
        raise AssertionError("Local raw input copy was mutated")
    receipt = {
        "scope": "OpenRLHF boundary suppression and normalization transplant; NOT full OpenRLHF baseline reproduction",
        "input_stage": input_stage,
        "raw_provenance_is_callers_responsibility": True,
        "reward_or_raw_baseline_recomputed": False,
        "upstream_verl_whitening_calls": 0,
        "normalization_calls": 1,
        "boundary_masking_before_normalization": True,
        "source": source_receipt,
        "input_shape": list(raw.shape),
        "input_and_output_dtype": "float32",
        "device": "cpu",
        "valid_token_count": int(valid.sum().item()),
        "boundary_token_count": int(boundary.sum().item()),
        "natural_zero_nonboundary_count": int(((before == 0) & valid & ~boundary).sum().item()),
        "initial_zero_advantage_count_after_boundary_mask": int(initial_zero.sum().item()),
        "nonzero_advantage_count_after_boundary_mask": oracle_statistics["nonzero_count"],
        "all_boundary_outputs_zero": True,
        "all_initial_zero_outputs_zero": True,
        "raw_input_not_modified": True,
        "all_zero_behavior": "raise ValueError before invoking official zero-denominator normalizer",
        "right_padding_removed_before_official_method": True,
        "packing_samples": False,
        "identity_all_reduce_calls": strategy.calls,
        "float64_oracle": {"passed": True, "max_abs_error": absolute_error.max().item(),
                           "atol": ORACLE_ATOL, "rtol": ORACLE_RTOL,
                           "statistics": oracle_statistics},
        "limits": [
            "No reward model scores changed; high-scoring empty responses may still affect peers' raw baselines.",
            "Normal EOS and all supplied step boundaries also lose their direct policy-gradient term.",
            "Nonempty unhelpful text retains nonboundary advantages and is not fixed by this transplant.",
            "Zero direct EOS gradient does not freeze its probability under shared model parameter updates.",
            "Both experimental arms need the identical adapter and frozen input/mask protocol for a fair comparison.",
        ],
    }
    return output, receipt


def run_self_tests():
    raw = torch.tensor([[0.5, -0.2, 0.9, 99.0], [0.0, 1.5, -0.5, -99.0],
                        [0.8, 88.0, 88.0, 88.0]], dtype=torch.float32)
    before = raw.clone()
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    boundary = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    result, receipt = apply_openrlhf_boundary_norm(raw, valid, boundary)
    assert torch.equal(raw, before)
    assert result[2, 0] == 0  # EOS-only response.
    assert result[0, 2] == 0 and result[1, 2] == 0  # Normal terminal EOS.
    assert result[1, 0] == 0  # A naturally zero, nonboundary advantage.
    assert torch.all(result[~valid] == 0)
    rejected = []
    cases = [
        ("all_zero", lambda: apply_openrlhf_boundary_norm([[1.0]], [[1]], [[1]])),
        ("already_whitened", lambda: apply_openrlhf_boundary_norm([[1.0]], [[1]], [[0]], input_stage="whitened")),
        ("noncontiguous_mask", lambda: apply_openrlhf_boundary_norm([[1.0, 0.0, 2.0]], [[1, 0, 1]], [[0, 0, 0]])),
        ("boundary_in_padding", lambda: apply_openrlhf_boundary_norm([[1.0, 2.0]], [[1, 0]], [[0, 1]])),
    ]
    for name, function in cases:
        try:
            function()
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f"Expected rejection for {name}")
    return {"passed": True, "synthetic_only": True, "no_gpu": True,
            "rejected_invalid_cases": rejected, "example_receipt": receipt,
            "example_output": result.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(run_self_tests(), indent=2, ensure_ascii=False))
