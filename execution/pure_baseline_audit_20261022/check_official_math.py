"""CPU-only numerical audit of pinned PURE functions; no model or optimizer.

Selected, reviewed functions are extracted from the preserved official source.
Their outputs are compared to independent scalar Python arithmetic. This does
not test model scoring, actor-token alignment, CUDA memory, or training quality.
"""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import math
import random
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE / "official_source"
torch.set_num_threads(1)


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def source_namespace():
    manifest = json.loads((HERE / "source_manifest.json").read_text())
    for rel, item in manifest["files"].items():
        assert hashlib.sha256((SRC / rel).read_bytes()).hexdigest() == item["sha256"]
    ns = {"torch": torch, "defaultdict": defaultdict}
    load_functions(SRC / "verl/utils/torch_functional.py",
                   ["masked_mean", "masked_var", "masked_whiten"], ns)
    ns["verl_F"] = SimpleNamespace(**{n: ns[n] for n in ["masked_mean", "masked_var", "masked_whiten"]})
    load_functions(SRC / "verl/trainer/ppo/core_algos.py",
                   ["compute_return", "compute_rloo_outcome_advantage", "compute_grpo_outcome_advantage",
                    "compute_policy_loss", "kl_penalty"], ns)
    return ns


def official_transform(probabilities, ends, width):
    # Extract the reviewed softmax-to-signed-score and approximate-min block.
    tree = ast.parse((SRC / "verl/workers/fsdp_workers.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProcessRewardModelWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_micro_batch")
    start = next(i for i, n in enumerate(fn.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.value) == "rm_score.softmax(dim=-1)")
    block = ast.Module(body=fn.body[start:-1], type_ignores=[])
    logits = torch.zeros((1, width, 2), dtype=torch.float32)
    mask = torch.zeros((1, width), dtype=torch.bool)
    for p, end in zip(probabilities, ends):
        logits[0, end] = torch.tensor([math.log1p(-p), math.log(p)])
        mask[0, end] = True
    ns = {"torch": torch, "rm_score": logits, "reward_mask": mask,
          "self": SimpleNamespace(disable_approx_min_form_credit_assignment=False, temperature=0.1)}
    exec(compile(block, "official_prm_transform", "exec"), ns)
    return ns["rm_score"][0]


def reference_reward(ps, ends, width):
    signed = [2 * p - 1 for p in ps]
    logits = [-r / 0.1 for r in signed]
    weights = [math.exp(v - max(logits)) for v in logits]
    total = sum(weights)
    result = [0.0] * width
    for r, w, end in zip(signed, weights, ends):
        result[end] = r * w / total
    return result


def reference_rloo(rewards, masks, groups, normalize):
    n = len(rewards[0])
    returns = [[(sum(rewards[i][t:sum(masks[i])]) if masks[i][t] else 0.0)
                for t in range(n)] for i in range(len(rewards))]
    advantages = []
    for i in range(len(rewards)):
        others = [j for j in range(len(rewards)) if groups[j] == groups[i] and j != i]
        baseline = sum(sum(returns[j]) for j in others) / (len(others) * n)
        advantages.append([r - baseline for r in returns[i]])
    if normalize:
        valid = [a for row, mask in zip(advantages, masks) for a, m in zip(row, mask) if m]
        mean = sum(valid) / len(valid)
        var = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
        advantages = [[((a - mean) / math.sqrt(var + 1e-8) if m else 0.0)
                       for a, m in zip(row, mask)] for row, mask in zip(advantages, masks)]
    return torch.tensor(advantages), torch.tensor(returns)


def main():
    ns = source_namespace()
    rng = random.Random(285)
    checks, max_error = [], 0.0
    for case in range(32):
        width = 2048 if case == 31 else 23
        reward_rows, masks = [], []
        groups = [i // 8 for i in range(16)]
        for row in range(16):
            length = rng.randint(2, min(width, 23))
            ends = sorted(set([length - 1] + rng.sample(range(length - 1), rng.randrange(min(5, length)))))
            ps = [rng.uniform(0.0001, 0.9999) for _ in ends]
            actual = official_transform(ps, ends, width)
            expected = torch.tensor(reference_reward(ps, ends, width))
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            reward_rows.append(actual * 0.5)
            masks.append([1] * length + [0] * (width - length))
        rewards, mask = torch.stack(reward_rows), torch.tensor(masks)
        for normalize in [False, True]:
            actual_a, actual_r = ns["compute_rloo_outcome_advantage"](
                rewards, mask, groups, adv_norm=normalize, return_aggregate_method="sum")
            expected_a, expected_r = reference_rloo(rewards.tolist(), masks, groups, normalize)
            torch.testing.assert_close(actual_a, expected_a, atol=5e-6, rtol=5e-6)
            torch.testing.assert_close(actual_r, expected_r, atol=2e-6, rtol=2e-6)
            max_error = max(max_error, float((actual_a - expected_a).abs().max()))
        checks.append({"case": case, "width": width, "groups": 2, "responses": 16, "passed": True})

    # Demonstrate the public implementation's fixed-width token baseline for VR.
    rewards = torch.zeros((2, 6)); rewards[0, 1] = 1.0
    masks = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    vr_a, _ = ns["compute_rloo_outcome_advantage"](rewards, masks, [0, 0])
    assert abs(float(vr_a[1, 0]) + 1 / 3) < 1e-6

    ratios = torch.tensor([[0.5, 1.0, 1.5], [1.4, 0.7, 1.0]])
    adv = torch.tensor([[1.0, -0.7, 0.3], [-1.0, 0.8, 0.0]])
    loss_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    loss, *_ = ns["compute_policy_loss"](torch.zeros_like(ratios), ratios.log(), adv, loss_mask, .2)
    manual = sum(max(-float(a) * float(r), -float(a) * min(1.2, max(.8, float(r))))
                 for a, r, m in zip(adv.flatten(), ratios.flatten(), loss_mask.flatten()) if m) / 5
    assert abs(float(loss) - manual) < 1e-6
    zeros = torch.zeros((8, 6)); mask = torch.ones_like(zeros)
    a, _ = ns["compute_rloo_outcome_advantage"](zeros, mask, [0] * 8, adv_norm=True)
    assert torch.equal(a, zeros)
    old_lp = torch.tensor([[-2.0, -3.0, -1.0]])
    ref_lp = torch.tensor([[-2.3, -2.5, -1.0]])
    kld = ns["kl_penalty"](old_lp, ref_lp, "low_var_kl")
    manual_kl = torch.tensor([[max(-10, min(10, math.exp(d) - d - 1))
                              for d in (ref_lp - old_lp)[0].tolist()]])
    torch.testing.assert_close(kld, manual_kl, atol=1e-7, rtol=1e-6)

    report = {
        "passed": True, "device": "cpu", "torch": torch.__version__,
        "official_commit": json.loads((HERE / "source_manifest.json").read_text())["commit"],
        "random_batch_cases": checks, "rloo_normalization_modes_checked": [False, True],
        "max_advantage_absolute_error": max_error,
        "additional_checks": {"ppo_clipping": True, "all_zero_reward_whitening": True,
                              "low_variance_kl": True, "vr_padding_example": True},
        "vr_padding_example": {"N": 6, "lengths": [2, 4], "outcomes": [1, 0],
                               "official_wrong_response_advantage_before_whitening": float(vr_a[1, 0]),
                               "usual_sequence_rloo_advantage": -1.0},
        "limits": ["synthetic CPU tensors only", "no model loaded", "no optimizer or training",
                   "no actual actor/PRM token alignment tested", "no CUDA peak or throughput measured",
                   "CPU FP32 arithmetic agreement is not a BF16 model-logit equivalence claim"]}
    (HERE / "numerical_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "random_batch_cases"}, indent=2))


if __name__ == "__main__":
    main()
