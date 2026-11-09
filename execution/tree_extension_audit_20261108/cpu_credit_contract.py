"""CPU algebra checks for a proposed TreeRL-paper + C-bounded extension.

This is a specification example, NOT a training implementation or a port of the
official TreeRL code. Tokens below are artificial. No model, GPU, or optimizer
is imported. In particular these checks cannot establish sampling efficiency,
mathematical step validity, or agreement with the upstream runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


def fixture(outcomes):
    """Two initial chains, one fork per chain, three added continuations: 8 leaves."""
    if len(outcomes) != 8 or any(y not in (0, 1) for y in outcomes):
        raise ValueError("Eight binary leaf outcomes required")
    children = {"root": ["prefix_a", "prefix_b"]}
    tokens = {"root": [], "prefix_a": [11, 12, 13], "prefix_b": [21, 22]}
    parent = {"prefix_a": "root", "prefix_b": "root"}
    labels = {}
    for branch, prefix in enumerate(("prefix_a", "prefix_b")):
        leaves = [f"leaf_{i}" for i in range(branch * 4, branch * 4 + 4)]
        children[prefix] = leaves
        for i, leaf in enumerate(leaves, start=branch * 4):
            children[leaf] = []
            parent[leaf] = prefix
            tokens[leaf] = [100 + i, 999]  # artificial EOS sentinel
            labels[leaf] = outcomes[i]
    return children, tokens, parent, labels


def credits(children, parent, labels):
    """Paper Eqs. (2)-(4) plus Algorithm 1's inverse-sqrt leaf reweighting."""
    descendants = {}

    def walk(node):
        leaves = [node] if not children[node] else [l for c in children[node] for l in walk(c)]
        descendants[node] = leaves
        return leaves

    walk("root")
    values = {node: mean(labels[l] for l in leaves) for node, leaves in descendants.items()}
    result = {}
    for node in parent:
        ga = values[node] - values["root"]
        la = values[node] - values[parent[node]]
        n = len(descendants[node])
        result[node] = dict(value=values[node], global_advantage=ga,
                            local_advantage=la, descendant_leaves=n,
                            tree_advantage=(ga + la) / math.sqrt(n))
    return result, descendants


def bounded_leaf_bonus(labels, prm, coefficient=0.2):
    mu = mean(prm[l] for l in labels if labels[l]) if any(labels.values()) else 0.0
    return {l: coefficient * labels[l] * (prm[l] - mu) for l in labels}


def project(children, tokens, parent, labels, prm):
    """Assign the proposed PRM bonus only to leaf-exclusive segments.

    The suffix-only assignment is an additional hypothesis. A whole-trajectory
    PRM minimum need not identify which suffix deserves credit.
    """
    info, descendants = credits(children, parent, labels)
    bonus = bounded_leaf_bonus(labels, prm)
    rows = []
    for leaf in labels:
        path, node = [], leaf
        while node != "root":
            path.append(node)
            node = parent[node]
        ids, a_tree, a_prm = [], [], []
        for node in reversed(path):
            ids.extend(tokens[node])
            a_tree.extend([info[node]["tree_advantage"]] * len(tokens[node]))
            delta = bonus[leaf] if len(descendants[node]) == 1 else 0.0
            a_prm.extend([delta] * len(tokens[node]))
        rows.append(dict(leaf=leaf, outcome=labels[leaf], token_ids=ids,
                         tree=a_tree, process=a_prm,
                         total=[a + b for a, b in zip(a_tree, a_prm)]))
    width = max(len(r["token_ids"]) for r in rows)
    for row in rows:
        length = len(row["token_ids"])
        row["mask"] = [1] * length + [0] * (width - length)
        for key in ("tree", "process", "total"):
            row[key] += [0.0] * (width - length)
    return info, bonus, rows


def current_c_bounded(ys, ms):
    mu_y = mean(ys)
    sd = stdev(ys)
    mu_m = mean(m for y, m in zip(ys, ms) if y) if any(ys) else 0.0
    return [(y - mu_y) / (sd + 1e-4) + .2 * y * (m - mu_m)
            for y, m in zip(ys, ms)]


def run_checks():
    checks = {}
    ys = [1, 1, 1, 0, 1, 0, 0, 0]
    ch, tok, par, labels = fixture(ys)
    prm = {leaf: .1 + .1 * i for i, leaf in enumerate(labels)}
    info, bonus, rows = project(ch, tok, par, labels, prm)
    checks["eight_leaves_topology"] = 2 * (1 + 1 * 1 * 3) == len(labels) == 8
    checks["strong_prefix_advantage_is_point25"] = info["prefix_a"]["tree_advantage"] == .25
    checks["wrong_leaf_suffix_advantage_is_minus1point25"] = info["leaf_3"]["tree_advantage"] == -1.25
    wrong = next(r for r in rows if r["leaf"] == "leaf_3")
    checks["wrong_leaf_can_have_positive_shared_prefix_outcome_credit"] = wrong["tree"][:3] == [.25] * 3
    checks["wrong_leaf_never_gets_positive_prm_bonus"] = all(
        not any(r["process"]) for r in rows if not r["outcome"])
    checks["shared_prefix_has_no_prm_bonus"] = all(
        r["process"][:len(tok[par[r["leaf"]]])] == [0.0] * len(tok[par[r["leaf"]]]) for r in rows)
    checks["process_bonus_abs_at_most_point2"] = all(abs(a) <= .2 for a in bonus.values())
    checks["total_is_direct_sum_without_another_normalization"] = all(
        all(math.isclose(t, a + b, abs_tol=1e-12) for t, a, b in zip(r["total"], r["tree"], r["process"])) for r in rows)
    checks["padding_advantages_are_zero"] = all(
        a == 0 for r in rows for a, m in zip(r["total"], r["mask"]) if not m)
    for y, name in ((0, "all_wrong"), (1, "all_correct")):
        a, b, c, d = fixture([y] * 8)
        node_info, _, projected = project(a, b, c, d, prm)
        checks[f"{name}_outcome_tree_credit_is_zero"] = all(n["tree_advantage"] == 0 for n in node_info.values())
        if not y:
            checks["all_wrong_combined_tree_and_prm_credit_is_zero"] = all(not any(r["total"]) for r in projected)
    checks["old_c_bounded_all_wrong_credit_is_also_zero"] = not any(current_c_bounded([0] * 8, list(prm.values())))
    initial = [11, 12, 13, 103, 999]
    cut = 3  # fork BEFORE selected token 103
    prefix, suffix, cap = initial[:cut], [207, 208, 999], 6
    checks["fork_prefix_excludes_selected_token"] = prefix == [11, 12, 13] and initial[cut] not in prefix
    checks["suffix_cap_is_total_cap_minus_prefix_length"] = len(prefix + suffix) <= cap and cap - len(prefix) == 3
    scores = [.9, .8, .7, .6, .5, .4, .3, .2]
    progrs_wrong_bonus = [.5 * (m - mean(scores)) for m in scores]
    checks["progrs_all_wrong_centering_can_still_give_positive_bonus"] = max(progrs_wrong_bonus) > 0 and abs(sum(progrs_wrong_bonus)) < 1e-12
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    return dict(scope="Artificial-token CPU algebra checks only; not model/runtime acceptance.",
                passed=True, checks=checks, count=len(checks),
                mixed_fixture_node_credits=info,
                wrong_leaf_example=wrong,
                no_policy_update=True, gpu_used=False,
                limitations=["Whole-trajectory PRM minimum may come from a shared prefix; suffix-only bonus is unvalidated.",
                             "Official code also has RLOO preprocessing and selected-leaf accounting; this is a paper-formula reference, not a full reproduction.",
                             "All-wrong outcome/process policy advantages are zero; KL regularization can still update parameters.",
                             "Inverse-sqrt weighting reduces repeated-prefix weight; it does not deduplicate forward/backward compute."])


def cost_projection(root):
    paths = [root / "execution" / name / "completed_evidence/comparison.json" for name in
             ("length_norm_pair_seed17_48_20261107", "prefix_triplet_seed17_48_20261107")]
    costs, sources = [], []
    for path in paths:
        raw = path.read_bytes()
        sources.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest()))
        for name, item in json.loads(raw)["training_costs"].items():
            costs.append(dict(arm=name, seconds=item["total_seconds"], tokens=item["new_generation_tokens"]))
    hours_per_million = sum(c["seconds"] for c in costs) / 3600 / (sum(c["tokens"] for c in costs) / 1e6)
    return dict(measured_chain_hours_per_million=hours_per_million, measured_runs=costs, sources=sources,
                projections=[dict(case="Two 3M screens plus two arms x two seeds x fresh 6M confirmation", million_tokens=30, chain_equivalent_hours=30 * hours_per_million),
                             dict(case="Same two arms, resume both seed17 screens 3M->6M and fresh two seed42 6M", million_tokens=24, chain_equivalent_hours=24 * hours_per_million),
                             dict(case="Two 3M screens, then selected arm seed17 3M->6M and selected arm seed42 6M", million_tokens=15, chain_equivalent_hours=15 * hours_per_million)],
                warning="Historical chain-runtime extrapolation only; not a TreeRL throughput benchmark. Excludes evaluation, GPU queueing and engineering. Tree training/PRM input tokens must be measured separately.")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    result = run_checks()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (here / "cpu_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    cost = cost_projection(here.parents[1])
    (here / "cost_projection.json").write_text(json.dumps(cost, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], cpu_checks=result["count"],
                          measured_hours_per_million=cost["measured_chain_hours_per_million"],
                          projections=cost["projections"]), indent=2))
