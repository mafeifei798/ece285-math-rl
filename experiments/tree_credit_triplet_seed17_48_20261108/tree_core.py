"""Frozen one-level TreeRL-paper credit contract; no model imports.

We use raw descendant means and (global + local) / sqrt(descendant leaves).
This is NOT the full upstream OpenRLHF/RLOO/leaf-selection implementation.
All 8 leaves are retained. A shared prefix is reweighted, not deduplicated.
"""
import math
import re
from statistics import mean, stdev


def choose_fork(ids, logprobs, token_strings, eos_id):
    """Surprisal of the sampled token; branch strictly BEFORE that token.

    Following the upstream marker masking idea, exclude answer/conclusion/boxed
    regions, EOS, and the first token (must have a nonempty reusable prefix).
    Too-short/no-eligible roots fall back to independent empty-prefix draws.
    Ground truth and PRM scores are not inputs.
    """
    assert len(ids) == len(logprobs) == len(token_strings)
    assert all(math.isfinite(float(x)) and float(x) <= 1e-4 for x in logprobs)
    end = next((i for i, x in enumerate(ids) if x == eos_id), len(ids))
    for i, piece in enumerate(token_strings[:end]):
        if re.search(r"answer|conclusion|boxed", piece.lower()):
            end = i
            break
    eligible = list(range(1, end))
    if not eligible:
        return dict(cut=0, fallback="no_eligible_nonempty_prefix", surprisal=None,
                    eligible_count=0, excluded_tail_start=end)
    cut = max(eligible, key=lambda i: (-float(logprobs[i]), -i))
    return dict(cut=cut, fallback=None, surprisal=-float(logprobs[cut]),
                eligible_count=len(eligible), excluded_tail_start=end)


def group_coefficients(outcomes, minima, prefix_lengths, arm):
    assert arm in ("chain_cb", "tree_vr", "tree_cb")
    assert len(outcomes) == len(minima) == len(prefix_lengths) == 8
    assert all(y in (0, 1) for y in outcomes)
    assert all(math.isfinite(m) and 0 <= m <= 1 for m in minima)
    assert all(prefix_lengths[k:k+4] == [prefix_lengths[k]] * 4 for k in (0, 4))
    root = mean(outcomes)
    correct_min = mean(m for y, m in zip(outcomes, minima) if y) if any(outcomes) else 0.0
    sd = stdev(outcomes)
    result = []
    for i, y in enumerate(outcomes):
        delta = .2 * y * (minima[i] - correct_min) if arm != "tree_vr" else 0.0
        if arm == "chain_cb":
            a = (y - root) / (sd + 1e-4)
            result.append(dict(prefix_length=0, prefix_credit=a, suffix_credit=a,
                               process_bonus=delta, root_value=root, parent_value=root,
                               shared_descendant_count=1, reweight_factor=1.0))
        else:
            cut = prefix_lengths[i]
            parent = mean(outcomes[(i // 4) * 4:(i // 4) * 4 + 4]) if cut else root
            prefix_a = ((parent - root) + (parent - root)) / math.sqrt(4) if cut else 0.0
            suffix_a = (y - root) + (y - parent)
            result.append(dict(prefix_length=cut, prefix_credit=prefix_a, suffix_credit=suffix_a,
                               process_bonus=delta, root_value=root, parent_value=parent,
                               shared_descendant_count=4 if cut else 1,
                               reweight_factor=.5 if cut else 1.0))
        assert abs(delta) <= .2 + 1e-12
        if not y:
            assert delta == 0 and result[-1]["suffix_credit"] <= 0
    if not any(outcomes):
        assert all(r["prefix_credit"] == r["suffix_credit"] == r["process_bonus"] == 0 for r in result)
    return result


def make_rows(records, trace, arm):
    assert len(records) == len(trace["rows"]) and len(records) % 8 == 0
    made = []
    for start in range(0, len(records), 8):
        group = records[start:start+8]
        assert len({r["problem_id"] for r in group}) == 1
        coefficients = group_coefficients([r["outcome_reward"] for r in group],
                    [r["min_score"] for r in group],
                    [r["prefix_length"] for r in trace["rows"][start:start+8]], arm)
        for record, meta, c in zip(group, trace["rows"][start:start+8], coefficients):
            assert record["completion_token_ids"] == meta["full_token_ids"]
            n = len(record["completion_token_ids"])
            cut = c["prefix_length"]
            assert 0 <= cut < n <= 2048
            outcome = [c["prefix_credit"]] * cut + [c["suffix_credit"]] * (n - cut)
            process = [0.0] * cut + [c["process_bonus"]] * (n - cut)
            total = [a + b for a, b in zip(outcome, process)]
            made.append(dict(outcome_component=outcome, process_component=process,
                             advantages=total, coefficients=c))
    return made
