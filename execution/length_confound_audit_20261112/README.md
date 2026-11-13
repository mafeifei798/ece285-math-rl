# Length and output-format audit

Analysis performed November 12, 2026; public summary added November 13. These are post-hoc CPU analyses of saved seed17/42 runs. No model calls, new training, reward changes, or relaxed regrading were performed. The midterm report and original results remain unchanged.

## What the saved outputs show

On seed17 GSM8K, A has 83 `missing_or_invalid_box` responses and C-bounded has 123. None ended through truncation or reached the 2,048-token cap. Their maximum lengths are 1,419 and 687 tokens, respectively. Separately, A has 33 truncated responses and C-bounded has 10. Missing boxes and truncation are distinct categories here.

Of those missing-box responses, 83/83 for A and 122/123 for C-bounded start with a placeholder box. This describes output format, not mathematical correctness without a box.

| C-bounded training: correct responses in mixed groups | seed17 | seed42 |
|---|---:|---:|
| Responses | 3,592 | 3,389 |
| Pooled length–minimum-score Spearman correlation | -0.435 | -0.494 |
| Pearson correlation after subtracting each group's means | -0.272 | -0.273 |
| Correlation of centered within-group ranks | -0.183 | -0.164 |

Shorter correct responses tend to have higher minimum step scores, including within the same rollout group. This does not identify a causal length effect. Step count, formatting, mathematical quality, and minimum aggregation may all contribute. Responses and repeated problems are not independent observations; no naive significance test is reported.

Illustrative length matching also has incomplete coverage. With a maximum length ratio of 1.1, only 1,675/3,592 (seed17) and 1,507/3,389 (seed42) correct responses have a different-score partner in their group. These are feasibility counts, not an implemented permutation or selected training threshold.

## Paired format decomposition

Eligibility means `finish_reason == stop` and the final box is complete, nonempty, and at most 2,048 characters. It does not require a correct or symbolically parseable answer. All correctness labels are the original labels.

| GSM8K | seed17 | seed42 |
|---|---:|---:|
| Questions with both outputs eligible | 1,143 | 1,178 |
| A correct within that subset | 846 | 849 |
| C-bounded correct within that subset | 889 | 879 |
| C-bounded minus A, correct count in that subset | +43 | +30 |
| Net correct-count difference on remaining questions | -10 | -24 |
| Full-test net correct-count difference | +33 | +6 |

This is an arithmetic decomposition of observed paired outcomes. The subset is selected using both models' outputs; it is not a causal estimate of how many answers a format repair would recover. The analogous both-eligible MATH500 net differences are +26 and +8 questions.

## Files and reproduction

- `audit.py`: length correlations, truncation checks, and matching feasibility.
- `results.json`: aggregate output and original input hashes.
- `format_pairs.py`: paired format strata and placeholder overlap.
- `format_pairs.json`: public aggregate output. Illustrative response-ending snippets are omitted from this public copy; all numerical aggregates are preserved.
- `PUBLIC_EXPORT.json`: source/export hashes and the exact export transformation.

The scripts use the standard Python library and project helpers. Full reproduction requires the original prediction JSONL and compact training signals at the paths recorded in the scripts and hashes. Full response text and raw training rollouts are not distributed in this public repository. `format_pairs.py` run against private inputs also emits example endings; these were removed only from the public JSON export.

Run from a complete local project checkout with those inputs:

```bash
python execution/length_confound_audit_20261112/audit.py
python execution/length_confound_audit_20261112/format_pairs.py
```

Original PRM assignment, random assignment, and length-sorted assignment are proposed controls, not completed experiments. Protocol calibration and checkpoint learning curves also remain proposed; this audit does not start them.
