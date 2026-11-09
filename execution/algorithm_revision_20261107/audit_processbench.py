"""Descriptive CPU localization audit on already-scored, previously viewed examples.

This is NOT an untouched holdout and is NOT used to optimize a threshold.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'cd_prm_7b_pair_20261028/artifacts'
out = {'scope': 'Previously inspected ProcessBench MATH diagnostic subset; descriptive audit, not untouched threshold-selection holdout', 'arms': {}}
for arm in ['B', 'C', 'D']:
    path = SOURCE / f'scores_{arm}.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows = [r for r in rows if r['dataset'] == 'processbench']
    table = []
    for subset in ['all_flawed', 'answer_wrong_flawed']:
        selected = [r for r in rows if r['label'] >= 0 and (subset == 'all_flawed' or not r['outcome'])]
        for threshold in [.3, .5, .7, .8, .9]:
            counts = Counter()
            for row in selected:
                first = next((i for i, score in enumerate(row['step_scores']) if score < threshold), None)
                label = row['label']
                category = 'undetected' if first is None else 'exact' if first == label else 'early' if first < label else 'late'
                counts[category] += 1
                if first is not None and first > 0:
                    counts['protect_prefix'] += 1
                    counts['protect_prefix_includes_error'] += int(first > label)
            assert sum(counts[k] for k in ['undetected', 'exact', 'early', 'late']) == len(selected)
            table.append(dict(subset=subset, threshold=threshold, n=len(selected), **counts))
    out['arms'][arm] = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'n': len(rows), 'table': table}
(HERE / 'processbench_audit.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out['arms']['C']['table'][-2], indent=2))
