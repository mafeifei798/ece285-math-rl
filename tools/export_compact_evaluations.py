"""Export saved evaluation labels without questions, answers, or completions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    source = parser.parse_args().source.resolve()
    directory = ROOT / 'results/compact_predictions'
    manifest_path = directory / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    entries = {e['path']: e for e in manifest['files']}
    records = {}
    ab = source / 'execution/ab_corrected_6m_seed17_20261022/final_evidence'
    studies = [
        ('AB_V0_seed17_6M', ab / 'raw/runs',
         {'A': 'A_corrected_seed17_6M_20261022/evaluation',
          'B': 'B_corrected_seed17_6M_20261022/evaluation'}),
        ('V1_seed17_6M', source / 'execution/abc_seed17_resumed_v1_20261104/completed_evidence/evaluation/288',
         {'A': 'A', 'C_mix': 'C_mix', 'C_bounded': 'C_bounded'}),
        ('V1_seed42_6M', source / 'execution/ac_seed42_v1_20261106/completed_evidence/evaluation/288',
         {'A': 'A', 'C_bounded': 'C_bounded'}),
    ]
    for study, base, arms in studies:
        for arm, sub in arms.items():
            label = f'{arm}_{study}'
            for dataset in ['math500', 'gsm8k']:
                original = base / sub / f'{dataset}.jsonl'
                with original.open() as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                compact = [{k: row.get(k, '') for k in ['problem_id', 'question_sha256',
                           'reward', 'generated_tokens', 'finish_reason', 'status']} for row in rows]
                assert all(row['reward'] in [0, 1] for row in compact)
                assert len({row['problem_id'] for row in compact}) == len(compact)
                dest = directory / f'{label}_{dataset}.csv'
                with dest.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(compact[0]))
                    writer.writeheader()
                    writer.writerows(compact)
                rel = str(dest.relative_to(ROOT))
                entries[rel] = {
                    'path': rel, 'label': label, 'dataset': dataset,
                    'questions': len(rows), 'correct': int(sum(row['reward'] for row in rows)),
                    'generated_tokens': sum(row['generated_tokens'] for row in rows),
                    'original_path': str(original.relative_to(source)),
                    'original_sha256': digest(original), 'shared_sha256': digest(dest),
                    'scope': 'Saved verifier labels; no new grading or model generation.',
                }
                records[(label, dataset)] = {row['problem_id']: row for row in compact}
    pairs = []
    for study, _, arms in studies:
        for arm in arms:
            if arm == 'A':
                continue
            for dataset in ['math500', 'gsm8k']:
                a_label, b_label = f'A_{study}', f'{arm}_{study}'
                a, b = records[a_label, dataset], records[b_label, dataset]
                assert a.keys() == b.keys()
                wins = sum(b[key]['reward'] > a[key]['reward'] for key in a)
                losses = sum(b[key]['reward'] < a[key]['reward'] for key in a)
                pairs.append({'a_label': a_label, 'b_label': b_label, 'dataset': dataset,
                              'wins': wins, 'losses': losses, 'questions': len(a),
                              'difference_pp': 100*(wins-losses)/len(a)})
    manifest['snapshot_date'] = '2026-11-09'
    manifest['files'] = sorted(entries.values(), key=lambda e: e['path'])
    manifest['pairs'] = pairs
    manifest['limits'] = 'Recounts frozen outcome labels; does not regrade CoT or measure process validity.'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'compact_files': len(entries), 'new_or_refreshed_files': len(records),
                      'paired_comparisons': len(pairs), 'model_calls': 0}))


if __name__ == '__main__':
    main()
