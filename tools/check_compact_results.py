"""CPU-only arithmetic and checksum check; does not import training libraries."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'results/compact_predictions/MANIFEST.json').read_text())
    saved = {}
    for entry in manifest['files']:
        path = ROOT / entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['shared_sha256']
        with path.open() as f:
            rows = list(csv.DictReader(f))
        keys = [r['problem_id'] for r in rows]
        assert len(keys) == len(set(keys)) == entry['questions']
        assert sum(float(r['reward']) for r in rows) == entry['correct']
        assert all(float(r['reward']) in (0., 1.) for r in rows)
        assert sum(int(r['generated_tokens']) for r in rows) == entry['generated_tokens']
        saved[entry['label'], entry['dataset']] = {r['problem_id']: float(r['reward']) for r in rows}
        print(f"{entry['label']:20s} {entry['dataset']:8s} "
              f"{entry['correct']}/{entry['questions']} "
              f"{100 * entry['correct'] / entry['questions']:.2f}%")
    for pair in manifest.get('pairs', []):
        a = saved[pair['a_label'], pair['dataset']]
        b = saved[pair['b_label'], pair['dataset']]
        assert a.keys() == b.keys()
        wins = sum(b[k] > a[k] for k in a)
        losses = sum(b[k] < a[k] for k in a)
        assert (wins, losses, len(a)) == (pair['wins'], pair['losses'], pair['questions'])
        assert abs(100*(wins-losses)/len(a) - pair['difference_pp']) < 1e-9
    print('All compact counts, paired differences and checksums verified. Full completions have not been regraded.')

if __name__ == '__main__':
    main()
