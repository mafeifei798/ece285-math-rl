"""Independent PURE validation paths and immutable configuration."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
N = 2048
GROUP = 8
BATCH = 32

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)

def read(path):
    return json.loads(Path(path).read_text())

def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

def verify_release():
    release = read(HERE / 'release.json')
    for name, expected in release['files'].items():
        assert sha(ROOT / name) == expected, name
    return release

def frozen_verifier():
    # Preserve the repaired, archived verifier verbatim. Do not use PRM gates.
    sys.path.insert(0, str(ROOT / 'experiments/reward_repair_v3_20261021'))
    from repairs import repaired_outcome
    return repaired_outcome
