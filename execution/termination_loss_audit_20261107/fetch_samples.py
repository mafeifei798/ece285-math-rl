"""Read exactly the twenty preselected historical truncations over SSH. No GPU."""
from pathlib import Path
import hashlib
import json
import runpy
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SSH = runpy.run_path(str(ROOT / 'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']


def main():
    dest = HERE / 'training_samples.json'
    assert not dest.exists(), 'Preserve retrieved evidence; do not overwrite.'
    selections = json.loads((HERE / 'audit.json').read_text())['sampled_training_truncations']
    remote = r'''
from pathlib import Path
import hashlib,json
root=Path('/path/to/ece285_math_rl_project/runs/ABC_seed17_resumed_V1_20261104')
selected=SELECTIONS
result={"rows":[],"source_files":[],"remote_writes":False,"GPU_work":False}
for arm in ["A","C_bounded"]:
    path=root/arm/'rollouts.jsonl'
    wanted={(r['step'],r['tag']):r for r in selected if r['arm']==arm}
    h=hashlib.sha256();n=0
    with path.open('rb') as f:
        for line in f:
            h.update(line)
            if not line.strip():continue
            r=json.loads(line);n+=1
            key=(r['optimizer_step_before_update'],r['tag'])
            if key in wanted:
                ident=wanted.pop(key)
                assert r['status']=='truncated' and not r['terminated']
                assert r['problem_id']==ident['problem_id']
                keys=['problem_id','optimizer_step_before_update','tag','completion','completion_tokens','status','gold','prompt_text','terminated','outcome_reward','advantage']
                result['rows'].append(dict(ident,**{k:r[k] for k in keys if k in r}))
    assert not wanted,wanted
    result['source_files'].append(dict(path=str(path),sha256=h.hexdigest(),rows=n,bytes=path.stat().st_size))
assert len(result['rows'])==20
print(json.dumps(result,ensure_ascii=False))
'''.replace('SELECTIONS', repr(selections))
    r = subprocess.run(SSH + ['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],
                       input=remote, text=True, capture_output=True, timeout=100)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    value = json.loads(r.stdout)
    dest.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'sample_retrieval_receipt.json').write_text(json.dumps({
        'rows': len(value['rows']), 'remote_writes': False, 'GPU_work': False,
        'samples_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
        'selection_sha256': hashlib.sha256((HERE / 'audit.json').read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    print(json.dumps({'rows':len(value['rows']), 'bytes':dest.stat().st_size,
                      'remote_writes':False, 'GPU_work':False}))


if __name__ == '__main__':
    main()
