"""Install the independent release only. Does not dispatch GPU work."""
from pathlib import Path
import base64,hashlib,json,runpy,subprocess,zlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']


def main():
    source=ROOT/'experiments/tree_credit_triplet_seed17_48_20261108'
    files={p.name:p.read_text() for p in source.iterdir() if p.is_file()}
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    code=r'''
from pathlib import Path
import base64,zlib,json,hashlib,os,subprocess,psutil
root=Path('/path/to/ece285_math_rl_project')
s=root/'experiments/tree_credit_triplet_seed17_48_20261108';run=root/'runs/Tree_credit_triplet_seed17_48_20261108';audit=root/'audits/Tree_credit_triplet_seed17_48_20261108'
assert not any(p.exists() for p in [s,run,audit]),'Existing release/job requires inspection'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied; do not interrupt'
for p in psutil.process_iter(['pid','cmdline']):
    try:
        c=p.info['cmdline'] or []
        assert p.pid==os.getpid() or not (any(str(root) in str(x) for x in c) and any(any(n in str(x) for n in ['controller.py','train_arm.py','evaluate.py']) for x in c)),('Existing project process',p.pid)
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)));release=json.loads(files['release.json'])
for name,h in release['dependencies'].items():assert sha(root/name)==h,('Frozen dependency changed',name)
protected={}
for base in [root/'experiments',root/'runs/ABC_seed17_resumed_V1_20261104',root/'runs/AC_seed42_V1_20261106',root/'runs/Prefix_triplet_seed17_48_20261107',root/'runs/Length_norm_pair_seed17_48_20261107']:
    for p in base.rglob('*'):
        if not p.is_file() or p.suffix not in ['.py','.json','.jsonl','.txt','.log','.csv']:continue
        if any(x.startswith('checkpoint-') or x=='sampling_adapters' for x in p.relative_to(base).parts):continue
        protected[str(p.relative_to(root))]=sha(p)
for p in [s,run,audit]:p.mkdir(parents=True)
for name,content in files.items():assert Path(name).name==name;(s/name).write_text(content)
for name,h in release['files'].items():assert sha(s/name)==h,name
(audit/'protected_before.json').write_text(json.dumps(protected,indent=2)+'\n')
out={'prepared':True,'GPU_started':False,'source':str(s),'run':str(run),'audit':str(audit),
     'files':len(files),'protected_files':len(protected),'release_sha256':sha(s/'release.json')}
(audit/'prepared.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''.replace('PAYLOAD',repr(payload))
    r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=code,text=True,capture_output=True,timeout=150)
    if r.returncode:raise RuntimeError(r.stderr[-5000:])
    x=json.loads(r.stdout);(HERE/'prepared.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))


if __name__=='__main__':main()
