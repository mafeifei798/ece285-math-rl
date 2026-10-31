"""Create only the independent authorized continuation and perform CPU admission."""
from pathlib import Path
import json,hashlib,ast,zlib,base64,subprocess
from inspect_remote import SSH
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'experiments/abc_seed17_resume6m_v1_20261030'
files={p.name:p.read_text() for p in SOURCE.iterdir() if p.suffix in ['.py','.json'] and p.name!='release.json'}
for name,value in files.items():
 if name.endswith('.py'):ast.parse(value,filename=name)
release={'files':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()},'dependencies':json.loads(files['dependency_hashes.json']),
    'scope':'User-authorized seed17 only; exact existing state continuation, shared V1, unchanged reward/loss, common144/288 milestones; no new PRM/extra seed'}
(SOURCE/'release.json').write_text(json.dumps(release,indent=2)+'\n');files['release.json']=(SOURCE/'release.json').read_text()
payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
remote=r'''
from pathlib import Path
import json,zlib,base64,os,subprocess
r=Path('/srv/research/ece285_math_rl_project');s=r/'experiments/abc_seed17_resume6m_v1_20261030';p=r/'runs/ABC_seed17_resume6M_V1_20261030';a=r/'audits/ABC_seed17_resume6M_V1_20261030'
assert not s.exists() and not p.exists() and not a.exists(),'Existing attempt; inspect PID/checkpoints/receipts before action'
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)));s.mkdir();p.mkdir();a.mkdir()
for n,v in files.items():
 assert Path(n).name==n
 (s/n).write_text(v)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',VLLM_USE_V1='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
result=subprocess.run([str(r/'.venv/bin/python'),'-B',str(s/'preflight.py')],capture_output=True,text=True,env=env)
(a/'preflight_stdout.log').write_text(result.stdout);(a/'preflight_stderr.log').write_text(result.stderr)
assert result.returncode==0,{'returncode':result.returncode,'stdout':result.stdout[-3500:],'stderr':result.stderr[-5000:]}
d=json.loads((a/'preflight.json').read_text());print(json.dumps({'passed':d['passed'],'protected_files':d['protected_files'],'CPU_reference_rows':d['CPU_reference_rows'],'origins':d['originals'],'no_optimizer_started':True,'source':str(s),'run':str(p),'source_release_sha256':d['source_release_sha256']}))
'''.replace('PAYLOAD',repr(payload))
r=subprocess.run(SSH+['/srv/research/ece285_math_rl_project/.venv/bin/python -B -'],input=remote,text=True,capture_output=True)
assert r.returncode==0,(r.stdout[-5000:],r.stderr[-5000:])
d=json.loads(r.stdout);(HERE/'admission.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
