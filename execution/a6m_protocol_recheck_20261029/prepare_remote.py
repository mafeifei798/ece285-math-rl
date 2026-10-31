"""Create only a new independent evaluation source/run; perform CPU admission."""
from pathlib import Path
import json,hashlib,subprocess,zlib,base64,ast
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'experiments/a6m_protocol_recheck_20261029'
names=['common_eval.py','preflight.py','generate.py','analyze.py','controller.py','derivation.json']
files={n:(SOURCE/n).read_text() for n in names}
for n,text in files.items():
 if n.endswith('.py'):ast.parse(text,filename=n)
release={'files':{n:hashlib.sha256(text.encode()).hexdigest() for n,text in files.items()},
    'scope':'evaluation_only_one_historical_checkpoint_two_benchmarks','training':False}
(SOURCE/'release.json').write_text(json.dumps(release,indent=2)+'\n')
files['release.json']=(SOURCE/'release.json').read_text()
payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
REMOTE=r'''
from pathlib import Path
import json,zlib,base64,subprocess,os,time
r=Path('/srv/research/ece285_math_rl_project')
s=r/'experiments/a6m_protocol_recheck_20261029';run=r/'runs/A6M_protocol_recheck_20261029';a=r/'audits/A6M_protocol_recheck_20261029'
assert not s.exists() and not run.exists() and not a.exists(),'Existing attempt: inspect receipts/locks; never overwrite or duplicate dispatch'
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
assert set(files)=={'common_eval.py','preflight.py','generate.py','analyze.py','controller.py','derivation.json','release.json'}
s.mkdir();run.mkdir();a.mkdir()
for name,content in files.items():
    assert Path(name).name==name
    (s/name).write_text(content)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
result=subprocess.run([str(r/'.venv/bin/python'),'-B',str(s/'preflight.py')],capture_output=True,text=True,env=env)
(a/'preflight_stdout.log').write_text(result.stdout);(a/'preflight_stderr.log').write_text(result.stderr)
assert result.returncode==0,{'returncode':result.returncode,'stderr':result.stderr[-4000:],'stdout':result.stdout[-4000:]}
print(json.dumps({'preflight':json.loads((a/'preflight.json').read_text()),'contract':json.loads((run/'contract.json').read_text()),'source':str(s),'run':str(run),'audit':str(a),'no_GPU_started':True}))
'''.replace('PAYLOAD',repr(payload))
command=['ssh','-p','11032','-i','/home/researcher/.ssh/research_server_ed25519',
 '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
 'researcher@server.example.invalid','/srv/research/ece285_math_rl_project/.venv/bin/python -B -']
r=subprocess.run(command,input=REMOTE,text=True,capture_output=True)
assert r.returncode==0,(r.stdout[-5000:],r.stderr[-5000:])
d=json.loads(r.stdout);(HERE/'admission.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':d['preflight']['passed'],'protected_files':d['preflight']['protected_files'],'models':list(d['contract']['states']),'generations':1819,'no_GPU_started':True},ensure_ascii=False))
