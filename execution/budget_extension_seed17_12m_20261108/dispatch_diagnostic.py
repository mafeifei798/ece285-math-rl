from pathlib import Path
import base64,zlib,json,runpy,subprocess,random,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']

def main():
    audit=ROOT/'execution/answer_order_audit_20261108/audit.py';fn=runpy.run_path(str(audit))
    path=ROOT/'execution/abc_seed17_resumed_v1_20261104/completed_evidence/evaluation/288/C_bounded/math500.jsonl'
    data=[json.loads(x) for x in path.open() if x.strip()]
    qpath=ROOT/'execution/final_four_eval_20261029/evidence/data/prepared/math500.jsonl'
    questions={x['problem_id']:x for x in map(json.loads,qpath.open())}
    pool=[x for x in data if x['reward']==1 and fn['describe'](x)['leading_placeholder_box_with_body']]
    assert len(pool)==217
    sample=random.Random(2851108).sample(pool,200)
    for row in sample:row['question']=questions[row['problem_id']]['problem']
    files={'sample200.json':json.dumps(sample,ensure_ascii=False),'diagnostic.py':(HERE/'diagnostic.py').read_text(),'answer_order_audit.py':audit.read_text()}
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    code=r'''
from pathlib import Path
import json,zlib,base64,hashlib,subprocess,os,time,psutil
r=Path('/path/to/ece285_math_rl_project');out=r/'audits/Budget_extension_seed17_12M_20261108/placeholder_diagnostic'
assert not (out/'dispatch.json').exists(),'Existing diagnostic: inspect, never duplicate'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(raw.strip().splitlines()[0])>=24000
out.mkdir(parents=True,exist_ok=True);files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
for name,text in files.items():
 p=out/name
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
log=(out/'job.log').open('ab');p=subprocess.Popen([str(r/'.venv/bin/python'),'-B','-u',str(out/'diagnostic.py')],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
x={'pid':p.pid,'create_time':psutil.Process(p.pid).create_time(),'unix':time.time(),'scope':'frozen PRM inference only; no policy generation or training',
 'files_sha256':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in files}}
(out/'dispatch.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
'''.replace('PAYLOAD',repr(payload))
    p=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=code,text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr)
    receipt=json.loads(p.stdout);(HERE/'diagnostic_dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
