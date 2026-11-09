"""Evaluation-only provenance and immutable batch utilities; no training imports."""
from pathlib import Path
import json,hashlib,os,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PILOT=ROOT/'runs/C_bounded_triplet_seed17_1M_20261029_v2'
RUN=ROOT/'runs/Final_four_eval_seed17_1M_20261029'
AUDIT=ROOT/'audits/Final_four_eval_seed17_1M_20261029'
ARMS=('initial','A','C_mix','C_bounded')
DATASETS=('math500','gsm8k')
PROTOCOL=dict(seed=285,temperature=0.0,top_p=1.0,top_k=-1,max_tokens=2048,
    max_model_len=4096,dtype='bfloat16',gpu=1,gpu_memory_utilization=0.15,
    batch_size=64,max_num_seqs=64,max_num_batched_tokens=4096,
    enable_lora=True,max_loras=1,max_cpu_loras=3,max_lora_rank=16,enforce_eager=True,
    skip_special_tokens=True,include_stop_str_in_output=False,
    prompt='exact existing prepared dataset prompt',verifier='reward-repair-v3-20261021',
    prm_inference=False,training=False,checkpoint_selection=False,
    ordering='fixed dataset, block-interleaved models with rotating first model')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def append(p,d):
    with Path(p).open('a') as f:f.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def release_check():
    release=read(HERE/'release.json')
    for n,h in release['files'].items():assert sha(HERE/n)==h,('source_changed',n)
    return sha(HERE/'release.json')
def guard():
    assert ROOT.resolve().is_relative_to(Path('/path/to/project_parent'))
def unchanged():
    before=read(AUDIT/'protected_before.json')
    for p,h in before.items():assert sha(ROOT/p)==h,('protected_input_changed',p)
    return len(before)
def event(stage,**kw):atomic(RUN/'status.json',dict(stage=stage,pid=os.getpid(),unix=time.time(),training=False,**kw))
def batch_path(arm,dataset,offset):return RUN/arm/dataset/'batches'/f'{offset:06d}.json'
def checked_batch(p,expected,arm,dataset,contract_sha):
    d=read(p)
    assert d['arm']==arm and d['dataset']==dataset and d['contract_sha256']==contract_sha
    assert [x['problem_id'] for x in d['records']]==[x['problem_id'] for x in expected]
    for record,q in zip(d['records'],expected):
        assert record['question_sha256']==q['question_sha256'] and record['gold']==q['answer']
        assert record['generated_tokens']==len(record['completion_token_ids'])
        assert record['reward'] in [0,1]
    return d
