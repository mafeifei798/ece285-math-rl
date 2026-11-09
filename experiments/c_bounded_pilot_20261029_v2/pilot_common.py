"""Immutable, isolated three-arm pilot bookkeeping; imports do not start jobs."""
from pathlib import Path
import hashlib,json,os,time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PAIR=ROOT/'runs/C_bounded_triplet_seed17_1M_20261029_v2'
AUDIT=ROOT/'audits/C_bounded_triplet_seed17_1M_20261029_v2'
ORIGIN=ROOT/'experiments/c_bounded_integration_20261029_v2'
ARMS=('A','C_mix','C_bounded')

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def append(p,v):
    with Path(p).open('a') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def verify():
    r=read(HERE/'release.json')
    for n,h in r['files'].items():assert sha(HERE/n)==h,('Changed pilot source',n)
    for n,h in r['dependencies'].items():assert sha(ROOT/n)==h,('Changed dependency',n)
    return sha(HERE/'release.json')
def guard():
    assert ROOT.resolve().is_relative_to(Path('/path/to/project_parent'))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
def adapter_digest(model):
    import torch
    from peft import get_peft_model_state_dict
    h=hashlib.sha256()
    for n,v in sorted(get_peft_model_state_dict(model).items()):
        t=v.detach().cpu().contiguous();h.update(n.encode());h.update(str(tuple(t.shape)).encode());h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def tree_digest(value):
    import torch
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            x=x.detach().cpu().contiguous();h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=repr):visit(k);visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for a in x:visit(a)
        else:h.update(repr(x).encode());h.update(b'\0')
    visit(value);return h.hexdigest()
def event(stage,**v):
    atomic(PAIR/'controller.json',dict(stage=stage,pid=os.getpid(),unix=time.time(),**v))
