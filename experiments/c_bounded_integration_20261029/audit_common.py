"""Isolated C bounded-advantage no-update integration provenance."""
from pathlib import Path
import os,json,hashlib,time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/C_bounded_no_update_20261029'
CD=ROOT/'runs/CD_PRM7B_pair_seed17_20261028'
C_ADAPTER=CD/'C/adapter'

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def event(stage,**values):
    d=dict(stage=stage,pid=os.getpid(),unix=time.time(),optimizer_steps=0,**values)
    atomic(RUN/'status.json',d);print(json.dumps(d,ensure_ascii=False),flush=True)
def verify_release():
    r=read(HERE/'release.json')
    for n,h in r['files'].items():assert sha(HERE/n)==h,n
    for n,h in r['dependencies'].items():assert sha(ROOT/n)==h,n
    return sha(HERE/'release.json')
def guard_root():
    assert ROOT.resolve().is_relative_to(Path('/srv/research')),ROOT
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
def protect():
    protected=read(CD/'protected_originals.json')
    changed=[n for n,h in protected.items() if sha(ROOT/n)!=h]
    assert not changed,changed
    extra={}
    for a in ['C','D']:
        for n in ['completion.json','initialization.json','latest_checkpoint.json','metrics.jsonl',
                  'adapter/adapter_model.safetensors','adapter/adapter_config.json']:
            p=CD/a/n;extra[str(p.relative_to(ROOT))]=sha(p)
    for n in ['summary.json','scores_B.jsonl','scores_C.jsonl','scores_D.jsonl','paired_wiring.json']:
        p=CD/n;extra[str(p.relative_to(ROOT))]=sha(p)
    return dict(original_AB_files=len(protected),original_AB_unchanged=True,CD_and_scoring_hashes=extra)
def parameter_hash(model):
    import torch
    h=hashlib.sha256()
    with torch.no_grad():
        for n,p in model.named_parameters():
            h.update(n.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode())
            h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()
def row_fingerprint(output,index):
    import torch
    h=hashlib.sha256()
    for n in ['prompt_ids','prompt_mask','completion_ids','completion_mask']:
        v=output[n][index].detach().cpu().contiguous()
        h.update(n.encode());h.update(str(tuple(v.shape)).encode());h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
