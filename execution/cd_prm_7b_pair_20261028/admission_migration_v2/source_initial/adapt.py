"""Matched PRM-only LoRA adaptation; C/D differ solely by success coefficient."""
import argparse
import gc
import math
from common import *


def component(model,r,key):
    import torch
    ids=torch.tensor([r['input_ids']],device='cuda:0',dtype=torch.long)
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,r['positions']].float()
    assert logits.shape[-1]==2 and torch.isfinite(logits).all()
    if key in {'fail','success'}:
        E=pooled_error(logits.softmax(-1)[:,1])
        loss=-E.log() if key=='fail' else -torch.log1p(-E)
    else:loss=torch.nn.functional.cross_entropy(logits[-1:],torch.tensor([r['label']],device=logits.device))
    assert torch.isfinite(loss)
    return loss


def cpu_gradient(model):
    import torch
    return torch.cat([p.grad.detach().float().reshape(-1).cpu() for p in model.parameters() if p.requires_grad])


def preflight():
    torch=setup_torch();batch,data=load_data();tok,model,info=load_prm(training=True)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert names and all('lora_' in n for n in names)
    head=[n for n,p in model.named_parameters() if '.score.' in n]
    assert head
    frozen=[(p,p._version) for p in model.parameters() if not p.requires_grad]
    initial=tensor_hash(model)
    for key in ['fail','success','anchor_positive','anchor_negative']:
        component(model,data[key][batch['batches'][0][key][0]],key).backward()
    params=[p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert sum(float(p.grad.abs().sum()) for p in params)>0
    assert all(p.grad is None and p._version==v for p,v in frozen)
    assert tensor_hash(model)==initial
    atomic(OUT/'model_preflight.json',dict(passed=True,prm_id='Qwen/Qwen2.5-Math-PRM-7B',revision=CFG['prm_revision'],
        native_marker_id=tok.encode('<extra_0>',add_special_tokens=False),classification_head_names=head,classification_head_frozen=True,
        trainable_parameters=sum(p.numel() for p in params),initial_adapter_hash=initial,no_optimizer_update=True,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,loading_info=info))
    event('model_preflight_passed',peak_GiB=torch.cuda.max_memory_allocated()/2**30)


def train(arm):
    import numpy as np
    torch=setup_torch();verify_release();bm,data=load_data();steps=bm['steps'];dest=OUT/arm;dest.mkdir(exist_ok=True)
    if (dest/'completion.json').exists():
        d=read(dest/'completion.json');assert d['adapter_sha256']==sha(dest/'adapter/adapter_model.safetensors');return
    tok,model,loading=load_prm(training=True)
    params=[p for p in model.parameters() if p.requires_grad]
    frozen=[(n,p,p._version) for n,p in model.named_parameters() if not p.requires_grad]
    opt=torch.optim.AdamW(params,lr=CFG['learning_rate'],betas=tuple(CFG['adam_betas']),weight_decay=CFG['weight_decay'])
    config_sha=sha(HERE/'config.json');release_sha=verify_release();initial=tensor_hash(model)
    init=dict(initial_adapter_hash=initial,prm_revision=CFG['prm_revision'],batch_manifest_sha256=sha(OUT/'batch_manifest.json'),
        steps=steps,config_sha256=config_sha,classification_head_frozen=True,trainable_parameters=sum(p.numel() for p in params),success_coefficient=CFG['arms'][arm])
    if (dest/'initialization.json').exists():assert read(dest/'initialization.json')==init
    else:atomic(dest/'initialization.json',init)
    start_step=0
    if (dest/'latest_checkpoint.json').exists():
        receipt=read(dest/'latest_checkpoint.json');p=dest/receipt['file'];assert sha(p)==receipt['sha256']
        ck=torch.load(p,map_location='cpu',weights_only=False)
        assert ck['config_sha256']==config_sha and ck['release_sha256']==release_sha
        state=model.state_dict()
        for n,v in ck['adapter'].items():state[n].copy_(v.to(device=state[n].device,dtype=state[n].dtype))
        opt.load_state_dict(ck['optimizer']);random.setstate(ck['random']);np.random.set_state(ck['numpy'])
        torch.set_rng_state(ck['torch']);torch.cuda.set_rng_state_all(ck['cuda']);start_step=ck['step']
    started=time.time();success_grad=None
    for step in range(start_step,steps):
        opt.zero_grad(set_to_none=True);values={};tokens=0
        for key in ['fail','success','anchor_positive','anchor_negative']:
            coefficient=1. if key=='fail' else CFG['arms'][arm] if key=='success' else .5*CFG['anchor_coefficient']
            selected=bm['batches'][step][key];values[key]=0.
            for j in selected:
                r=data[key][j];raw=component(model,r,key);values[key]+=float(raw.detach())/len(selected)
                if step==0 and arm=='C' and key=='success':
                    g=torch.autograd.grad(raw/len(selected),params,retain_graph=True)
                    vec=torch.cat([v.detach().float().reshape(-1).cpu() for v in g]);success_grad=vec if success_grad is None else success_grad+vec
                    del g,vec
                (raw*(coefficient/len(selected))).backward();tokens+=len(r['input_ids'])
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert all(p.grad is None and p._version==v for n,p,v in frozen)
        if step==0:
            torch.save(cpu_gradient(model),dest/'first_gradient.pt')
            if arm=='C':torch.save(success_grad,dest/'first_success_gradient.pt');success_grad=None
        norm=float(torch.nn.utils.clip_grad_norm_(params,CFG['max_grad_norm']));assert norm>0 and math.isfinite(norm)
        opt.step()
        metric=dict(arm=arm,step=step+1,losses=values,total_loss=values['fail']+CFG['arms'][arm]*values['success']+.5*(values['anchor_positive']+values['anchor_negative']),
            gradient_norm_before_clip=norm,input_tokens=tokens,elapsed_seconds=time.time()-started,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
        with (dest/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        if (step+1)%CFG['checkpoint_every']==0 or step+1==steps:
            ck=dict(adapter={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},optimizer=opt.state_dict(),step=step+1,
                random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),config_sha256=config_sha,release_sha256=release_sha)
            cp=dest/'checkpoints'/f'step_{step+1:06d}.pt';cp.parent.mkdir(exist_ok=True)
            tmp=cp.with_suffix('.partial');torch.save(ck,tmp);tmp.replace(cp)
            atomic(dest/'latest_checkpoint.json',dict(file=str(cp.relative_to(dest)),sha256=sha(cp),step=step+1));del ck
        if step==0 or (step+1)%8==0:event('adapting',**metric,steps_total=steps)
    assert all(p.grad is None and p._version==v for n,p,v in frozen)
    assert tensor_hash(model)!=initial
    model.save_pretrained(dest/'adapter',safe_serialization=True)
    atomic(dest/'completion.json',dict(**init,complete=True,arm=arm,steps_completed=steps,
        final_adapter_hash=tensor_hash(model),adapter_sha256=sha(dest/'adapter/adapter_model.safetensors'),frozen_parameters_unchanged=True,
        elapsed_seconds_this_invocation=time.time()-started,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,policy_optimizer_updates=0))
    event('adaptation_complete',arm=arm,steps=steps)


def wiring():
    import torch
    c,d=read(OUT/'C/initialization.json'),read(OUT/'D/initialization.json')
    for k in ['initial_adapter_hash','prm_revision','batch_manifest_sha256','config_sha256','steps','trainable_parameters']:assert c[k]==d[k],k
    gc=torch.load(OUT/'C/first_gradient.pt',weights_only=True,map_location='cpu')
    gd=torch.load(OUT/'D/first_gradient.pt',weights_only=True,map_location='cpu')
    gs=torch.load(OUT/'C/first_success_gradient.pt',weights_only=True,map_location='cpu')
    delta=gc-gd;err=float((delta-gs).abs().max())
    ok=torch.allclose(delta,gs,atol=2e-6,rtol=.005)
    atomic(OUT/'paired_wiring.json',dict(passed=ok,identical_initialization=True,identical_batches=True,
        first_unclipped_gradient_C_minus_D_equals_success_gradient=ok,max_abs_error=err,tolerance=dict(atol=2e-6,rtol=.005)))
    assert ok,err


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','C','D','wiring']);a=p.parse_args()
    assert_root();verify_release()
    if a.mode=='preflight':preflight()
    elif a.mode=='wiring':wiring()
    else:train(a.mode)
