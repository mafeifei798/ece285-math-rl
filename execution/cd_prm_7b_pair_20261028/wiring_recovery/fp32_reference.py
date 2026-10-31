"""Read-only real-model wiring diagnosis. No optimizer and no training resume."""
from pathlib import Path
import os
import sys
import json
import hashlib
import time

ROOT = Path('/srv/research/ece285_math_rl_project')
SOURCE = ROOT / 'experiments/cd_prm_7b_pair_20261028'
OUT = ROOT / 'runs/CD_PRM7B_pair_seed17_20261028/wiring_recovery/fp32_reference'
sys.path.insert(0, str(SOURCE))
from common import (assert_root, verify_release, setup_torch, load_data, load_prm,
                    tensor_hash, read, CFG, atomic)
from adapt import component, cpu_gradient


def distance(a, b):
    import torch
    # FP64 reductions are necessary for reliable 20M-coordinate norms/cosines.
    a = a.double(); b = b.double(); e = a - b
    return dict(norm_a=float(a.norm()), norm_b=float(b.norm()),
                error_norm=float(e.norm()), relative_to_b=float(e.norm()/b.norm()) if b.norm() else None,
                max_abs_error=float(e.abs().max()),
                cosine=float(torch.dot(a,b)/(a.norm()*b.norm())) if a.norm() and b.norm() else None)


def main():
    assert_root(); verify_release(); OUT.mkdir(exist_ok=True)
    torch = setup_torch(); bm, data = load_data()
    tok, model, _ = load_prm(training=True)
    model = model.float()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    params = [p for p in model.parameters() if p.requires_grad]
    initial = tensor_hash(model)
    expected = read(OUT.parent/'C/initialization.json')['initial_adapter_hash']
    assert initial == expected
    frozen = [(p,p._version) for p in model.parameters() if not p.requires_grad]
    all_versions = [(p,p._version) for p in model.parameters()]
    started = time.time(); results = {}
    for label, arm in [('C0','C'),('D0','D')]:
        model.zero_grad(set_to_none=True); losses = {}; success_grad = None
        for key in ['fail','success','anchor_positive','anchor_negative']:
            coef = 1. if key=='fail' else CFG['arms'][arm] if key=='success' else .5*CFG['anchor_coefficient']
            selected=bm['batches'][0][key]; losses[key]=0.
            for j in selected:
                raw=component(model,data[key][j],key);losses[key]+=float(raw.detach())/len(selected)
                if arm=='C' and key=='success':
                    # Match the exact historical diagnostic instrumentation.
                    g=torch.autograd.grad(raw/len(selected),params,retain_graph=True)
                    vec=torch.cat([v.detach().float().reshape(-1).cpu() for v in g])
                    success_grad=vec if success_grad is None else success_grad+vec
                    del g,vec
                (raw*(coef/len(selected))).backward()
        grad=cpu_gradient(model);assert torch.isfinite(grad).all()
        torch.save(grad,OUT/f'{label}.pt')
        if success_grad is not None:torch.save(success_grad,OUT/f'{label}_success.pt')
        results[label]=dict(losses=losses,elapsed_seconds=time.time()-started)
        assert all(p.grad is None and p._version==v for p,v in frozen)
        assert all(p._version==v for p,v in all_versions)
        print(json.dumps(dict(stage='no_update_gradient_replay',pass_id=label,**results[label])),flush=True)
        del grad,success_grad
    assert tensor_hash(model)==initial
    model.zero_grad(set_to_none=True);del model;torch.cuda.empty_cache()
    tensors={n:torch.load(OUT/f'{n}.pt',map_location='cpu',weights_only=True) for n in ['C0','D0','C0_success']}
    old={n:torch.load(OUT.parent/p,map_location='cpu',weights_only=True) for n,p in [('C','C/first_gradient.pt'),('D','D/first_gradient.pt'),('success','C/first_success_gradient.pt')]}
    comparisons={
      'historical_identity':distance(old['C']-old['D'],old['success']),
      'replay_C_vs_saved':distance(tensors['C0'],old['C']),
      'replay_D_vs_saved':distance(tensors['D0'],old['D']),
      'replay_success_vs_saved':distance(tensors['C0_success'],old['success']),
      'replay_identity':distance(tensors['C0']-tensors['D0'],tensors['C0_success']),
    }
    strict_pass=torch.allclose(tensors['C0']-tensors['D0'],tensors['C0_success'],atol=2e-6,rtol=.005)
    result=dict(complete=True,strict_identity_check_passed=bool(strict_pass),reference_precision='FP32, math SDPA, deterministic algorithms; diagnostic only',optimizer_updates=0,initialization_verified=True,all_parameters_unchanged=True,
      model_release_sha256=verify_release(),initial_adapter_hash=initial,passes=results,comparisons=comparisons,
      peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.time()-started)
    atomic(OUT/'gradient_replay.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
