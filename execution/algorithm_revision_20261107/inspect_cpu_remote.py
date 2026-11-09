"""Read-only remote audit. No CUDA access, training, generation, or remote writes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, hashlib
from pathlib import Path
from collections import Counter

ROOT = Path('/path/to/ece285_math_rl_project')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

out={'scope':'read-only CPU audit, no model forwards or optimizer updates','rollouts':{}}
for arm in ['A','C_mix','C_bounded']:
    p=ROOT/'runs/ABC_seed17_resumed_V1_20261104'/arm/'rollouts.jsonl'
    rows=[json.loads(s) for s in p.open()]
    groups=Counter(); thresholds={str(t):Counter() for t in [.3,.5,.7,.8,.9]}
    total=Counter()
    for start in range(0,len(rows),8):
        g=rows[start:start+8]
        assert len(g)==8 and len({(r['problem_id'],r['optimizer_step_before_update']) for r in g})==1
        n=sum(r['outcome_reward'] for r in g)
        typ='all_wrong' if n==0 else 'all_correct' if n==8 else 'mixed'
        groups[typ]+=1
        for r in g:
            if r['outcome_reward']: continue
            total['wrong']+=1
            scores=r['step_scores']
            if not scores:
                total['wrong_unscored']+=1
                total['unscored_'+r['status']]+=1
                continue
            total['wrong_scored']+=1
            total['wrong_scored_'+typ]+=1
            for t,c in thresholds.items():
                k=next((i for i,s in enumerate(scores) if s<float(t)),None)
                cat='undetected' if k is None else 'first_step' if k==0 else 'after_first'
                c[cat]+=1;c[typ+'_'+cat]+=1
    out['rollouts'][arm]={'sha256':sha(p),'rows':len(rows),'groups':dict(groups),'wrong_counts':dict(total),'thresholds':{k:dict(v) for k,v in thresholds.items()}}

import torch
from safetensors import safe_open
torch.set_num_threads(2)
assets=json.loads((ROOT/'receipts/assets.json').read_text()); model=Path(assets['model_path'])
base_index=model/'model.safetensors.index.json'
wm=json.loads(base_index.read_text())['weight_map'] if base_index.exists() else None
out['merge_cpu_replay']={'note':'One merge/unmerge on original frozen base plus final checkpoint; not reconstruction of historical GPU weights or logits. CPU FP32 delta GEMM may differ from GPU arithmetic.','layers':[]}
for arm in ['A','C_bounded']:
    cp=ROOT/'runs/ABC_seed17_resumed_V1_20261104'/arm/'checkpoint-288'
    ap=cp/'adapter_model.safetensors'; cfg=json.loads((cp/'adapter_config.json').read_text())
    for layer in ['model.layers.0.self_attn.q_proj','model.layers.0.mlp.gate_proj']:
        with safe_open(ap,framework='pt',device='cpu') as f:
            a=f.get_tensor('base_model.model.'+layer+'.lora_A.weight')
            b=f.get_tensor('base_model.model.'+layer+'.lora_B.weight')
        key=layer+'.weight'
        bp=model/wm[key] if wm else model/'model.safetensors'
        with safe_open(bp,framework='pt',device='cpu') as f: w=f.get_tensor(key)
        delta=(b.float()@a.float())*(cfg['lora_alpha']/cfg['r'])
        merged=w.clone(); merged.add_(delta)
        actual_delta=merged.float()-w.float()
        restored=merged.clone(); restored.sub_(delta.to(w.dtype))
        drift=restored.float()-w.float()
        rms=lambda x:float(x.square().mean().sqrt())
        out['merge_cpu_replay']['layers'].append({'arm':arm,'layer':layer,'base_dtype':str(w.dtype),'adapter_dtype':str(a.dtype),'adapter_sha256':sha(ap),'numel':w.numel(),'delta_rms':rms(delta),'merged_delta_rms':rms(actual_delta),'relative_merge_error_rms':rms(actual_delta-delta)/rms(delta),'nonzero_delta_rounded_to_zero_fraction':float(((actual_delta==0)&(delta!=0)).float().mean()),'roundtrip_changed_fraction':float((restored!=w).float().mean()),'roundtrip_drift_rms':rms(drift),'roundtrip_drift_relative_to_delta':rms(drift)/rms(delta)})
out['data']={'prepared_train_keys':list(json.loads(next((ROOT/'data/prepared/train.jsonl').open()))),'raw_files':[str(p.relative_to(ROOT)) for p in (ROOT/'data/raw').glob('*math*')]}
print(json.dumps(out,indent=2))
