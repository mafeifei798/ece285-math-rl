"""Exercise signal construction, shuffled microbatches and tamper detection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,types
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from signals import ROOT
from runtime_mixin import PrefixSignalMixin
from token_loss import TokenAdvantageLossMixin

OUT=ROOT/'audits/prefix_protection_20261107'
tok=AutoTokenizer.from_pretrained(json.loads((ROOT/'receipts/assets.json').read_text())['model_path'],local_files_only=True)
records=json.loads((OUT/'selected_groups.json').read_text())[0]['group']
width=max(len(r['completion_token_ids']) for r in records)
class Origin:
    def _generate_and_score_completions(self,inputs):return inputs
class Harness(PrefixSignalMixin,TokenAdvantageLossMixin,Origin):
    def __init__(self,variant):
        self.initialize_signals(variant,types.SimpleNamespace(records=records),tok,OUT/'plumbing_trace.jsonl')
        self.model=types.SimpleNamespace(training=True);self.beta=.04;self.top_entropy_quantile=1.
        self.importance_sampling_level='token';self.epsilon_low=.2;self.epsilon_high=.2
        self.use_vllm=True;self.vllm_importance_sampling_correction=True;self.loss_type='grpo'
        self.current_gradient_accumulation_steps=32;self.args=types.SimpleNamespace(delta=None)
        self.accelerator=types.SimpleNamespace(gather=lambda x:x);self._metrics={'train':defaultdict(list)}
    def _get_per_token_logps_and_entropies(self,model,ids,mask,n,**kwargs):
        self.lp=torch.zeros(ids.shape[0],n,requires_grad=True)
        return self.lp,torch.ones_like(self.lp)

checks=[]
for variant in ['bounded','uniform','prefix']:
    ids=torch.full((8,width),tok.pad_token_id,dtype=torch.long);mask=torch.zeros_like(ids)
    for i,r in enumerate(records):
        n=len(r['completion_token_ids']);ids[i,:n]=torch.tensor(r['completion_token_ids']);mask[i,:n]=1
    out={'completion_ids':ids,'completion_mask':mask,'advantages':torch.zeros(8),
         'prompt_ids':torch.ones(8,2,dtype=torch.long),'prompt_mask':torch.ones(8,2),
         'old_per_token_logps':torch.zeros(8,width),'ref_per_token_logps':torch.full((8,width),-.2),
         'importance_sampling_ratio':torch.ones(8,width)}
    h=Harness(variant);out=h._generate_and_score_completions(out)
    order=torch.tensor([7,0,5,3,1,2,6,4]);total=0.
    for ix in order.split(2):
        batch={k:v[ix] for k,v in out.items()}
        loss=h._compute_loss(h.model,batch);loss.backward();assert torch.isfinite(h.lp.grad).all();total+=float(loss.detach())
    changed={k:v.clone() for k,v in out.items()}
    changed['advantages']=(changed['advantages']-changed['advantages'].mean())/changed['advantages'].std()
    rejected=False
    try:h._compute_loss(h.model,changed)
    except AssertionError as exc:
        assert 'normalized' in str(exc);rejected=True
    assert rejected
    checks.append({'variant':variant,'shuffled_microbatch_loss_passed':True,'inserted_second_normalization_rejected':True,'optimizer_updates':0,'loss_sum':total})
(OUT/'plumbing_validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2)+'\n')
print(json.dumps(checks,indent=2))
