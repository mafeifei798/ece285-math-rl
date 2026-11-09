"""Replay existing rollouts and test loss/value/gradient invariants on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json, hashlib, types
from pathlib import Path
from collections import Counter,defaultdict
import torch
from transformers import AutoTokenizer
from trl import GRPOTrainer
from signals import signals,prefix_token_count,BoundaryError,ROOT
from token_loss import TokenAdvantageLossMixin

torch.set_num_threads(2)
OUT=ROOT/'audits/prefix_protection_20261107';OUT.mkdir(parents=True,exist_ok=True)
assets=json.loads((ROOT/'receipts/assets.json').read_text())
tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
tests=[]
for text in ['a = 2.\n\nb = 3.','先算 2 + 2。\r\n\r\n错误：2 + 2 = 5。','é = 2\n\n\\boxed{5}']:
    ids=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
    cut=text.index('b =') if 'b =' in text else text.index('错误') if '错误' in text else text.index('\\boxed')
    n=prefix_token_count(tok,ids,text,cut)
    assert tok.decode(ids[:n],clean_up_tokenization_spaces=False)==text[:len(tok.decode(ids[:n],clean_up_tokenization_spaces=False))]
    assert n<len(ids)-1
    tests.append({'test':'byte_boundary_unicode_crlf','protected':n})
try:prefix_token_count(tok,[tok.eos_token_id],'changed',1)
except BoundaryError: tests.append({'test':'mismatch_rejected'})
else:raise AssertionError('Bad text accepted')

class Harness:
    beta=.04;temperature=1.;top_entropy_quantile=1.;importance_sampling_level='token'
    epsilon_low=.2;epsilon_high=.2;use_vllm=True;vllm_importance_sampling_correction=True
    loss_type='grpo';current_gradient_accumulation_steps=32
    def __init__(self,lp):
        self.lp=lp;self.model=types.SimpleNamespace(training=True)
        self.args=types.SimpleNamespace(delta=None);self._metrics={'train':defaultdict(list)}
        self.accelerator=types.SimpleNamespace(gather=lambda x:x)
    def _get_per_token_logps_and_entropies(self,*a,**kw):return self.lp,torch.ones_like(self.lp)

for shift in [-.5,0.,.5]:
    mask=torch.tensor([[1,1,1,0],[1,1,1,1]])
    scalar=torch.tensor([-.7,.8]);lp=torch.full((2,4),shift,requires_grad=True)
    inp={'prompt_ids':torch.zeros(2,2,dtype=torch.long),'prompt_mask':torch.ones(2,2),
         'completion_ids':torch.ones(2,4,dtype=torch.long),'completion_mask':mask,
         'advantages':scalar,'old_per_token_logps':torch.zeros(2,4),
         'ref_per_token_logps':torch.full((2,4),-.2),'importance_sampling_ratio':torch.full((2,4),1.05)}
    loss_old=GRPOTrainer._compute_loss(Harness(lp),None,inp);grad_old=torch.autograd.grad(loss_old,lp)[0]
    inp['advantages']=scalar[:,None].expand(-1,4).clone()*mask
    lp2=lp.detach().clone().requires_grad_(True)
    loss_new=TokenAdvantageLossMixin._compute_loss(Harness(lp2),None,inp);grad_new=torch.autograd.grad(loss_new,lp2)[0]
    torch.testing.assert_close(loss_old,loss_new,rtol=0,atol=0)
    torch.testing.assert_close(grad_old,grad_new,rtol=0,atol=0)
    tests.append({'test':'original_TRL_scalar_vs_matrix_loss_and_gradient','shift':shift,'exact':True})

path=ROOT/'runs/ABC_seed17_resumed_V1_20261104/C_bounded/rollouts.jsonl'
counts=Counter();n=0;max_error=0.;mass_error=0.;protected_tokens=0;groups=[];selection=[]
with path.open() as f:
    for line in f:
        groups.append(json.loads(line))
        if len(groups)!=8:continue
        made=signals(groups,tok)
        recorded=torch.tensor([r['advantage'] for r in groups])
        max_error=max(max_error,float((recorded-made['scalar']).abs().max()))
        mass_error=max(mass_error,float((made['prefix'].abs().sum(1)-made['uniform'].abs().sum(1)).abs().max()))
        for i,a in enumerate(made['annotations']):
            counts[a['status']]+=1;protected_tokens+=a['protected_tokens']
            if (a['status']=='prefix_protected' and len(selection)<3 and 32<len(groups[i]['completion_token_ids'])<=700
                and all(s['group'][0]['problem_id']!=groups[0]['problem_id'] for s in selection)):
                selection.append({'source_row':n+i,'group':groups,'selected_index':i})
        n+=8;groups=[]
assert not groups and n==9216 and max_error<2e-5
out={'passed':True,'rows':n,'statuses':dict(counts),'protected_tokens':protected_tokens,
     'max_recorded_scalar_advantage_error':max_error,'max_uniform_prefix_absolute_mass_error':mass_error,
     'wrong_positive_tokens':0,'normalizations_after_scalar':0,'tests':tests,
     'selected_groups':len(selection),'training_started':False}
(OUT/'cpu_validation.json').write_text(json.dumps(out,indent=2)+'\n')
(OUT/'selected_groups.json').write_text(json.dumps(selection,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2))
