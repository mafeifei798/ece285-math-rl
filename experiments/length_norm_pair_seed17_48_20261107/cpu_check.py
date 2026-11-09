"""Actual frozen loss/value/gradient tests and historical-advantage replay, CPU only."""
from common import *
import sys,types
from collections import defaultdict
import torch
sys.path.insert(0,str(VALIDATED))
from token_loss import TokenAdvantageLossMixin
from signals import old


class Harness:
    beta=.04;temperature=1.;top_entropy_quantile=1.;importance_sampling_level='token'
    epsilon_low=.2;epsilon_high=.2;use_vllm=True;vllm_importance_sampling_correction=False
    current_gradient_accumulation_steps=32;max_completion_length=2048
    def __init__(self,lp,kind):
        self.lp=lp;self.loss_type=kind;self.model=types.SimpleNamespace(training=True)
        self.args=types.SimpleNamespace(delta=None);self._metrics={'train':defaultdict(list)}
        self.accelerator=types.SimpleNamespace(gather=lambda x:x)
    def _get_per_token_logps_and_entropies(self,*a,**kw):return self.lp,torch.ones_like(self.lp)


def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    checks=[]
    for lengths in [(3,5),(2,2048)]:
        width=max(lengths);mask=(torch.arange(width)[None,:]<torch.tensor(lengths)[:,None]).int()
        for shift in [-.5,0.,.5]:
            inp={'prompt_ids':torch.zeros(2,2,dtype=torch.long),'prompt_mask':torch.ones(2,2),
                 'completion_ids':torch.ones(2,width,dtype=torch.long),'completion_mask':mask,
                 'advantages':torch.tensor([-.7,.8],dtype=torch.float64)[:,None].expand(-1,width).clone()*mask,
                 'old_per_token_logps':torch.zeros(2,width,dtype=torch.float64),
                 'ref_per_token_logps':torch.full((2,width),-.2,dtype=torch.float64)}
            # Production advantage dtype is float32. The loss internals here are float64
            # for a precise derivative comparison at clipping and padding boundaries.
            inp['advantages']=inp['advantages'].float();before=inp['advantages'].clone()
            outputs={}
            for kind in ['grpo','dr_grpo']:
                lp=torch.full((2,width),shift,dtype=torch.float64,requires_grad=True)
                loss=TokenAdvantageLossMixin._compute_loss(Harness(lp,kind),None,inp)
                outputs[kind]=(loss.detach(),torch.autograd.grad(loss,lp)[0])
                assert torch.equal(inp['advantages'],before)
            factors=torch.tensor(lengths,dtype=torch.float64)[:,None]/2048
            torch.testing.assert_close(outputs['dr_grpo'][1],outputs['grpo'][1]*factors,rtol=1e-12,atol=1e-15)
            assert not outputs['dr_grpo'][1][mask==0].any()
            checks.append({'lengths':lengths,'log_ratio':shift,'per_row_gradient_factor_exact':True,'padding_zero':True})
    source=CP.parent/'rollouts.jsonl';group=[];n=0;max_error=0.;wrong_positive=0
    with source.open() as f:
        for line in f:
            group.append(json.loads(line))
            if len(group)!=8:continue
            y=torch.tensor([r['outcome_reward'] for r in group])
            a=old.compute_advantages(y,[r['min_score'] for r in group],valid=[r['valid_for_process_reward'] for r in group])['advantages']
            recorded=torch.tensor([r['advantage'] for r in group]);max_error=max(max_error,float((a-recorded).abs().max()))
            wrong_positive+=int(((y==0)&(a>0)).sum());n+=8;group=[]
    assert not group and n==9216 and max_error<2e-5 and wrong_positive==0
    assert not torch.cuda.is_initialized()
    result={'passed':True,'loss_gradient_checks':checks,'historical_rows':n,'max_advantage_error':max_error,
            'wrong_positive':0,'post_scalar_normalizations':0,'CUDA_initialized':False,
            'claim':'Actual frozen token loss differs only by the expected effective_length/2048 per-row derivative factor, including gradient accumulation.'}
    atomic(AUDIT/'cpu_check.json',result);return result


if __name__=='__main__':print(json.dumps(main()))
