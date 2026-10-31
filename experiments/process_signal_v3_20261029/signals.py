"""Sequence advantages only: no verifier, PRM, optimizer, or model imports.

The common outcome term is fixed. PAPO is a controlled estimator transplant,
not a reproduction of its multi-GPU generative-grader training recipe.
"""
from dataclasses import dataclass,asdict
import math
import torch

MODES=('outcome','mixed','papo','bounded')

@dataclass(frozen=True)
class SignalConfig:
    mode:str
    coefficient:float=.2
    group_size:int=8
    outcome_epsilon:float=1e-4
    process_epsilon:float=1e-6
    def __post_init__(self):
        if self.mode not in MODES:raise ValueError('Unknown signal mode')
        if self.group_size!=8:raise ValueError('Only the audited group8 protocol is admitted')
        if not math.isfinite(self.coefficient) or self.coefficient<0:raise ValueError('Invalid coefficient')
        if not all(math.isfinite(v) and v>0 for v in [self.outcome_epsilon,self.process_epsilon]):raise ValueError('Finite positive epsilons required')

def _vector(value,*,device=None):
    return torch.as_tensor(value,dtype=torch.float32,device=device).detach().flatten()

@torch.no_grad()
def construct_signal(config,outcomes,process_scores=None,*,process_scored=None,valid_output=None,
                     group_ids=None,callback_rewards=None,device=None):
    y=_vector(outcomes,device=device)
    if not y.numel() or y.numel()%config.group_size:raise ValueError('Complete groups required')
    if not torch.isfinite(y).all() or not ((y==0)|(y==1)).all():raise ValueError('Binary verified outcomes required')
    if group_ids is not None:
        if len(group_ids)!=len(y):raise ValueError('Group IDs and answers differ in length')
        for start in range(0,len(y),8):
            if len(set(group_ids[start:start+8]))!=1:raise ValueError('Responses of a group were mixed')
    if config.mode=='outcome':
        if process_scores is not None and _vector(process_scores).abs().sum()!=0:
            raise ValueError('Outcome arm cannot consume nonzero process scores')
        m=torch.zeros_like(y)
        # PRM-not-requested and invalid-answer are separate metadata concepts.
        scored=torch.zeros_like(y,dtype=torch.bool)
    else:
        if process_scores is None:raise ValueError('Process arm requires a scored/placeholder vector')
        m=_vector(process_scores,device=y.device)
        if m.shape!=y.shape or not torch.isfinite(m).all() or not ((m>=0)&(m<=1)).all():
            raise ValueError('Process probabilities must be finite in [0,1]')
        scored=torch.ones_like(y,dtype=torch.bool) if process_scored is None else torch.as_tensor(process_scored,dtype=torch.bool,device=y.device).flatten()
        if scored.shape!=y.shape or (m[~scored]!=0).any() or (y[~scored]!=0).any():
            raise ValueError('Unscored process rows require explicit zero placeholders and zero outcome')
    if valid_output is not None:
        valid=torch.as_tensor(valid_output,dtype=torch.bool,device=y.device).flatten()
        if valid.shape!=y.shape or (y[~valid]!=0).any() or (m[~valid]!=0).any():
            raise ValueError('Invalid output must have zero outcome and zero process placeholder')
    yg,mg=y.view(-1,8),m.view(-1,8)
    out=(yg-yg.mean(1,keepdim=True))/(yg.std(1,keepdim=True)+config.outcome_epsilon)
    correct_count=yg.sum(1,keepdim=True)
    correct_mean=(mg*yg).sum(1,keepdim=True)/correct_count.clamp_min(1)
    centered=yg*(mg-correct_mean)
    # Use callback scalar values if available; Python float composition matches
    # the original y+lambda*m callback before its conversion to TRL float32.
    raw_rewards=y if config.mode=='outcome' else _vector(
        [float(a)+config.coefficient*float(b) for a,b in zip(y.cpu(),m.cpu())],device=y.device)
    if callback_rewards is not None:
        callback=_vector(callback_rewards,device=y.device)
        if callback.shape!=y.shape:raise ValueError('Callback rewards have wrong shape')
        torch.testing.assert_close(callback,raw_rewards,atol=2e-7,rtol=2e-7)
        raw_rewards=callback
    proc=torch.zeros_like(yg);std_correct=torch.zeros(yg.size(0),device=y.device)
    for j in range(yg.size(0)):
        mask=yg[j].bool()
        if mask.sum()>=2:std_correct[j]=mg[j,mask].std()
        if config.mode=='papo' and mask.sum()>=2 and std_correct[j]>config.process_epsilon:
            proc[j,mask]=config.coefficient*centered[j,mask]/(std_correct[j]+config.process_epsilon)
        elif config.mode=='bounded':proc[j]=config.coefficient*centered[j]
    if config.mode=='mixed':
        rg=raw_rewards.view(-1,8)
        actual=(rg-rg.mean(1,keepdim=True))/(rg.std(1,keepdim=True)+config.outcome_epsilon)
    else:actual=out+proc
    if not torch.isfinite(actual).all():raise ValueError('Nonfinite advantages')
    if config.mode!='mixed' and (actual[yg==0]>0).any():raise AssertionError('Wrong answer received positive advantage')
    if config.mode=='bounded' and proc.abs().max()>config.coefficient+1e-6:raise AssertionError('Process bound exceeded')
    return dict(config=asdict(config),advantages=actual.flatten(),outcome_component=out.flatten(),
        process_component=proc.flatten(),deviation_from_outcome=(actual-out).flatten(),
        raw_centered_process=centered.flatten(),correct_subset_std=std_correct,
        outcomes=y,process_scores=m,process_scored=scored,callback_rewards=raw_rewards,
        process_component_is_separable=config.mode!='mixed',post_combination_normalizations=0)
