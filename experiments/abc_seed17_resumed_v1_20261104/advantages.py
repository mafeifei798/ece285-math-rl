"""Correct-only bounded process preference. No post-combination normalization."""
import math
import torch

@torch.no_grad()
def compute_advantages(outcomes,process_minima,*,valid=None,coefficient=.2,group_size=8):
    if group_size!=8 or coefficient!=.2:raise ValueError('Audited fixed group8/coefficient0.2')
    y=torch.as_tensor(outcomes,dtype=torch.float32).detach().flatten()
    m=torch.as_tensor(process_minima,dtype=torch.float32,device=y.device).detach().flatten()
    if y.numel()==0 or y.shape!=m.shape or y.numel()%group_size:raise ValueError('Complete original groups required')
    if not torch.isfinite(y).all() or not torch.isfinite(m).all():raise ValueError('Nonfinite signal')
    if not ((y==0)|(y==1)).all() or not ((m>=0)&(m<=1)).all():raise ValueError('Invalid signal range')
    if valid is not None:
        v=torch.as_tensor(valid,dtype=torch.bool,device=y.device).flatten()
        if v.shape!=y.shape or (m[~v]!=0).any() or (y[~v]!=0).any():raise ValueError('Invalid outputs require zero placeholders')
    yg=y.view(-1,group_size);mg=m.view(-1,group_size)
    outcome=(yg-yg.mean(1,keepdim=True))/(yg.std(1,keepdim=True)+1e-4)
    count=yg.sum(1,keepdim=True)
    mean_correct=(mg*yg).sum(1,keepdim=True)/count.clamp_min(1)
    process=coefficient*yg*(mg-mean_correct)
    actual=outcome+process
    assert process.abs().max()<=coefficient+1e-6
    assert torch.allclose(process.sum(1),torch.zeros_like(count.flatten()),atol=2e-6,rtol=0)
    assert not (actual[yg==0]>0).any()
    assert torch.equal(actual[count.flatten()==0],torch.zeros_like(actual[count.flatten()==0]))
    return dict(advantages=actual.flatten(),outcome_component=outcome.flatten(),process_component=process.flatten(),
                mean_correct=mean_correct.flatten(),post_combination_normalizations=0)
