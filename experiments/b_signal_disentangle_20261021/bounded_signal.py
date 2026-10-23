"""Offline/prospective advantage builder; not installed in any trainer."""
import math
import torch


@torch.no_grad()
def compute_advantages(outcomes, process_minima, *, valid=None, coefficient=0.2, group_size=8):
    """Keep outcome-GRPO advantages, add an unnormalized process residual.

    m is the existing frozen PRM's minimum step probability, zero for invalid
    outputs. No new scorer, reranking, label inference, mask, or reward head.
    Returned advantages must NOT be normalized again by a trainer.
    This does not correct a wrong PRM ordering and has no proven policy gain.
    """
    y=torch.as_tensor(outcomes,dtype=torch.float32).detach().flatten()
    m=torch.as_tensor(process_minima,dtype=torch.float32,device=y.device).detach().flatten()
    if y.shape!=m.shape or y.numel()==0 or y.numel()%group_size:
        raise ValueError('Preserve complete original question groups')
    if not math.isfinite(coefficient) or coefficient<0:
        raise ValueError('Nonnegative finite coefficient required')
    if not torch.isfinite(y).all() or not torch.isfinite(m).all():raise ValueError('Nonfinite signal')
    if not ((y==0)|(y==1)).all() or not ((m>=0)&(m<=1)).all():raise ValueError('Invalid binary outcome or PRM probability')
    if valid is not None:
        v=torch.as_tensor(valid,dtype=torch.bool,device=y.device).flatten()
        if v.shape!=y.shape or (m[~v]!=0).any() or (y[~v]!=0).any():
            raise ValueError('Unscored invalid rows must carry zero placeholders, not invented PRM scores')
    yg=y.view(-1,group_size);mg=m.view(-1,group_size)
    outcome=(yg-yg.mean(1,keepdim=True))/(yg.std(1,keepdim=True)+1e-4)
    process=coefficient*(mg-mg.mean(1,keepdim=True))
    return {'advantages':(outcome+process).flatten(),'outcome_component':outcome.flatten(),
            'process_component':process.flatten(),
            'absolute_process_bound':coefficient*(group_size-1)/group_size}
