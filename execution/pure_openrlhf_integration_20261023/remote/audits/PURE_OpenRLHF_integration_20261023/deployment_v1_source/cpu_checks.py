"""CPU-only contract tests for the pinned OpenRLHF signal and PolicyLoss path.

No model loading, remote access, generation, optimizer or parameter updates.
Optional --output writes one new JSON receipt and refuses to overwrite.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from signals import make_signals, micro_loss, official_functions, TENSOR_FIELDS, HERE


def fixture():
    lengths = [1,5,6,4,6,5,4,6]
    positions = [[0],[1,4],[2,5],[3],[1,5],[2,4],[3],[1,3,5]]
    old = -torch.arange(48,dtype=torch.float32).reshape(8,6) / 80 - 1
    delta = torch.linspace(-.2,.25,48).reshape(8,6)
    ref = old - delta
    logits = [[[.15*j,(-1. if i%2 else 1.) + .25*j] for j in range(len(row))]
              for i,row in enumerate(positions)]
    return dict(lengths=lengths,outcomes=[0,1,0,1,1,0,0,1],step_logits=logits,
                positions=positions,groups=['q']*8,old_logps=old,ref_logps=ref)


def independent_oracle(arm, spec, *, kl_coef, k3):
    """Independent float64 formula for the entire signal chain."""
    old = spec['old_logps'].double()
    ref = spec['ref_logps'].double()
    batch,width = old.shape
    lengths = spec['lengths']
    valid = torch.arange(width)[None,:] < torch.tensor(lengths)[:,None]
    boundary = torch.zeros_like(valid)
    task = torch.zeros_like(old)
    for i,ends in enumerate(spec['positions']):
        boundary[i,ends] = True
        if arm == 'P-PRM':
            # Convert input to FP32 first, matching the integration input contract,
            # then do the numerical oracle operations entirely in float64.
            logits = torch.tensor(spec['step_logits'][i],dtype=torch.float32).double()
            probs = logits.softmax(-1)
            signed = probs[:,1]-probs[:,0]
            task[i,ends] = (-signed/.1).softmax(-1)*signed
    adjusted = task.clone()
    for i,ends in enumerate(spec['positions']):
        peers = [j for j in range(batch) if j != i and spec['groups'][j] == spec['groups'][i]]
        baseline = float(task[peers].sum()) / sum(lengths[j] for j in peers)
        prev = -1
        for end in ends:
            adjusted[i,end] -= baseline*(end-prev-1)
            prev=end
        if arm=='P-VR':
            own = spec['outcomes'][i]-1.
            peer = math.fsum(spec['outcomes'][j]-1. for j in peers)/len(peers)
            adjusted[i,lengths[i]-1] += own-peer
    kl = (old-ref)*valid
    if k3:
        kl = torch.exp(-kl)-1+kl
    reward = adjusted-kl_coef*kl
    raw = reward.flip(1).cumsum(1).flip(1)
    returns = raw.masked_fill(boundary,0)
    flat = returns[valid]
    zeros = flat==0
    denominator = int((~zeros).sum())
    if denominator==0:
        raise ValueError('independent all-zero oracle')
    mean=flat.sum()/denominator
    variance=((flat-mean)**2).sum()/denominator
    flat_norm=(flat-mean)/torch.sqrt(variance.clamp(min=1e-8))
    flat_norm[zeros]=0
    normalized=torch.zeros_like(returns);normalized[valid]=flat_norm
    return dict(task=task,kl=kl,baseline_reward=adjusted,reward=reward,
                raw_returns=raw,returns=returns,advantages=normalized)


def raises(call, expected):
    try:
        call()
    except expected as exc:
        return str(exc)
    raise AssertionError(f'Expected {expected.__name__}')


def source_ast_check():
    manifest=json.loads((HERE/'vendor_manifest.json').read_text())
    official=HERE.parents[1]/'execution/pure_official_fix_audit_20261023'
    vendor_tree=ast.parse((HERE/'vendor.py').read_text())
    local_verification=[]
    for item in manifest['nodes']:
        vendored=next(n for n in vendor_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==item['name'])
        actual_ast=ast.dump(vendored,include_attributes=False)
        assert hashlib.sha256(actual_ast.encode()).hexdigest()==item['ast_sha256']
        path=official/item['source_file']
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['source_sha256']
            body=ast.parse(path.read_text()).body
            if item['source_class']:
                body=next(n for n in body if isinstance(n,ast.ClassDef) and n.name==item['source_class']).body
            original=next(n for n in body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==item['name'])
            assert ast.dump(original,include_attributes=False)==actual_ast
            local_verification.append(item['name'])
    return dict(ast_nodes_verified=len(manifest['nodes']),original_source_nodes_compared=local_verification,
                vendor_sha256=manifest['vendor_sha256'],pin=manifest['commit'])


def run_checks():
    torch.set_num_threads(1)
    source=source_ast_check()
    spec=fixture()
    source_old=spec['old_logps'].clone();source_ref=spec['ref_logps'].clone()
    tests={}; batches={}; max_errors={}
    for arm in ('P-VR','P-PRM'):
        for k3 in (False,True):
            batch=make_signals(arm,**spec,kl_coef=.01,use_kl_estimator_k3=k3)
            oracle=independent_oracle(arm,spec,kl_coef=.01,k3=k3)
            errors={}
            for name,value in oracle.items():
                actual=getattr(batch,name).double()
                error=(actual-value).abs()
                assert torch.all(error<=2e-5+2e-5*value.abs()),(arm,k3,name,float(error.max()))
                errors[name]=float(error.max())
            assert batch.advantages[0,0]==0
            assert batch.kl[0,0]!=0  # Empty's final direct advantage suppressed even with nonzero KL.
            assert torch.all(batch.advantages[batch.boundary]==0)
            assert batch.receipt['normalization_calls']==1
            assert not batch.receipt['compute_reward_helper_used']
            max_errors[f'{arm}/k3={k3}']=errors
            batches[(arm,k3)]=batch
    tests['nonzero_KL_and_complete_float64_signal_oracle']=True
    tests['boundary_and_empty_zero_after_nonzero_KL']=True
    assert torch.equal(source_old,spec['old_logps']) and torch.equal(source_ref,spec['ref_logps'])
    tests['caller_inputs_unchanged']=True

    class PoisonPRM:
        def __len__(self): raise AssertionError('P-VR must not inspect PRM logits')
        def __iter__(self): raise AssertionError('P-VR must not inspect PRM logits')
    no_prm_spec=dict(spec,step_logits=PoisonPRM())
    vr=make_signals('P-VR',**no_prm_spec,kl_coef=.01,
                    step_logits_dtype=PoisonPRM(),prm_transform_device=PoisonPRM())
    assert all(torch.equal(getattr(vr,name),getattr(batches[('P-VR',False)],name)) for name in TENSOR_FIELDS)
    tests['PVR_never_reads_PRM_logits']=True
    tests['PVR_ignores_PRM_dtype_and_device']=True
    flipped=dict(spec,outcomes=[1-x for x in spec['outcomes']])
    prm_flipped=make_signals('P-PRM',**flipped,kl_coef=.01)
    assert all(torch.equal(getattr(prm_flipped,name),getattr(batches[('P-PRM',False)],name)) for name in TENSOR_FIELDS)
    tests['PRM_reward_independent_of_verifier_labels']=True
    vr_flipped=make_signals('P-VR',**flipped,kl_coef=.01)
    assert not torch.equal(vr_flipped.baseline_reward,vr.baseline_reward)
    tests['PVR_responds_to_verifier_labels']=True

    # Native BF16 PRM transform is kept in BF16 until the final FP32 task copy.
    bf16=make_signals('P-PRM',**spec,kl_coef=.01,
                       step_logits_dtype='bfloat16',prm_transform_device='cpu')
    independent_bf16_task=torch.zeros_like(bf16.task)
    for i,positions in enumerate(spec['positions']):
        native=torch.tensor(spec['step_logits'][i],dtype=torch.bfloat16)
        probability=torch.softmax(native,dim=-1)
        signed=probability[:,1]-probability[:,0]
        weights=torch.softmax(-signed/.1,dim=-1)
        independent_bf16_task[i,positions]=(weights*signed).float()
    assert torch.equal(bf16.task,independent_bf16_task)
    assert bf16.receipt['prm_transform']['input_dtype']=='torch.bfloat16'
    assert bf16.receipt['prm_transform']['intermediate_and_output_dtype']=='torch.bfloat16'
    assert bf16.receipt['prm_transform']['device']=='cpu'
    assert bf16.receipt['prm_transform']['cast_after_complete_softmax_signed_softmin']
    assert bf16.receipt['normalization_calls']==1
    tests['native_BF16_softmax_signed_softmin_before_FP32_task_copy']=True

    natural=dict(lengths=[3]*16,outcomes=[0]*8+[0,1]*4,step_logits=None,
        positions=[[2]]*16,groups=['allwrong']*8+['mixed']*8,
        old_logps=torch.full((16,3),-1.),ref_logps=torch.full((16,3),-1.))
    natural_batch=make_signals('P-VR',**natural,kl_coef=0.)
    assert torch.all(natural_batch.returns[:8]==0) and torch.all(natural_batch.advantages[:8]==0)
    assert torch.any(natural_batch.advantages[8:]!=0)
    tests['natural_zero_advantages_preserved']=True
    zero_case=dict(natural,outcomes=[0]*16)
    zero_error=raises(lambda:make_signals('P-VR',**zero_case,kl_coef=0.),ValueError)
    assert 'ALL_ZERO_ADVANTAGES' in zero_error
    tests['all_zero_advantages_block_instead_of_substitute']=True
    raises(lambda:vr.normalize(),RuntimeError)
    raises(lambda:make_signals('P-VR',**spec,input_stage='already_normalized'),RuntimeError)
    tests['repeat_normalization_blocked']=True
    tampered=make_signals('P-PRM',**spec);tampered.advantages[1,0]+=1
    raises(tampered.assert_sealed,RuntimeError)
    tests['sealed_signal_mutation_detected']=True

    functions=official_functions()
    policy_loss=functions['PolicyLoss'](.2)
    ratio=torch.tensor([[1.5,.5,1.1,.9]],dtype=torch.float64)
    direct_new=ratio.log().requires_grad_()
    direct_adv=torch.tensor([[1.,-1.,1.,-1.]],dtype=torch.float64)
    direct_loss,_=policy_loss(direct_new,torch.zeros_like(direct_new),direct_adv,torch.ones_like(direct_new,dtype=torch.bool))
    direct_loss.backward()
    assert abs(float(direct_loss.detach())-(-1.2+.8-1.1+.9)/4)<1e-12
    assert torch.allclose(direct_new.grad,torch.tensor([[0.,0.,-1.1/4,.9/4]],dtype=torch.float64),atol=1e-12,rtol=0)
    tests['official_policy_clip_positive_negative_loss_and_gradient']=True

    batch=batches[('P-PRM',False)]
    new=(batch.old_logps+torch.linspace(-.45,.5,batch.old_logps.numel()).reshape(batch.old_logps.shape)).clone().requires_grad_()
    losses=[]; loss_receipts=[]
    for row in range(len(batch.lengths)):
        loss,receipt=micro_loss(batch,row,new[row])
        assert receipt['normalization_calls_in_loss']==0
        losses.append(loss);loss_receipts.append(receipt)
    total=torch.stack(losses).sum()
    full,full_info=policy_loss(new,batch.old_logps,batch.advantages,batch.mask)
    assert abs(float((total-full).detach()))<1e-6
    total.backward()
    expected_gradient=torch.zeros_like(new,dtype=torch.float64)
    for row,length in enumerate(batch.lengths):
        for t in range(length):
            ratio_value=math.exp(float(new.detach()[row,t])-float(batch.old_logps[row,t]))
            advantage=float(batch.advantages[row,t])
            clipped=(advantage>0 and ratio_value>1.2) or (advantage<0 and ratio_value<.8)
            if not clipped:
                expected_gradient[row,t]=-advantage*ratio_value/(length*len(batch.lengths))
    assert torch.allclose(new.grad.double(),expected_gradient,atol=2e-7,rtol=2e-6)
    assert torch.all(new.grad[batch.boundary]==0) and torch.all(new.grad[~batch.mask]==0)
    assert new.grad[0,0]==0
    tests['microbatch1_divided_by_B_equals_response_mean_full_loss']=True
    tests['micro_loss_gradients_match_independent_piecewise_oracle']=True
    tests['no_direct_boundary_empty_or_padding_policy_gradient']=True
    guard_new=(batch.old_logps[1]+math.log(20)).clone().requires_grad_()
    with contextlib.redirect_stdout(io.StringIO()):
        guard_loss,guard_receipt=micro_loss(batch,1,guard_new)
    guard_loss.backward()
    assert guard_receipt['ratio_guard_triggered'] and float(guard_loss.detach())==0 and torch.all(guard_new.grad==0)
    tests['official_ratio_over_10_microbatch_guard']=True

    logits=torch.tensor([[[1.,-1.,.2],[.5,.9,-.7]]],dtype=torch.float32)
    labels=torch.tensor([[2,0]])
    got=functions['log_probs_from_logits'](logits,labels)
    expected=(logits.double()-torch.logsumexp(logits.double(),dim=-1,keepdim=True)).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got.double(),expected,atol=1e-7,rtol=1e-6)
    tests['official_FP32_log_softmax_and_gather']=True
    assert all(tests.values())
    return dict(passed=True,tests=tests,test_count=len(tests),source=source,
        signal_oracle_max_abs_errors=max_errors,
        policy_loss_gradient_oracle_max_abs_error=float((new.grad.double()-expected_gradient).abs().max()),
        signal_normalization_calls_per_batch=1,normalization_calls_in_each_loss=0,
        all_zero_advantages_error=zero_error,
        gpu_used=False,model_loaded=False,new_generations=0,optimizer_created=False,parameter_updates=0,
        micro_loss_receipts=loss_receipts,
        source_files={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                      for name in ('signals.py','cpu_checks.py','vendor.py','vendor_manifest.json')},
        limitations=['Synthetic CPU contract checks do not establish real-model alignment, memory use, throughput or accuracy.',
                     'The >10 average-ratio loss guard is applied per microbatch; full-batch equality test intentionally avoids triggering it.',
                     'No end-to-end distributed-runtime reproduction or training authorization is implied.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=run_checks()
    if args.output:
        with args.output.open('x') as handle:
            json.dump(result,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n')
    print(json.dumps({key:result[key] for key in ('passed','test_count','tests','source_files','policy_loss_gradient_oracle_max_abs_error')},indent=2))
