"""One actual on-policy batch; both loss reductions, zero optimizer steps."""
from common import *
import fcntl,sys,traceback,math
from train_arm import build
import torch
from no_update import tensor_hash
from signals import signals

OUT=AUDIT/'no_update'


def main():
    guard();verify();source_unchanged();assert read(AUDIT/'preflight.json')['passed']
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'intent.json').exists(),'Inspect prior smoke evidence before any retry'
    atomic(OUT/'intent.json',{'pid':os.getpid(),'unix':time.time(),'optimizer_steps':0})
    started=time.time();t=build('grpo',OUT);t.is_no_update=True
    t.state.global_step=START;t.current_gradient_accumulation_steps=32
    t.model.train();t.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    t.model.enable_input_require_grads()
    t._load_rng_state(str(CP))
    before_base=tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n)
    before_adapter=adapter_digest(t.model)
    byid={r['problem_id']:r for r in rows(ROOT/'data/prepared/train.jsonl')}
    chosen=t.schedule[0];inputs=[dict(byid[q]) for q in chosen for _ in range(8)]
    atomic(OUT/'progress.json',{'stage':'generate_real_batch','pid':os.getpid(),'unix':time.time(),'optimizer_steps':0})
    when=time.monotonic();output=t._generate_and_score_completions(inputs);generation_seconds=time.monotonic()-when
    records=t.signal_ledger.records;made=signals(records,t.signal_tokenizer,device=output['advantages'].device)
    assert torch.equal(made['bounded'],output['advantages'])
    assert [r['problem_id'] for r in records[::8]]==chosen
    assert all(torch.equal(output['completion_ids'][i][output['completion_mask'][i].bool()].cpu(),torch.tensor(r['completion_token_ids'])) for i,r in enumerate(records))
    candidates=[i for i,r in enumerate(records) if made['scalar'][i].abs()>0]
    assert candidates,'No nonzero policy signal in admitted real batch'
    selected=[min(candidates,key=lambda i:records[i]['completion_tokens']),max(candidates,key=lambda i:records[i]['completion_tokens'])]
    selected=list(dict.fromkeys(selected));checks=[]
    for i in selected:
        sample={k:v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0]==len(records) else v for k,v in output.items()}
        length=int(sample['completion_mask'].sum());original=sample['advantages'].clone();results={}
        for name,kind in [('grpo','grpo'),('fixed_length','dr_grpo')]:
            t.loss_type=t.args.loss_type=kind;t.experiment_arm=name;t.used_tags=set();t.loss_rows=0
            t.model.zero_grad(set_to_none=True)
            with t.compute_loss_context_manager():loss=t._compute_loss(t.model,sample)
            loss.backward()
            grad=math.sqrt(sum(float(p.grad.float().square().sum()) for p in t.model.parameters() if p.requires_grad and p.grad is not None))
            assert torch.isfinite(loss) and math.isfinite(grad) and grad>0
            assert torch.equal(sample['advantages'],original)
            assert all(torch.isfinite(p.grad).all() for p in t.model.parameters() if p.requires_grad and p.grad is not None)
            results[name]={'loss':float(loss.detach()),'gradient_L2':grad}
        factor=length/2048
        assert math.isclose(results['fixed_length']['loss'],results['grpo']['loss']*factor,rel_tol=3e-5,abs_tol=1e-7),results
        # BF16 forward/backward can round gradients differently under rescaling.
        # The synthetic float64 CPU gate is exact; real-model tolerance is fixed here before execution.
        observed=results['fixed_length']['gradient_L2']/results['grpo']['gradient_L2']
        assert math.isclose(observed,factor,rel_tol=.05,abs_tol=1e-5),(observed,factor)
        checks.append({'row':i,'length':length,'outcome':records[i]['outcome_reward'],'expected_reduction_factor':factor,
                       'observed_gradient_norm_ratio':observed,'results':results,'advantage_unchanged':True})
    t.model.zero_grad(set_to_none=True)
    assert t.optimizer is None and t.state.global_step==START
    assert tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n)==before_base
    assert adapter_digest(t.model)==before_adapter
    t.prm.assert_frozen();source_unchanged();verify()
    result={'passed':True,'optimizer_steps':0,'same_actual_batch_for_both_reductions':True,
            'source_checkpoint':str(CP),'source_adapter_sha256':sha(CP/'adapter_model.safetensors'),
            'actor_base_unchanged':True,'actor_adapter_unchanged':True,'PRM_frozen':True,
            'rows':len(records),'generated_tokens':sum(r['completion_tokens'] for r in records),
            'generation_and_PRM_seconds':generation_seconds,'loss_checks':checks,'normalizations_after_scalar':0,
            'wrong_positive_tokens':int((made['bounded'][torch.tensor([not r['outcome_reward'] for r in records],device=made['bounded'].device)]>0).sum()),
            'gradient_accumulation_steps':32,'max_completion_length':2048,'seconds':time.time()-started,
            'max_torch_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'claim':'Finite loss/backward with unchanged model parameters. No optimizer constructed, no training update.'}
    assert result['wrong_positive_tokens']==0
    atomic(OUT/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic(OUT/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'optimizer_steps':0})
        raise
