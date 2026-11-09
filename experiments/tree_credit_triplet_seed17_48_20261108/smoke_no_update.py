"""Actual chain/tree generations and all three losses, no optimizer constructed."""
from common import *
import fcntl,traceback,math
from train_arm import build
from integration import fingerprint
from no_update import tensor_hash
import torch

OUT=AUDIT/'no_update'


def main():
    guard();verify();source_unchanged()
    assert read(AUDIT/'preflight.json')['passed']
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'intent.json').exists(),'Inspect previous no-update evidence before recovery'
    atomic(OUT/'intent.json',{'pid':os.getpid(),'unix':time.time(),'optimizer_steps':0})
    started=time.time();t=build('tree_cb',OUT);t.is_no_update=True
    t.state.global_step=START;t.current_gradient_accumulation_steps=32
    t.model.train();t.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    t.model.enable_input_require_grads();t._load_rng_state(str(CP))
    base_before=tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n)
    adapter_before=adapter_digest(t.model)
    data={r['problem_id']:r for r in rows(ROOT/'data/prepared/train.jsonl')}
    chosen=t.schedule[0];inputs=[dict(data[q]) for q in chosen for _ in range(8)]
    checks=[];generation=[]
    for mode in ['chain_cb','tree_cb']:
        t.experiment_arm=mode;t.run=OUT/mode;t.run.mkdir()
        atomic(OUT/'progress.json',{'stage':'real_generation_'+mode,'optimizer_steps':0,'pid':os.getpid(),'unix':time.time()})
        before=time.monotonic();output=t._generate_and_score_completions(inputs)
        seconds=time.monotonic()-before
        records=t.signal_ledger.records;trace=t.tree_sampler.last_trace
        assert len(records)==32 and [r['problem_id'] for r in records[::8]]==chosen
        for i,(r,meta) in enumerate(zip(records,trace['rows'])):
            mask=output['completion_mask'][i].bool()
            assert output['completion_ids'][i][mask].cpu().tolist()==r['completion_token_ids']==meta['full_token_ids']
            assert len(meta['sampling_logprobs'])==len(meta['full_token_ids'])<=2048
            ids=r['completion_token_ids'];assert r['terminated']==(t.signal_tokenizer.eos_token_id in ids)
            if r['terminated']:assert ids[-1]==t.signal_tokenizer.eos_token_id
        generation.append({'mode':mode,'rows':32,'unique_generation_tokens':trace['unique_generation_tokens'],
                           'expanded_leaf_tokens':sum(r['completion_tokens'] for r in records),'seconds':seconds,
                           'PRM_cost':t.signal_ledger.cost,'fallback_roots':sum(bool(r['fallback']) for r in trace['roots']),
                           'raw_ID_mask_logprob_provenance_exact':True})
        variants=['chain_cb'] if mode=='chain_cb' else ['tree_vr','tree_cb']
        for variant in variants:
            t.experiment_arm=variant
            output=t.install_signal(output,records,trace,variant)
            candidates=[i for i,r in enumerate(records) if output['advantages'][i].abs().sum()>0]
            assert candidates,'Actual smoke batch must contain nonzero policy credit'
            selected=sorted(set([min(candidates,key=lambda i:records[i]['completion_tokens']),
                                 max(candidates,key=lambda i:records[i]['completion_tokens'])]))
            for i in selected:
                one={k:v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0]==32 else v for k,v in output.items()}
                before_adv=one['advantages'].clone();expected=fingerprint(one,0)
                t.used_tags=set();t.loss_rows=0;t.model.zero_grad(set_to_none=True)
                with t.compute_loss_context_manager():loss=t._compute_loss(t.model,one)
                loss.backward()
                grad=math.sqrt(sum(float(p.grad.float().square().sum()) for p in t.model.parameters() if p.requires_grad and p.grad is not None))
                assert torch.isfinite(loss) and math.isfinite(grad) and grad>0
                assert fingerprint(one,0)==expected and torch.equal(one['advantages'],before_adv)
                assert all(torch.isfinite(p.grad).all() for p in t.model.parameters() if p.requires_grad and p.grad is not None)
                checks.append({'variant':variant,'row':i,'outcome':records[i]['outcome_reward'],
                               'length':records[i]['completion_tokens'],'loss':float(loss.detach()),'gradient_L2':grad,
                               'advantages_and_components_unchanged_at_loss':True,'post_construction_normalizations':0})
            torch.save({k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in output.items()},OUT/f'{variant}_actual_signal.pt')
    t.model.zero_grad(set_to_none=True)
    assert t.optimizer is None and t.state.global_step==START
    assert tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n)==base_before
    assert adapter_digest(t.model)==adapter_before
    t.prm.assert_frozen();verify();source_unchanged()
    atomic(OUT/'result.json',{'passed':True,'optimizer_steps':0,'optimizer_constructed':False,
           'source_checkpoint':str(CP),'source_adapter_sha256':sha(CP/'adapter_model.safetensors'),
           'actor_base_unchanged':True,'actor_adapter_unchanged':True,'PRM_frozen':True,
           'generations':generation,'loss_checks':checks,'max_torch_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
           'seconds':time.time()-started,'post_construction_normalizations':0,
           'scope':'Actual chain and tree batches; tree_vr/tree_cb compared on identical tree leaves. Smoke loaded C PRM; formal tree_vr will not load/score a PRM.'})
    print(json.dumps(read(OUT/'result.json')),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic(OUT/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'optimizer_steps':0})
        raise
