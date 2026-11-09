"""Real actor/vLLM validation on existing training responses, zero updates.

No optimizer is constructed. All model mutations are confined to fresh in-memory
instances used to diagnose the pinned legacy merge/unmerge path. Disk checkpoints
are read-only. Small generated probes are diagnostics, not benchmark evaluations.
"""
import argparse, os, json, time, hashlib, fcntl, types, gc
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from signals import ROOT,signals
from token_loss import TokenAdvantageLossMixin
from runtime_mixin import PrefixSignalMixin,fingerprint
from trl import GRPOTrainer

OUT=ROOT/'audits/prefix_protection_20261107'
CP=ROOT/'runs/ABC_seed17_resumed_V1_20261104/C_bounded/checkpoint-288'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def tensor_hash(items):
    h=hashlib.sha256()
    for n,p in items:
        h.update(n.encode());v=p.detach().cpu().contiguous();h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def dump(name,obj):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)

def event(stage,**kw):
    print(json.dumps({'stage':stage,**kw}),flush=True)
    dump('progress.json',dict(stage=stage,pid=os.getpid(),unix=time.time(),**kw))

def summarise(a,b):
    delta=torch.cat([x-y for x,y in zip(a,b)]).abs()
    return {'n_tokens':delta.numel(),'mean_abs_logprob_difference':float(delta.mean()),
            'p95_abs_logprob_difference':float(torch.quantile(delta,.95)),
            'max_abs_logprob_difference':float(delta.max()),
            'fraction_abs_logprob_difference_above_0_1':float((delta>.1).float().mean())}

def main():
    assert str(ROOT).startswith('/path/to/project_parent/')
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'no_update.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'no_update_result.json').exists(),'Completed audit must not be rerun'
    assert json.loads((OUT/'cpu_validation.json').read_text())['passed']
    started=time.time();torch.set_num_threads(2);torch.manual_seed(17)
    protected={str(CP/n):sha(CP/n) for n in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json']}
    assets=json.loads((ROOT/'receipts/assets.json').read_text());tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
    selected=json.loads((OUT/'selected_groups.json').read_text());records=[]
    for s in selected:
        records.append(s['group'][s['selected_index']])
        correct=next(r for r in s['group'] if r['outcome_reward'])
        records.append(correct)
    # Restrict teacher forcing to a deterministic prefix of each existing
    # response; no response resampling or new outcome labels are involved.
    prompt_ids=[tok.encode(r['prompt_text'],add_special_tokens=False) for r in records]
    completion_ids=[r['completion_token_ids'][:700] for r in records]
    event('load_HF_actor',rows=len(records))
    model=AutoModelForCausalLM.from_pretrained(assets['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(model,CP,is_trainable=True).eval()
    base_before=tensor_hash((n,p) for n,p in model.named_parameters() if 'lora_' not in n)
    adapter_before=tensor_hash((n,p) for n,p in model.named_parameters() if 'lora_' in n)
    base_copy={n:p.detach().cpu().clone() for n,p in model.named_parameters() if 'lora_' not in n}
    @torch.no_grad()
    def hf_logps():
        values=[]
        for p,c in zip(prompt_ids,completion_ids):
            inp=torch.tensor([p+c],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=model(input_ids=inp,attention_mask=torch.ones_like(inp),logits_to_keep=len(c)+1,use_cache=False).logits[:,:-1,:].float()
            lp=z.log_softmax(-1).gather(-1,torch.tensor(c,device='cuda')[None,:,None]).squeeze().detach().cpu()
            values.append(lp)
        return values
    native_hf=hf_logps();event('HF_native_done')

    # Real backward pass for each policy loss, without optimizer creation/step.
    class Harness(PrefixSignalMixin,TokenAdvantageLossMixin):
        _get_per_token_logps_and_entropies=GRPOTrainer._get_per_token_logps_and_entropies
        def __init__(self):
            self.model=model;self.temperature=1.;self.model_kwarg_keys=['logits_to_keep']
            self.beta=.04;self.top_entropy_quantile=1.;self.importance_sampling_level='token'
            self.epsilon_low=.2;self.epsilon_high=.2;self.loss_type='grpo';self.current_gradient_accumulation_steps=32
            self.use_vllm=True;self.vllm_importance_sampling_correction=True
            self.args=types.SimpleNamespace(delta=None);self._metrics={'train':defaultdict(list)}
            self.accelerator=types.SimpleNamespace(gather=lambda x:x)
    s=selected[0];rr=s['group'];i=s['selected_index'];made=signals(rr,tok,device='cuda')
    ids=rr[i]['completion_token_ids'];p=tok.encode(rr[i]['prompt_text'],add_special_tokens=False)
    h=Harness();model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    full=torch.tensor([p+ids],device='cuda');ones=torch.ones_like(full)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        old_lp,_=h._get_per_token_logps_and_entropies(model,full,ones,len(ids))
        with model.disable_adapter():ref_lp,_=h._get_per_token_logps_and_entropies(model,full,ones,len(ids))
    inputs={'prompt_ids':torch.tensor([p],device='cuda'),'prompt_mask':torch.ones(1,len(p),device='cuda'),
            'completion_ids':torch.tensor([ids],device='cuda'),'completion_mask':torch.ones(1,len(ids),device='cuda'),
            'old_per_token_logps':old_lp,'ref_per_token_logps':ref_lp,'importance_sampling_ratio':torch.ones_like(old_lp)}
    loss_checks=[]
    for arm in ['bounded','uniform','prefix']:
        model.zero_grad(set_to_none=True);inputs['advantages']=made[arm][i:i+1,:len(ids)].clone()
        inputs['signal_trace_id']=torch.tensor([i],device='cuda')
        h.signal_expected={i:{'fingerprint':fingerprint(inputs,0),'outcome':int(rr[i]['outcome_reward'])}}
        before=inputs['advantages'].clone()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=h._compute_loss(model,inputs)
        loss.backward()
        norm=sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters() if 'lora_' in n and p.grad is not None)**.5
        assert norm>0 and torch.isfinite(loss) and torch.equal(before,inputs['advantages'])
        assert not (inputs['advantages']>0).any()
        loss_checks.append({'arm':arm,'loss':float(loss.detach()),'gradient_L2':norm,'wrong_positive_tokens':0,'advantage_unchanged_at_loss':True,
                            'advantage_absolute_mass':float(before.abs().sum()),'protected_tokens':int((before==0).sum())})
    model.zero_grad(set_to_none=True);model.eval();model.gradient_checkpointing_disable()
    assert tensor_hash((n,p) for n,p in model.named_parameters() if 'lora_' not in n)==base_before
    assert tensor_hash((n,p) for n,p in model.named_parameters() if 'lora_' in n)==adapter_before
    event('backward_checks_done_no_updates',checks=loss_checks)

    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    event('load_vLLM_V1')
    llm=LLM(model=assets['model_path'],dtype='bfloat16',gpu_memory_utilization=.18,max_model_len=4096,
        max_num_seqs=8,max_num_batched_tokens=4096,enable_lora=True,max_lora_rank=16,max_loras=1,
        max_cpu_loras=2,enforce_eager=True,enable_prefix_caching=False,seed=285)
    assert type(llm.llm_engine).__module__.startswith('vllm.v1.')
    request=LoRARequest('saved_C_bounded_288',1,str(CP))
    def vllm_lp(req):
        xs=llm.generate([{'prompt_token_ids':p+c} for p,c in zip(prompt_ids,completion_ids)],
            SamplingParams(temperature=0.,max_tokens=1,prompt_logprobs=1),lora_request=req,use_tqdm=False)
        return [torch.tensor([o.prompt_logprobs[len(p)+j][token].logprob for j,token in enumerate(c)]) for o,p,c in zip(xs,prompt_ids,completion_ids)]
    native_v=vllm_lp(request)
    def greedy(req):
        xs=llm.generate([{'prompt_token_ids':p} for p in prompt_ids[::2]],SamplingParams(temperature=0.,max_tokens=64),lora_request=req,use_tqdm=False)
        return [{'token_ids':list(o.outputs[0].token_ids),'text':o.outputs[0].text} for o in xs]
    native_probe=greedy(request);event('vLLM_native_done')
    # Reproduce the legacy synchronization on this throwaway in-memory actor.
    model.merge_adapter();merged_hf=hf_logps()
    engine_model=llm.llm_engine.model_executor.driver_worker.model_runner.model
    def weights():
        for n,p in model.named_parameters():
            if 'lora_' not in n:yield n.removeprefix('base_model.model.').replace('.base_layer',''),p.data
    engine_model.load_weights(weights())
    merged_v=vllm_lp(None);merged_probe=greedy(None)
    model.unmerge_adapter()
    base_after=tensor_hash((n,p) for n,p in model.named_parameters() if 'lora_' not in n)
    post_unmerge=hf_logps();event('legacy_roundtrip_done',base_changed=base_before!=base_after)
    # Restore the exact base before any further forward or diagnostic use.
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in base_copy:p.copy_(base_copy[n].to(p.device))
    assert tensor_hash((n,p) for n,p in model.named_parameters() if 'lora_' not in n)==base_before
    assert tensor_hash((n,p) for n,p in model.named_parameters() if 'lora_' in n)==adapter_before
    assert all(sha(Path(p))==v for p,v in protected.items())
    result={'completed':True,'optimizer_steps':0,'training_started':False,'scope':'Six selected existing training trajectories, teacher-forced prefixes <=700 tokens, three 64-token greedy probes; not benchmark accuracy',
        'checkpoint':str(CP),'protected_files':protected,'base_hash_before':base_before,'base_hash_after_legacy_roundtrip':base_after,'base_restored_exactly':True,
        'adapter_unchanged':True,'loss_checks':loss_checks,'HF_native_vs_vLLM_native':summarise(native_hf,native_v),
        'HF_native_vs_HF_merged':summarise(native_hf,merged_hf),'HF_native_vs_vLLM_merged':summarise(native_hf,merged_v),
        'HF_native_vs_HF_post_unmerge':summarise(native_hf,post_unmerge),'vLLM_native_vs_vLLM_merged':summarise(native_v,merged_v),
        'native_probe':native_probe,'merged_probe':merged_probe,'probe_ids_equal':[a['token_ids']==b['token_ids'] for a,b in zip(native_probe,merged_probe)],
        'peak_torch_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'seconds':time.time()-started,
        'precision':'HF BF16 autocast, BF16 frozen base, saved FP32 LoRA; teacher-forced softmax/logprob computed in FP32',
        'must_not_infer':'These selected numeric probes do not measure mathematical accuracy, historical drift accumulation, or causal explanation of A/C gains.'}
    dump('no_update_result.json',result);event('complete_no_updates',seconds=result['seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        OUT.mkdir(parents=True,exist_ok=True)
        dump('failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'unix':time.time(),'optimizer_steps':0})
        raise
