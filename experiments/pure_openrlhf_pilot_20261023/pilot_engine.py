"""Training wrapper for immutable OpenRLHF signals, without live merge/unmerge."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from support import ROOT,HERE,read
BASE=ROOT/'experiments/pure_openrlhf_integration_20261023'
sys.path.insert(0,str(BASE))
from engine import OpenRLHFIntegration,adapter_hash,parameter_hash,blocked
from signals import micro_loss

def frozen_hash(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if p.requires_grad:continue
  t=p.detach().cpu().contiguous();h.update(name.encode());h.update(str(t.dtype).encode());h.update(t.view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

class PilotEngine(OpenRLHFIntegration):
 def __init__(self,run,questions,arm):
  super().__init__(run,questions)
  self.arm=arm;self.plan=read(HERE/'protocol.json')
  self.backend._move_model_to_vllm=blocked
  self.frozen_versions={n:p._version for n,p in self.model.named_parameters() if not p.requires_grad}
  self.optimizer=torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
   lr=5e-7,betas=(.9,.95),eps=1e-8,weight_decay=0.)
  assert self.backend.optimizer is None
  self.initial_base_hash=frozen_hash(self.model)
  self.sync_receipts=[]

 def verify_base(self):
  for n,p in self.model.named_parameters():
   if not p.requires_grad:
    assert p.grad is None and p._version==self.frozen_versions[n],('Frozen base mutated',n)

 @torch.no_grad()
 def sync_vllm(self):
  from peft.tuners.lora.layer import LoraLayer
  self.verify_base();before=adapter_hash(self.model)
  assert self.model.active_adapters==['default']
  modules=dict(self.model.named_modules())
  llm=self.backend.llm.llm_engine.model_executor.driver_worker.model_runner.model
  sent=[];probes={};started=time.perf_counter()
  for raw,p in self.model.named_parameters():
   if '.lora_' in raw:continue
   assert not p.requires_grad and 'modules_to_save' not in raw
   name=raw.removeprefix('base_model.model.').replace('.base_layer','')
   name=self.backend._fix_param_name_to_vllm(name,extra_prefixes=['modules_to_save.default.'])
   value=p.detach().float().cpu().clone()
   if raw.endswith('.base_layer.weight'):
    m=modules[raw[:-len('.base_layer.weight')]]
    assert isinstance(m,LoraLayer) and not m.merged and not m.use_dora.get('default',False)
    assert not getattr(m,'lora_variant',{}) and not m.lora_bias.get('default',False)
    assert set(m.lora_A)=={'default'} and set(m.lora_B)=={'default'}
    a=m.lora_A['default'].weight.detach().float().cpu()
    b=m.lora_B['default'].weight.detach().float().cpu()
    delta=b@a
    if m.fan_in_fan_out:delta=delta.T
    delta=delta*m.scaling['default'];assert delta.shape==value.shape
    value.add_(delta)
   value=value.to(p.dtype);assert torch.isfinite(value).all(),name
   loaded=llm.load_weights([(name,value)]);assert loaded,('vLLM did not load weight',name)
   sent.append(dict(name=name,shape=list(value.shape),dtype=str(value.dtype)))
   if name in ['model.layers.0.self_attn.q_proj.weight','model.layers.0.mlp.gate_proj.weight']:probes[name]=value.clone()
   del value
  live=dict(llm.named_parameters())
  q=probes['model.layers.0.self_attn.q_proj.weight'];gate=probes['model.layers.0.mlp.gate_proj.weight']
  assert torch.equal(live['model.layers.0.self_attn.qkv_proj.weight'][:q.shape[0]].detach().cpu(),q)
  assert torch.equal(live['model.layers.0.mlp.gate_up_proj.weight'][:gate.shape[0]].detach().cpu(),gate)
  assert any(x['name']=='model.embed_tokens.weight' for x in sent)
  if self.model.config.tie_word_embeddings:
   assert self.model.get_input_embeddings().weight.data_ptr()==self.model.get_output_embeddings().weight.data_ptr()
   assert llm.model.embed_tokens.weight.data_ptr()==llm.lm_head.weight.data_ptr()
  else:
   assert any(x['name']=='lm_head.weight' for x in sent)
  self.backend.llm.reset_prefix_cache();self.verify_base();assert adapter_hash(self.model)==before
  receipt=dict(passed=True,live_actor_merge_calls=0,adapter_unchanged=True,frozen_versions_unchanged=True,
   sent_parameters=len(sent),sent=sent,qkv_and_gate_readback_exact=True,seconds=time.perf_counter()-started,
   method='FP32 CPU merged copies, cast to base dtype, no mutation of actor; reset prefix cache',adapter_sha256=before)
  self.sync_receipts.append(receipt);return receipt

 def generate_batch(self,questions,batch_index):
  from vllm import SamplingParams
  assert len(questions)==4
  sync=self.sync_vllm();repeated=[q for q in questions for _ in range(8)]
  inputs=[self.tokenizer.encode(q['prompt'],add_special_tokens=False)[-2048:] for q in repeated]
  seeds=[17+32*batch_index+i for i in range(32)]
  params=[SamplingParams(n=1,temperature=1.,top_p=1.,top_k=-1,max_tokens=2048,min_tokens=1,
   seed=s,skip_special_tokens=False,include_stop_str_in_output=True) for s in seeds]
  torch.cuda.synchronize();t=time.perf_counter()
  out=self.backend.llm.generate([{'prompt_token_ids':ids} for ids in inputs],sampling_params=params,use_tqdm=False)
  torch.cuda.synchronize();elapsed=time.perf_counter()-t;rows=[]
  for i,(q,prompt,result) in enumerate(zip(repeated,inputs,out)):
   answer=result.outputs[0];emitted=list(answer.token_ids)
   assert list(result.prompt_token_ids)==prompt and 0<len(emitted)<=2048
   aligned=self.builder.build(prompt,emitted);processed=aligned['actor_completion_ids']
   rows.append(dict(problem_id=q['problem_id'],question=q['problem'],gold=q['answer'],prompt=q['prompt'],prompt_ids=prompt,
    emitted_completion_ids=emitted,completion_ids=processed,emitted_completion=self.tokenizer.decode(emitted,skip_special_tokens=True,clean_up_tokenization_spaces=False),
    completion=self.tokenizer.decode(processed,skip_special_tokens=True,clean_up_tokenization_spaces=False),
    original_terminated=self.tokenizer.eos_token_id in emitted,raw_ended_in_eos=emitted[-1]==self.tokenizer.eos_token_id,
    eos_token_id=self.tokenizer.eos_token_id,finish_reason=answer.finish_reason,stop_reason=answer.stop_reason,
    alignment=aligned,batch_step=batch_index+1,response_index=i,sampling_seed=seeds[i]))
  return rows,dict(seconds=elapsed,tokens=sum(len(r['emitted_completion_ids']) for r in rows),sync=sync)

 def update(self,batch,rows):
  self.model.zero_grad(set_to_none=True);traces=[];before=adapter_hash(self.model)
  for i,row in enumerate(rows):
   loss,receipt=micro_loss(batch,i,self.logps(row));assert torch.isfinite(loss)
   loss.backward();traces.append(receipt)
  self.verify_base();trainable=[p for p in self.model.parameters() if p.requires_grad]
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
  norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
  assert norm.item()>0,'Zero actual policy gradient; do not invent an update'
  batch.assert_sealed();self.optimizer.step();self.optimizer.zero_grad(set_to_none=True);self.verify_base()
  after=adapter_hash(self.model);assert after!=before,'No parameter change after optimizer step'
  assert self.backend.optimizer is None
  return dict(gradient_norm_before_clip=norm.item(),optimizer_steps_this_batch=1,
   adapter_before=before,adapter_after=after,loss_trace=traces,
   normalization_calls=batch.receipt['normalization']['calls'],loss_normalization_calls=0)

 @torch.no_grad()
 def validation(self,questions,verify,units):
  from vllm import SamplingParams
  self.sync_vllm();out=self.backend.llm.generate([q['prompt'] for q in questions],
   sampling_params=SamplingParams(temperature=0.,max_tokens=2048,min_tokens=0,top_p=1.,top_k=-1),use_tqdm=False)
  rows=[]
  for q,result in zip(questions,out):
   a=result.outputs[0];judged=verify(a.text,q['answer'],a.finish_reason=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
   rows.append(dict(problem_id=q['problem_id'],completion=a.text,generated_tokens=len(a.token_ids),finish_reason=a.finish_reason,
    empty=not a.text.strip(),short=len(a.token_ids)<=32,**judged))
  return dict(count=len(rows),correct=sum(x['reward'] for x in rows),accuracy=sum(x['reward'] for x in rows)/len(rows),
   empty=sum(x['empty'] for x in rows),short=sum(x['short'] for x in rows),rows=rows,
   limitation='64 policy-held-out questions; known PRM800K overlap; diagnostics only, no checkpoint selection')
