"""Bounded paired pilot stages. User-authorized updates, immutable official loss."""
import argparse,fcntl,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from support import ROOT,HERE,read,read_jsonl,atomic,sha,append,verify_release
from pilot_engine import PilotEngine,adapter_hash,frozen_hash,parameter_hash
from common import frozen_verifier
from signals import TENSOR_FIELDS
from diagnostic import summarize_batch
from no_update import protected_snapshot,MemoryMonitor

def preserved_history():
 snapshot=protected_snapshot()
 previous=ROOT/'runs/PURE_OpenRLHF_no_update_20261023_v1'
 snapshot['previous_OpenRLHF_no_update']={str(p.relative_to(ROOT)):sha(p) for p in previous.rglob('*') if p.is_file()}
 return snapshot

def sample_digest(rows):
 return hashlib.sha256(json.dumps([(r['prompt_ids'],r['emitted_completion_ids']) for r in rows],separators=(',',':')).encode()).hexdigest()

def save_checkpoint(engine,run,step,tokens,initial_digest,initial_base):
 final=run/'checkpoints'/f'step_{step:04d}';temp=final.with_name(final.name+'.partial')
 assert not final.exists() and not temp.exists();temp.mkdir(parents=True)
 engine.model.save_pretrained(temp/'adapter',safe_serialization=True)
 engine.tokenizer.save_pretrained(temp/'adapter')
 torch.save(dict(optimizer=engine.optimizer.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
  torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),batch_step=step,tokens=tokens),temp/'state.pt')
 meta=dict(batch_step=step,optimizer_steps=step,tokens=tokens,initial_samples_sha256=initial_digest,
  initial_base_sha256=initial_base,adapter_sha256=adapter_hash(engine.model),release_sha256=sha(HERE/'release.json'),
  files={str(p.relative_to(temp)):sha(p) for p in temp.rglob('*') if p.is_file()})
 atomic(temp/'checkpoint.json',meta);temp.rename(final)
 atomic(run/'latest.json',dict(checkpoint=str(final),checkpoint_receipt_sha256=sha(final/'checkpoint.json'),batch_step=step))
 return final,meta

def restore(engine,run):
 from peft import set_peft_model_state_dict
 from peft.utils.save_and_load import load_peft_weights
 latest=read(run/'latest.json');path=Path(latest['checkpoint'])
 assert path.parent==run/'checkpoints';assert sha(path/'checkpoint.json')==latest['checkpoint_receipt_sha256']
 meta=read(path/'checkpoint.json');assert meta['release_sha256']==sha(HERE/'release.json')
 for name,h in meta['files'].items():assert sha(path/name)==h,name
 set_peft_model_state_dict(engine.model,load_peft_weights(path/'adapter',device='cpu'),adapter_name='default')
 assert adapter_hash(engine.model)==meta['adapter_sha256'];engine.verify_base()
 state=torch.load(path/'state.pt',map_location='cpu',weights_only=False)
 engine.optimizer.load_state_dict(state['optimizer'])
 assert state['batch_step']==meta['batch_step'] and state['tokens']==meta['tokens']
 assert len(engine.optimizer.state)==sum(p.requires_grad for p in engine.model.parameters())
 for value in engine.optimizer.state.values():assert int(value['step'])==meta['optimizer_steps']
 random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
 assert frozen_hash(engine.model)==meta['initial_base_sha256']
 return meta

def collapse_check(run,step):
 if step<10:return False,[]
 windows=[]
 for lo in [step-9,step-4]:
  records=[read(run/'batches'/f'{i:04d}.json') for i in range(lo,lo+5)]
  rows=[r for rec in records for r in rec['rows']]
  empty=sum(not r['emitted_completion'].strip() for r in rows)/len(rows)
  bad_short=sum((len(r['emitted_completion_ids'])-int(r['raw_ended_in_eos'])<=32) and not r['reliably_parsed'] for r in rows)/len(rows)
  windows.append(dict(empty_rate=empty,short_unparsed_rate=bad_short,n=len(rows)))
 reasons=[]
 if all(w['empty_rate']>=.25 for w in windows):reasons.append('decoded-empty >=25% in each adjacent 5-batch window')
 if all(w['short_unparsed_rate']>=.5 for w in windows):reasons.append('short AND unparsed >=50% in each adjacent 5-batch window')
 return bool(reasons),dict(reasons=reasons,windows=windows)

def main():
 cli=argparse.ArgumentParser();cli.add_argument('--arm',choices=['P_VR','P_PRM'],required=True);cli.add_argument('--target-step',type=int,required=True);args=cli.parse_args()
 plan=read(HERE/'protocol.json');verify_release();assert args.target_step in plan['stage_targets']
 run=ROOT/plan['runs'][args.arm];run.mkdir(parents=True,exist_ok=True)
 lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (run/f'stage_{args.target_step}.json').exists(),'Stage already completed; do not rerun'
 if (run/'latest.json').exists():
  assert read(run/'status.json')['status']=='stage_complete','Inspect interrupted/failed run before any recovery'
  previous=read(run/'latest.json')['batch_step'];assert previous<args.target_step
 else:previous=0;assert not (run/'status.json').exists(),'Prior attempt exists without checkpoint'
 runtime=run/f'runtime_to_{args.target_step}';assert not runtime.exists();runtime.mkdir()
 state=dict(status='initializing',arm=args.arm,target_step=args.target_step,batch_step=previous,optimizer_steps=previous,pid=os.getpid(),time=time.time())
 def status(**kw):state.update(kw,time=time.time());atomic(run/'status.json',state)
 status();monitor=None
 try:
  monitor=MemoryMonitor(runtime);monitor.stage='model_load'
  preserved=preserved_history();atomic(runtime/'originals_before.json',preserved)
  manifest=read(ROOT/'data/prepared/manifest.json')
  for n in ['train.jsonl','validation.jsonl']:
   assert sha(ROOT/'data/prepared'/n)==manifest['files'][n]['sha256']
  train=read_jsonl(ROOT/'data/prepared/train.jsonl');validation=read_jsonl(ROOT/'data/prepared/validation.jsonl')
  rng=random.Random(17);rng.shuffle(train)
  vrng=random.Random(28517);vrng.shuffle(validation);validation=validation[:64]
  sequence=[q['problem_id'] for q in train[:160]]
  spec=dict(arm=args.arm,protocol=plan,release_sha256=sha(HERE/'release.json'),question_sequence=sequence,
   validation_ids=[r['problem_id'] for r in validation],data_manifest_sha256=sha(ROOT/'data/prepared/manifest.json'))
  if (run/'run_config.json').exists():assert read(run/'run_config.json')==spec
  else:atomic(run/'run_config.json',spec)
  engine=PilotEngine(runtime,train[:4],args.arm)
  verify=frozen_verifier();units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
  tokens=0;initial_digest=None;base_hash=engine.initial_base_hash
  if args.arm=='P_PRM':engine.load_prm();prm_hash=parameter_hash(engine.prm.model)
  else:prm_hash=None
  if previous:
   restored=restore(engine,run);tokens=restored['tokens'];initial_digest=restored['initial_samples_sha256'];base_hash=restored['initial_base_sha256']
   atomic(runtime/'restore.json',dict(passed=True,batch_step=previous,checkpoint=read(run/'latest.json'),optimizer_state_restored=True))
  monitor.stage='validation'
  if previous==0:
   atomic(run/'validation_0.json',engine.validation(validation,verify,units))
  for index in range(previous,args.target_step):
   step=index+1;status(status='running',stage='generation',batch_step=index,tokens=tokens)
   bfile=run/'batches'/f'{step:04d}.json';assert not bfile.exists(),'Uncommitted batch exists; manual inspection required'
   monitor.stage='generation';torch.cuda.reset_peak_memory_stats();start=time.time()
   rows,generation=engine.generate_batch(train[index*4:index*4+4],index)
   if step==1:initial_digest=sample_digest(rows);atomic(run/'initial_sampling.json',dict(sha256=initial_digest,seeds=[r['sampling_seed'] for r in rows]))
   for row in rows:
    judged=verify(row['emitted_completion'],row['gold'],row['original_terminated'],question=row['question'],problem_id=row['problem_id'],unit_metadata=units)
    assert not judged['status'].startswith('verifier_error'),judged
    row['outcome']=judged['reward'];row['outcome_details']=judged
    row['reliably_parsed']=bool(str(judged.get('answer','') or '').strip()) and row['original_terminated'] and judged['status'] in {'exact','numeric','symbolic','unit_numeric','unit_exact','unit_symbolic'}
   monitor.stage='PRM_and_logps';status(stage='PRM_and_logps')
   scored=engine.prm.score([r['alignment'] for r in rows],verify_literal_head=step==1) if args.arm=='P_PRM' else None
   old,ref=engine.old_and_reference(rows)
   batch=engine.prepare(args.arm,rows,scored,old,ref)
   diag=summarize_batch(rows,scored,batch)
   monitor.stage='backward_update';status(stage='backward_update')
   update=engine.update(batch,rows)
   if args.arm=='P_PRM':engine.prm.verify_frozen()
   tokens+=generation['tokens'];assert tokens<=step*32*2048
   (run/'batches').mkdir(exist_ok=True)
   torch.save({k:getattr(batch,k) for k in TENSOR_FIELDS},run/'batches'/f'{step:04d}_signals.pt')
   record=dict(batch_step=step,arm=args.arm,rows=rows,prm=scored,signal_receipt=batch.receipt,diagnostic=diag,update=update,
    generation=generation,elapsed_seconds=time.time()-start,cumulative_tokens=tokens,old_ref_max_abs=float((old-ref).abs().max()),
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
   atomic(bfile,record)
   monitor.stage='checkpoint';checkpoint,meta=save_checkpoint(engine,run,step,tokens,initial_digest,base_hash)
   append(run/'diagnostics.jsonl',dict(batch_step=step,tokens=tokens,diagnostic=diag,elapsed_seconds=record['elapsed_seconds'],
    gradient_norm=update['gradient_norm_before_clip'],old_ref_max_abs=record['old_ref_max_abs']))
   status(batch_step=step,optimizer_steps=step,tokens=tokens,latest_checkpoint=str(checkpoint),stage='checkpoint_complete')
   print(f'{args.arm} batch {step}/{args.target_step}: {tokens} tokens, outcome {sum(r["outcome"] for r in rows)}/32, grad {update["gradient_norm_before_clip"]:.5g}',flush=True)
  collapsed,collapse=collapse_check(run,args.target_step)
  if args.target_step in [10,20,40]:
   monitor.stage='validation';status(stage='validation')
   atomic(run/f'validation_{args.target_step}.json',engine.validation(validation,verify,units))
  monitor.stage='final_verification';engine.verify_base();assert frozen_hash(engine.model)==base_hash
  if args.arm=='P_PRM':engine.prm.verify_frozen();assert parameter_hash(engine.prm.model)==prm_hash
  assert preserved_history()==preserved,'Archived sources/results changed'
  norms=[read(run/'batches'/f'{i:04d}.json')['signal_receipt']['normalization_calls'] for i in range(1,args.target_step+1)]
  assert set(norms)=={1}
  reasons=collapse.get('reasons',[]) if isinstance(collapse,dict) else collapse
  accept=dict(finite_gradients=True,parameters_updated=True,frozen_base_unchanged=True,PRM_frozen=True,normalization_once=True,loss_normalization_zero=True,safe_vllm_sync=True)
  receipt=dict(arm=args.arm,batch_step=args.target_step,optimizer_steps=args.target_step,tokens=tokens,actual_generated_tokens=tokens,
   checkpoint=str(checkpoint),latest_checkpoint=str(checkpoint),initial_samples_sha256=initial_digest,initial_adapter_sha256=engine.initial_adapter,
   adapter_sha256=adapter_hash(engine.model),adapter_file_sha256=sha(checkpoint/'adapter/adapter_model.safetensors'),collapsed=collapsed,reasons=reasons,collapse_reasons=reasons,collapse_windows=collapse,
   acceptance=accept,parameters_updated=True,base_frozen=True,prm_frozen=True,normalization_once=True,no_second_normalization=True,
   frozen_base_sha256=base_hash,PRM_parameter_sha256=prm_hash,release_sha256=sha(HERE/'release.json'),
   runtime_peak_allocated_bytes=torch.cuda.max_memory_allocated(),runtime_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   old_results_preserved=True,completed=time.time())
  atomic(run/f'stage_{args.target_step}.json',receipt)
  status(status='completed' if args.target_step==40 else 'stage_complete',stage='done',collapsed=collapsed)
 except Exception as exc:
  atomic(runtime/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
  status(status='failed',error=str(exc));raise
 finally:
  if monitor is not None:monitor.close()
  if torch.distributed.is_available() and torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':main()
