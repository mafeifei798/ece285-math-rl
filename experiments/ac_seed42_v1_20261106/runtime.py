"""Actual pinned TRL, shared ledger, and signal provenance at every policy loss."""
import sys,inspect,hashlib,importlib.metadata,dataclasses,random,copy,time
from collections import Counter
import torch
from pilot_common import *
sys.path.insert(0,str(ORIGIN))
from audit_common import row_fingerprint,protect,parameter_hash
from c_prm import FrozenCScorer
from c_rewards import RewardLedger,repaired_outcome,valid_for_prm
from advantages import compute_advantages

def components_for_arm(arm,y,m,valid):
    # In A, False means "PRM not requested", not "answer invalid".
    # A's original binary-outcome GRPO tensor is always retained below.
    return compute_advantages(y,m,valid=None if arm=='A' else valid)

class OutcomeLedger:
    def __init__(self,units):
        self.units=units;self.records=[]
        self.cost=dict(prm_input_tokens=0,prm_padded_tokens=0,prm_scored_completions=0,prm_forward_seconds=0.)
    def callback(self,eos,owner):
        def outcome_only(prompts,completions,completion_ids,answer,problem_id,problem,**unused):
            self.records=[]
            for text,ids,gold,ident,question,prompt in zip(completions,completion_ids,answer,problem_id,problem,prompts):
                result=repaired_outcome(text,gold,eos in ids,question=question,problem_id=ident,unit_metadata=self.units)
                self.records.append(dict(optimizer_step_before_update=owner['trainer'].state.global_step,
                    problem_id=ident,gold=gold,completion=text,completion_tokens=len(ids),terminated=eos in ids,
                    status=result['status'],answer=result['answer'],outcome_reward=result['reward'],reward=result['reward'],
                    valid_for_process_reward=False,step_scores=[],min_score=0.,step_count=0,prm_input_tokens=0,
                    raw_no_double_newline='\n\n' not in text.strip(),completion_token_ids=list(ids),prompt_text=prompt))
            return [r['reward'] for r in self.records]
        return outcome_only

def new_accounting():
    return dict(generated_completion_tokens=0,generated_completions=0,generated_groups=0,correct_completions=0,
        truncated_completions=0,prm_input_tokens=0,prm_padded_tokens=0,prm_scored_completions=0,prm_forward_seconds=0.,
        outcome_group_counts={},status_counts={},loss_rows=0,loss_wrong_positive=0,process_nonzero_rows=0,
        process_abs_sum=0.,outcome_abs_sum=0.,positive_advantage_rows=0,empty_rows=0,short_wrong_rows=0)

class PilotMixin:
    def initialize_pilot(self,arm,ledger,run,accounting):
        self.arm=arm;self.ledger=ledger;self.run=run;self.accounting=accounting or new_accounting()
        self.expected={};self.used_tags=set();self.last_generation=None;self.prm=None;self.resume_receipt={}
    def _generate_and_score_completions(self,inputs):
        assert self.model.training and self.accelerator.num_processes==1
        output=super()._generate_and_score_completions(inputs)
        old=output['advantages'];records=self.ledger.records;n=old.numel();assert n==len(records)==32
        step=self.state.global_step
        # Persist real emitted IDs before any optional diagnostic can fail.
        raw=self.run/f'raw_generation_step-{step}.pt'
        assert not raw.exists(),'Uncommitted generation exists; recover it explicitly, never resample silently'
        torch.save({k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in output.items()},raw)
        for record in records:append(self.run/f'raw_generation_step-{step}.jsonl',record)
        for i in range(0,n,8):assert len({r['problem_id'] for r in records[i:i+8]})==1
        # Verify the prompt stream at the actual generation site, including after resume.
        schedule=read(AUDIT/'preflight.json')['first_epoch_question_groups']
        assert step<len(schedule),'Pilot exceeded admitted first-epoch schedule'
        assert [r['problem_id'] for r in records[::8]]==schedule[step],('Prompt skip/order differs',step)
        reward=torch.tensor([r['reward'] for r in records],dtype=torch.float32,device=old.device).reshape(-1,8)
        native=((reward-reward.mean(1,keepdim=True))/(reward.std(1,keepdim=True)+1e-4)).flatten()
        torch.testing.assert_close(old,native,atol=2e-5,rtol=2e-5)
        y=torch.tensor([r['outcome_reward'] for r in records],dtype=torch.float32,device=old.device)
        m=[r['min_score'] for r in records]
        bounded=components_for_arm(self.arm,y,m,[r['valid_for_process_reward'] for r in records])
        if self.arm=='C_bounded':
            actual=bounded['advantages'];output['advantages']=actual
            log=self._logs['advantages'];previous=list(log);log.clear();log.extend(previous[:-n]+actual.cpu().tolist())
        else:
            actual=old
            if self.arm=='A':torch.testing.assert_close(actual,bounded['outcome_component'],atol=2e-5,rtol=2e-5)
        assert torch.isfinite(actual).all()
        self.expected={};self.used_tags=set();serial=self.accounting['generated_completions']
        tags=torch.arange(serial,serial+n,device=old.device,dtype=torch.int64);output['pilot_trace_id']=tags
        for i,tag in enumerate(tags.cpu().tolist()):
            self.expected[tag]=dict(advantage=float(actual[i]),outcome=int(y[i]),fingerprint=row_fingerprint(output,i))
        wrong_positive=int(((y==0)&(actual>0)).sum())
        if self.arm!='C_mix':assert wrong_positive==0
        tokens=int(output['completion_mask'].sum());assert tokens==sum(r['completion_tokens'] for r in records)
        contribution=actual-bounded['outcome_component']
        if self.arm=='C_bounded':
            torch.testing.assert_close(contribution,bounded['process_component'],atol=1e-7,rtol=1e-6)
            assert contribution.abs().max()<=.200001
        a=self.accounting
        a['generated_completion_tokens']+=tokens;a['generated_completions']+=n;a['generated_groups']+=n//8
        a['correct_completions']+=int(y.sum());a['truncated_completions']+=sum(not r['terminated'] for r in records)
        for k,v in self.ledger.cost.items():a[k]+=v
        for k,values in [('status_counts',[r['status'] for r in records]),('outcome_group_counts',[
            'all_wrong' if y[i:i+8].sum()==0 else 'all_correct' if y[i:i+8].sum()==8 else 'mixed' for i in range(0,n,8)])]:
            c=Counter(a[k]);c.update(values);a[k]=dict(c)
        a['process_nonzero_rows']+=int((contribution.abs()>1e-8).sum());a['process_abs_sum']+=float(contribution.abs().sum())
        a['outcome_abs_sum']+=float(bounded['outcome_component'].abs().sum());a['positive_advantage_rows']+=int((actual>0).sum())
        a['empty_rows']+=sum(not r['completion'].strip() for r in records)
        a['short_wrong_rows']+=sum(r['completion_tokens']<=32 and not r['outcome_reward'] for r in records)
        self.last_generation=dict(step_before_update=step,rows=n,tokens=tokens,wrong_positive=wrong_positive,
            mean_abs_process=float(contribution.abs().mean()),max_abs_process=float(contribution.abs().max()),
            second_normalizations=0,prompt_groups=schedule[step])
        for i,r in enumerate(records):
            append(self.run/'rollouts.jsonl',dict(r,advantage=float(actual[i]),outcome_component=float(bounded['outcome_component'][i]),
                process_contribution=float(contribution[i]),tag=int(tags[i]),token_mask_sha256=self.expected[int(tags[i])]['fingerprint'],arm=self.arm))
        append(self.run/'generation_checks.jsonl',self.last_generation)
        return output
    def _compute_loss(self,model,inputs):
        tags=inputs.get('pilot_trace_id');assert tags is not None and tags.dtype==torch.int64
        ids=tags.cpu().tolist();expected=[self.expected[i] for i in ids]
        actual=inputs['advantages'];before=actual.detach().clone()
        values=torch.tensor([r['advantage'] for r in expected],dtype=torch.float32,device=actual.device)
        assert actual.dtype==torch.float32 and not actual.requires_grad and torch.equal(actual,values),'Advantage changed before loss'
        for i,r in enumerate(expected):
            assert row_fingerprint(inputs,i)==r['fingerprint'],'Token/signal alignment changed'
            assert ids[i] not in self.used_tags,'Response reused within num_iterations=1'
            if self.arm!='C_mix' and not r['outcome']:assert actual[i]<=0
        loss=super()._compute_loss(model,inputs)
        assert torch.equal(before,inputs['advantages']) and torch.isfinite(loss)
        self.used_tags.update(ids);self.accounting['loss_rows']+=len(ids)
        self.accounting['loss_wrong_positive']+=sum(not r['outcome'] and r['advantage']>0 for r in expected)
        return loss
    def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
        super()._load_from_checkpoint(resume_from_checkpoint,model=model)
        contract=read(Path(resume_from_checkpoint)/'pilot_checkpoint.json')
        assert adapter_digest(self.model)==contract['actor_adapter_digest']
        self.resume_receipt.update(adapter_exact=True,source=str(resume_from_checkpoint),source_step=contract['global_step'])
    def _load_optimizer_and_scheduler(self,checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint:
            p=Path(checkpoint)
            saved=torch.load(p/'optimizer.pt',map_location='cpu',weights_only=False)
            assert tree_digest(self.optimizer.state_dict())==tree_digest(saved),'Optimizer state restore mismatch'
            saved_scheduler=torch.load(p/'scheduler.pt',map_location='cpu',weights_only=False)
            assert tree_digest(self.lr_scheduler.state_dict())==tree_digest(saved_scheduler),'Scheduler restore mismatch'
            self.resume_receipt.update(optimizer_exact=True,scheduler_exact=True)
            atomic(self.run/'resume_loaded.json',self.resume_receipt)
    def _load_rng_state(self,checkpoint):
        super()._load_rng_state(checkpoint)
        if checkpoint:
            saved=torch.load(Path(checkpoint)/'rng_state.pth',map_location='cpu',weights_only=False)
            assert torch.equal(torch.get_rng_state(),saved['cpu'])
            cuda=saved.get('cuda')
            if isinstance(cuda,list):assert all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),cuda))
            elif cuda is not None:assert torch.equal(torch.cuda.get_rng_state(),cuda)
            self.resume_receipt.update(torch_cpu_rng_exact=True,torch_cuda_rng_exact=True)
            atomic(self.run/'resume_loaded.json',self.resume_receipt)

def trainer_class():
    from trl import GRPOTrainer
    from trl.trainer.utils import shuffle_sequence_dict
    assert importlib.metadata.version('trl')=='0.23.1'
    assert sha(inspect.getsourcefile(GRPOTrainer))=='f217183c84be9b38b23454920eb1b914726aa3c4b9dd4c076d455bdaa67bf1b8'
    assert sha(inspect.getsourcefile(shuffle_sequence_dict))=='07cccc9023d60528544390d70efc0a505d2dc1e92473133c045d33f7ed976360'
    class PilotTrainer(PilotMixin,GRPOTrainer):
        def __init__(self,*args,pilot_arm,pilot_ledger,pilot_run,pilot_accounting=None,**kw):
            super().__init__(*args,**kw);self.initialize_pilot(pilot_arm,pilot_ledger,pilot_run,pilot_accounting)
            assert self.args.loss_type=='grpo' and self.num_generations==8 and self.scale_rewards=='group'
            assert not self.use_liger_loss and self.num_iterations==1 and self.accelerator.num_processes==1
    return PilotTrainer

def build(arm,run,accounting=None,*,target_step):
    import numpy as np
    from datasets import Dataset
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import LoraConfig
    from trl import GRPOConfig
    verify();protocol=read(HERE/'protocol.json');cfg=read(ROOT/'training/baseline_config.json');assets=read(ROOT/'receipts/assets.json')
    saved=torch.load(ROOT/'runs/A_seed17/checkpoint-280/training_args.bin',map_location='cpu',weights_only=False)
    values={f.name:getattr(saved,f.name) for f in dataclasses.fields(saved) if f.init}
    values.update(output_dir=str(run),logging_dir=str(run/'tensorboard'),generation_batch_size=None,
        report_to=[],vllm_gpu_memory_utilization=.15,save_steps=5,save_total_limit=3,seed=42,data_seed=42)
    args=GRPOConfig(**values)
    assert args.max_steps==20000 and args.generation_batch_size==32 and args.steps_per_generation==32
    assert args.max_completion_length==2048 and args.learning_rate==5e-6 and args.beta==.04 and args.temperature==1.
    set_seed(42);tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(assets['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation=cfg['attention_implementation'])
    model.config.use_cache=False
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=cfg['lora_target_modules'],bias='none',task_type='CAUSAL_LM')
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
    ledger=OutcomeLedger(units) if arm=='A' else RewardLedger(None,units);owner={}
    trainer=trainer_class()(model=model,processing_class=tok,args=args,train_dataset=Dataset.from_list(rows(ROOT/'data/prepared/train.jsonl')),
        peft_config=lora,reward_funcs=ledger.callback(tok.eos_token_id,owner),pilot_arm=arm,pilot_ledger=ledger,pilot_run=run,pilot_accounting=accounting)
    owner['trainer']=trainer
    initial=adapter_digest(trainer.model)
    assert args.seed==args.data_seed==42
    common=PAIR/'common_initialization.json'
    initialization={'seed':42,'initial_adapter_digest':initial,'base_model_revision':assets['model_revision'],
                    'base_model_hashes':read(AUDIT/'preflight.json')['base_model_hashes'],'policy_checkpoint_loaded':False}
    if common.exists():assert read(common)==initialization,'Initial policy differs across arms or stages'
    else:
        assert arm=='A' and target_step==2,'A establishes common fresh initialization before any policy update'
        atomic(common,initialization)
    trainer.initial_adapter_digest=initial
    if arm!='A':
        py,npstate=random.getstate(),np.random.get_state()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):scorer=FrozenCScorer(receipt_dir=run)
        random.setstate(py);np.random.set_state(npstate)
        trainer.prm=scorer;ledger.scorer.frozen_scorer=scorer
        actor_ids={id(p) for p in trainer.model.parameters()}
        assert all(id(p) not in actor_ids for p in scorer.model.parameters())
    spec=dict(arm=arm,initial_adapter_digest=initial,source_release_sha256=verify(),protocol_sha256=sha(HERE/'protocol.json'),
        settings={k:v for k,v in args.to_dict().items() if 'token' not in k or k in ['max_prompt_length','max_completion_length']},
        model_revision=assets['model_revision'],C_adapter_sha256=protocol['C_adapter_sha256'] if arm!='A' else None,
        vllm_engine_seed=0,seed=42,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    assert os.environ.get('VLLM_USE_V1')=='1'
    engine_module=type(trainer.llm.llm_engine).__module__
    assert engine_module.startswith('vllm.v1.'),('Expected resolved V1 engine',engine_module)
    spec.update(resolved_vllm_engine_module=engine_module,VLLM_USE_V1='1',stage_target_step=target_step)
    atomic(run/f'resume_build_{target_step}.json',spec)
    return trainer,tok
