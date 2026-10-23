"""Complete official OpenRLHF signal integration on a single-card backend. No training entry point.

TRL is used only for the already-tested actor/LoRA/vLLM initialization and
weight synchronization. Its reward, advantage, compute_loss and train methods
are explicitly disabled. All PURE tensors and backward calls live here.
"""
import contextlib
import hashlib
import importlib.metadata
import inspect
import random
import time
import numpy as np
import torch
from common import ROOT, N, BATCH, GROUP, read, atomic, sha
from signals import make_signals, micro_loss, official_functions
from alignment import OfficialBoundaryBuilder

def adapter_hash(model):
    from peft import get_peft_model_state_dict
    h = hashlib.sha256()
    for name, value in sorted(get_peft_model_state_dict(model).items()):
        t = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        t = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def blocked(*a, **kw):
    raise RuntimeError('Formal training / original GRPO reward-loss path is disabled in this preflight')

class OpenRLHFIntegration:
    def __init__(self, run, rows):
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
        from trl import GRPOConfig, GRPOTrainer
        import trl.trainer.utils as trl_utils
        assert sha(inspect.getsourcefile(GRPOTrainer)) == 'f217183c84be9b38b23454920eb1b914726aa3c4b9dd4c076d455bdaa67bf1b8'
        assert sha(inspect.getsourcefile(trl_utils)) == '07cccc9023d60528544390d70efc0a505d2dc1e92473133c045d33f7ed976360'
        for package, expected in {'torch':'2.8.0','transformers':'4.56.2','trl':'0.23.1','peft':'0.17.1','vllm':'0.10.2'}.items():
            assert importlib.metadata.version(package) == expected
        # Subclass before construction, so accidental calls fail closed.
        class BackendOnly(GRPOTrainer):
            train = blocked
            _generate_and_score_completions = blocked
            _calculate_rewards = blocked
            compute_loss = blocked
            _compute_loss = blocked
            create_optimizer = blocked
            create_optimizer_and_scheduler = blocked
        self.run = run
        cfg = read(ROOT / 'training/baseline_config.json')
        assets = read(ROOT / 'receipts/assets.json')
        assert assets['model_revision'] == '4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2'
        set_seed(17)
        tok = AutoTokenizer.from_pretrained(assets['model_path'], local_files_only=True, padding_side='left')
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        actor = AutoModelForCausalLM.from_pretrained(assets['model_path'], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        actor.config.use_cache = False
        args = GRPOConfig(output_dir=str(run / 'disabled_training_output'), seed=17, data_seed=17,
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            per_device_train_batch_size=1, gradient_accumulation_steps=32,
            num_generations=8, max_prompt_length=2048, max_completion_length=2048,
            temperature=1.0, top_p=1.0, top_k=-1, learning_rate=1e-6,
            lr_scheduler_type='constant', warmup_steps=0, weight_decay=0.0,
            beta=0.0, num_iterations=1, loss_type='grpo', scale_rewards='none',
            use_vllm=True, vllm_mode='colocate', vllm_gpu_memory_utilization=0.3,
            vllm_enable_sleep_mode=False, vllm_importance_sampling_correction=False,
            mask_truncated_completions=False, report_to='none', save_strategy='no',
            max_steps=1)
        # beta=0 disables the UNUSED backend KL loss/ref; OpenRLHF uses its own
        # explicit old/ref log-probs and official default k1 KL reward coefficient 0.001.
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
            target_modules=cfg['lora_target_modules'], bias='none', task_type='CAUSAL_LM')
        def unused_reward(**kwargs):
            return blocked()
        self.backend = BackendOnly(model=actor, processing_class=tok, args=args,
            train_dataset=Dataset.from_list(rows), peft_config=lora, reward_funcs=unused_reward)
        self.model, self.tokenizer = self.backend.model, tok
        assert self.backend.accelerator.num_processes == 1 and torch.cuda.device_count() == 1
        assert self.backend.args.generation_batch_size == BATCH
        assert self.backend.optimizer is None
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.enable_input_require_grads()
        self.model.train()
        # disable_dropout defaults to True in the original backend; assert it.
        assert all(m.p == 0 for m in self.model.modules() if isinstance(m, torch.nn.Dropout))
        self.builder = OfficialBoundaryBuilder(tok)
        self.official_log_probs = official_functions()['log_probs_from_logits']
        self.initial_adapter = adapter_hash(self.model)
        assert self.initial_adapter == 'c634db8e3d4f356f71e8cbea2b4b79b90d9d7c0a58073b5046841d26da44f7f7'
        atomic(run / 'initialization.json', dict(actor_adapter_sha256=self.initial_adapter,
            matches_corrected_AB_initialization=True, optimizer_constructed=False,
            backend_original_reward_and_loss_disabled=True, actual_generation_batch=BATCH,
            native_actor_eos=tok.eos_token_id, native_actor_pad=tok.pad_token_id,
            packages={n:importlib.metadata.version(n) for n in ['torch','transformers','trl','peft','vllm']},
            model_revision=assets['model_revision'], backend_args=args.to_dict()))

    def load_prm(self):
        from prm import FrozenOfficialPRM
        states = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()
        try:
            self.prm = FrozenOfficialPRM(ROOT/'models/pure_official_prm_7b', self.tokenizer,
                device='cuda:0', assets_path=ROOT/'audits/PURE_official_prm_offline_20261022/model_assets.json',
                receipt_dir=self.run)
        finally:
            random.setstate(states[0]); np.random.set_state(states[1])
            torch.set_rng_state(states[2]); torch.cuda.set_rng_state(states[3])
        assert adapter_hash(self.model) == self.initial_adapter
        assert not set(map(id,self.model.parameters())) & set(map(id,self.prm.model.parameters()))

    def generate(self, questions):
        from vllm import SamplingParams
        assert len(questions) == 4
        repeated = [r for r in questions for _ in range(GROUP)]
        encoded = [self.tokenizer.encode(r['prompt'],add_special_tokens=False)[-2048:] for r in repeated]
        self.backend._move_model_to_vllm()
        assert adapter_hash(self.model) == self.initial_adapter
        sampling = SamplingParams(n=1,temperature=1.0,top_p=1.0,top_k=-1,
            max_tokens=N,min_tokens=1,skip_special_tokens=False,include_stop_str_in_output=True)
        torch.cuda.synchronize(); started=time.perf_counter()
        outputs = self.backend.llm.generate([{'prompt_token_ids':x} for x in encoded],
                                           sampling_params=sampling,use_tqdm=False)
        torch.cuda.synchronize(); seconds=time.perf_counter()-started
        rows=[]
        for question,prompt_ids,result in zip(repeated,encoded,outputs):
            item=result.outputs[0]; emitted=list(item.token_ids)
            assert list(result.prompt_token_ids)==prompt_ids and 0<len(emitted)<=N
            alignment=self.builder.build(prompt_ids,emitted)
            processed=alignment['actor_completion_ids']
            rows.append(dict(problem_id=question['problem_id'],question=question['problem'],gold=question['answer'],
                prompt=question['prompt'],prompt_ids=prompt_ids,
                emitted_completion_ids=emitted,completion_ids=processed,
                emitted_completion=self.tokenizer.decode(emitted,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                completion=self.tokenizer.decode(processed,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                generated_special_text=self.tokenizer.decode(emitted,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                original_terminated=self.tokenizer.eos_token_id in emitted,
                finish_reason=item.finish_reason,stop_reason=item.stop_reason,
                alignment=alignment,source='new vLLM exact emitted IDs, followed by pinned official process_sequences',
                sampler=dict(min_tokens=1,max_tokens=N,temperature=1.0,top_p=1.0,top_k=-1)))
        assert len(rows)==BATCH
        return rows,dict(seconds=seconds,emitted_tokens=sum(len(r['emitted_completion_ids']) for r in rows),
                         actor_action_tokens=sum(len(r['completion_ids']) for r in rows),
                         fresh_generation=True,min_tokens=1,rows=len(rows))

    def logps(self,row):
        # Official Actor.forward semantics, restricted to completion predictor
        # logits to reduce memory. No temperature division or second whitening.
        prompt,completion=row['prompt_ids'],row['completion_ids'];n=len(completion)
        ids=torch.tensor([prompt+completion],dtype=torch.long,device='cuda')
        attention=torch.tensor([row['alignment']['actor_attention_mask']],dtype=torch.long,device='cuda')
        assert attention.shape==ids.shape
        position_ids=attention.cumsum(-1)-1;position_ids.masked_fill_(attention==0,1)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            logits=self.model(input_ids=ids,attention_mask=attention,position_ids=position_ids,
                              use_cache=False,logits_to_keep=n+1).logits[:,:-1,:].float()
        assert logits.shape[1]==n
        return self.official_log_probs(logits,ids[:,-n:])

    @torch.no_grad()
    def old_and_reference(self,rows):
        old,ref=torch.zeros((BATCH,N)),torch.zeros((BATCH,N))
        for i,row in enumerate(rows):
            n=len(row['completion_ids']);old[i,:n]=self.logps(row)[0].cpu()
            with self.model.disable_adapter():ref[i,:n]=self.logps(row)[0].cpu()
        return old,ref

    def prepare(self,arm,rows,scored,old,ref):
        logits=None if arm=='P_VR' else [s['step_logits'] for s in scored]
        return make_signals(arm.replace('_','-'),[len(r['completion_ids']) for r in rows],
            [r['outcome'] for r in rows],logits,
            [r['alignment']['actor_boundary_positions'] for r in rows],
            [r['problem_id'] for r in rows],old,ref,kl_coef=.001,use_kl_estimator_k3=False,
            step_logits_dtype='bfloat16',prm_transform_device='cuda:0')

    def backward_no_update(self,batch,rows):
        self.model.zero_grad(set_to_none=True);audit=[]
        for i,row in enumerate(rows):
            logps=self.logps(row);loss,trace=micro_loss(batch,i,logps)
            assert torch.isfinite(loss);loss.backward();audit.append(trace)
            if i%8==7:print(f'{batch.arm}: backward {i+1}/32; no optimizer',flush=True)
        gradients=[]
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None and torch.isfinite(param.grad).all(),name
                gradients.append((name,float(param.grad.detach().float().square().sum())))
            else:assert param.grad is None,name
        norm=sum(v for _,v in gradients)**.5
        assert norm>0 and self.backend.optimizer is None
        batch.assert_sealed()
        self.model.zero_grad(set_to_none=True)
        assert adapter_hash(self.model)==self.initial_adapter
        return dict(arm=batch.arm,loss_calls=len(audit),valid_tokens=int(batch.mask.sum()),
                    gradient_norm=norm,nonzero_gradient_tensors=sum(v>0 for _,v in gradients),
                    trainable_tensors=len(gradients),optimizer_steps=0,
                    advantage_passed_to_loss_unchanged=True,loss_trace=audit)

