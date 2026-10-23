"""Independent PURE model integration. No formal training entry point.

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
from signals import make_signals, micro_loss
from alignment import align_actor

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

class PureIntegration:
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
        # beta=0 disables the UNUSED backend KL loss/ref; PURE uses its own
        # explicit old/ref log-probs and KL reward coefficient 0.001.
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
        self.initial_adapter = adapter_hash(self.model)
        assert self.initial_adapter == 'c634db8e3d4f356f71e8cbea2b4b79b90d9d7c0a58073b5046841d26da44f7f7'
        atomic(run / 'initialization.json', dict(actor_adapter_sha256=self.initial_adapter,
            matches_corrected_AB_initialization=True, optimizer_constructed=False,
            backend_original_reward_and_loss_disabled=True, actual_generation_batch=BATCH,
            native_actor_eos=tok.eos_token_id, native_actor_pad=tok.pad_token_id,
            packages={n:importlib.metadata.version(n) for n in ['torch','transformers','trl','peft','vllm']},
            model_revision=assets['model_revision'], backend_args=args.to_dict()))

    def load_prm(self):
        from prm import PureQwenPRM
        states = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()
        try:
            self.prm = PureQwenPRM(ROOT / 'models/qwen25_math_prm_7b', receipt_dir=self.run)
        finally:
            random.setstate(states[0]); np.random.set_state(states[1])
            torch.set_rng_state(states[2]); torch.cuda.set_rng_state(states[3])
        assert adapter_hash(self.model) == self.initial_adapter
        assert not set(map(id, self.model.parameters())) & set(map(id, self.prm.model.parameters()))

    def generate(self, questions):
        from vllm import SamplingParams
        assert len(questions) == 4
        # Same raw engine seed and ordered repeated prompts as old colocate path.
        repeated = [r for r in questions for _ in range(GROUP)]
        prompts = []
        for row in repeated:
            ids = self.tokenizer.encode(row['prompt'], add_special_tokens=False)[-2048:]
            prompts.append(self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        self.backend._move_model_to_vllm()
        assert adapter_hash(self.model) == self.initial_adapter
        sampling = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1,
            repetition_penalty=1.0, min_p=0.0, max_tokens=N, logprobs=0)
        torch.cuda.synchronize(); t = time.perf_counter()
        generated = self.backend.llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        torch.cuda.synchronize(); seconds = time.perf_counter() - t
        records = []
        for row, result in zip(repeated, generated):
            output = result.outputs[0]
            ids = list(output.token_ids)
            text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            prompt_ids = list(result.prompt_token_ids)
            assert prompt_ids == self.tokenizer.encode(row['prompt'], add_special_tokens=False)[-2048:]
            assert 0 < len(ids) <= N
            records.append(dict(problem_id=row['problem_id'], question=row['problem'], gold=row['answer'],
                prompt=row['prompt'], prompt_ids=prompt_ids, completion=text, completion_ids=ids,
                terminated=self.tokenizer.eos_token_id in ids, finish_reason=output.finish_reason,
                source='new real vLLM generation; exact emitted token IDs',
                vllm_text_matches_hf_decode=(output.text == text)))
        assert len(records) == BATCH
        return records, dict(seconds=seconds, tokens=sum(len(r['completion_ids']) for r in records))

    def logps(self, row):
        from trl.trainer.utils import selective_log_softmax
        prompt, completion = row['prompt_ids'], row['completion_ids']
        ids = torch.tensor([prompt + completion], dtype=torch.long, device='cuda')
        n = len(completion)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                logits_to_keep=n + 1).logits[:, :-1, :]
            assert logits.shape[1] == n
            logp = selective_log_softmax(logits, ids[:, -n:]).float()
        return logp

    @torch.no_grad()
    def old_and_reference(self, rows):
        old, ref = torch.zeros((BATCH, N), device='cuda'), torch.zeros((BATCH, N), device='cuda')
        for i, row in enumerate(rows):
            n = len(row['completion_ids'])
            old[i, :n] = self.logps(row)[0]
            with self.model.disable_adapter():
                ref[i, :n] = self.logps(row)[0]
        return old, ref

    def prepare(self, arm, rows, scored, old, ref):
        if arm == 'P_VR':
            # A standalone P-VR run must never require a PRM or its native
            # context/alignment gates. The joint no-update probe scores both
            # arms' shared responses only to compare their tensors.
            positions, probabilities = [[] for _ in rows], [[] for _ in rows]
        else:
            assert scored is not None and len(scored) == len(rows)
            positions = [align_actor(self.tokenizer, r['completion_ids'], r['completion'], s['spans'],
                         empty_terminal_step=s['empty_terminal_step'])['actor_positions'] for r, s in zip(rows, scored)]
            probabilities = [s['step_scores'] for s in scored]
        return make_signals(arm, [len(r['completion_ids']) for r in rows], [r['outcome'] for r in rows],
            probabilities, positions, [r['problem_id'] for r in rows], old, ref), positions

    def backward_no_update(self, signals, rows):
        self.model.zero_grad(set_to_none=True)
        audit = []
        for i, row in enumerate(rows):
            logp = self.logps(row)
            loss, trace = micro_loss(signals, i, logp)
            assert torch.isfinite(loss)
            loss.backward()  # Already scaled by 1/32; no Accelerator rescaling.
            audit.append(trace)
            if i % 8 == 7:
                print(f'{signals.arm}: backward {i + 1}/32; no optimizer', flush=True)
        gradients = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
                gradients.append((name, float(param.grad.detach().float().square().sum())))
            else:
                assert param.grad is None, name
        norm = sum(v for _, v in gradients) ** 0.5
        assert norm > 0 and self.backend.optimizer is None
        assert torch.equal(signals.advantages, signals.sealed_advantages)
        self.model.zero_grad(set_to_none=True)
        assert adapter_hash(self.model) == self.initial_adapter
        return dict(arm=signals.arm, loss_calls=len(audit), valid_tokens=int(signals.mask.sum()),
            gradient_norm=norm, nonzero_gradient_tensors=sum(v > 0 for _, v in gradients),
            trainable_tensors=len(gradients), optimizer_steps=0, whitening_calls=signals.whitening_calls,
            advantage_passed_to_loss_unchanged=True, loss_trace=audit)
