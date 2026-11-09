"""Per-instance LLM facade. No global patch; frozen actor for both tree rounds."""
import copy
import hashlib
import json
import math
import time
from tree_core import choose_fork


def lp_values(completion):
    ids = list(completion.token_ids)
    assert len(ids) == len(completion.logprobs)
    values = [float(lp[token].logprob) for token, lp in zip(ids, completion.logprobs)]
    assert all(math.isfinite(v) for v in values)
    # The pinned TRL reader takes the first element; logprobs=0 returns exactly the sampled token.
    assert all(len(lp) == 1 and token in lp for token, lp in zip(ids, completion.logprobs))
    return values


class TreeSamplerFacade:
    def __init__(self, llm, owner, tokenizer):
        self.inner = llm
        self.owner = owner
        self.tokenizer = tokenizer
        self.last_trace = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def generate(self, prompts, sampling_params, **kwargs):
        assert len(prompts) == 32 and sampling_params.n == 1 and sampling_params.logprobs == 0
        assert sampling_params.max_tokens == 2048 and "lora_request" in kwargs
        for i in range(0, 32, 8):
            assert all(p == prompts[i] for p in prompts[i:i+8])
        started = time.monotonic()
        adapter_id = kwargs["lora_request"].lora_int_id
        step = self.owner.state.global_step
        if self.owner.experiment_arm == "chain_cb":
            generated = self.inner.generate(prompts, sampling_params=sampling_params, **kwargs)
            self.last_trace = dict(arm="chain_cb", step=step, adapter_id=adapter_id, rows=[
                dict(prefix_length=0, full_token_ids=list(o.outputs[0].token_ids),
                     sampling_logprobs=lp_values(o.outputs[0]), group=i//8, leaf=i%8,
                     finish_reason=o.outputs[0].finish_reason) for i, o in enumerate(generated)],
                unique_generation_tokens=sum(len(o.outputs[0].token_ids) for o in generated),
                roots=[], generation_rounds=1, seconds=time.monotonic()-started)
            return generated

        # Two independent initial responses per prompt, one expansion round.
        initial_prompts = [prompts[g*8] for g in range(4) for _ in range(2)]
        roots = self.inner.generate(initial_prompts, sampling_params=sampling_params, **kwargs)
        assert len(roots) == 8 and all(len(o.outputs) == 1 for o in roots)
        plans, branch_prompts, branch_params = [], [], []
        for i, root in enumerate(roots):
            value = root.outputs[0]
            ids, probs = list(value.token_ids), lp_values(value)
            assert ids and len(ids) <= 2048
            pieces = [self.tokenizer.decode([x], skip_special_tokens=False) for x in ids]
            plan = choose_fork(ids, probs, pieces, self.tokenizer.eos_token_id)
            cut = plan["cut"]
            prefix = ids[:cut]
            assert self.tokenizer.eos_token_id not in prefix
            pids = list(root.prompt_token_ids)
            plan.update(root_index=i, group=i//2, initial_ids=ids, initial_logprobs=probs,
                        prompt_token_ids=pids, prefix_token_ids=prefix,
                        selected_token_id=ids[cut] if cut else None)
            plans.append(plan)
            for _ in range(3):
                branch_prompts.append(dict(prompt_token_ids=pids+prefix))
                params = copy.deepcopy(sampling_params)
                params.max_tokens = 2048 - cut  # TOTAL completion path cap, not an extra suffix budget.
                branch_params.append(params)
        branches = self.inner.generate(branch_prompts, sampling_params=branch_params, **kwargs)
        assert len(branches) == 24
        assert self.owner.state.global_step == step
        assert kwargs["lora_request"].lora_int_id == adapter_id
        result, rows = [], []
        for i, (root, plan) in enumerate(zip(roots, plans)):
            root_value = root.outputs[0]
            cut, prefix = plan["cut"], plan["prefix_token_ids"]
            for child in range(4):
                source = root if child == 0 else branches[i*3+child-1]
                value = source.outputs[0]
                suffix_ids = list(root_value.token_ids[cut:]) if child == 0 else list(value.token_ids)
                suffix_lps = list(root_value.logprobs[cut:]) if child == 0 else list(value.logprobs)
                full = prefix + suffix_ids
                dictionaries = list(root_value.logprobs[:cut]) + suffix_lps
                assert full and len(full) <= 2048 and len(dictionaries) == len(full)
                if child != 0:
                    assert list(source.prompt_token_ids) == plan["prompt_token_ids"] + prefix
                o = copy.copy(source)
                leaf = copy.copy(value)
                leaf.token_ids = full
                leaf.logprobs = dictionaries
                leaf.text = self.tokenizer.decode(full, skip_special_tokens=True)
                leaf.cumulative_logprob = sum(float(lp[token].logprob) for token, lp in zip(full, dictionaries))
                o.outputs = [leaf]
                o.prompt_token_ids = plan["prompt_token_ids"]
                o.prompt = prompts[(i//2)*8]
                result.append(o)
                probs = lp_values(leaf)
                assert probs[:cut] == plan["initial_logprobs"][:cut]
                rows.append(dict(group=i//2, leaf=(i%2)*4+child, root_index=i,
                                 prefix_length=cut, full_token_ids=full,
                                 sampling_logprobs=probs, suffix_generation_tokens=len(suffix_ids),
                                 finish_reason=leaf.finish_reason, fork_fallback=plan["fallback"]))
        assert len(result) == len(rows) == 32
        unique = sum(len(o.outputs[0].token_ids) for o in roots + branches)
        self.last_trace = dict(arm=self.owner.experiment_arm, step=step, adapter_id=adapter_id,
                              rows=rows, roots=plans, generation_rounds=2,
                              unique_generation_tokens=unique, seconds=time.monotonic()-started,
                              full_leaf_tokens=sum(len(r["full_token_ids"]) for r in rows),
                              shape=dict(M=2,N=1,L=1,T=3),
                              prefix_logprob_provenance_exact=True)
        return result
