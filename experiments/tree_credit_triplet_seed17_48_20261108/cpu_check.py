"""Algebra, sampler/provenance and historical-control checks; no CUDA context."""
import argparse
import importlib.util
import itertools
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace as NS
from tree_core import group_coefficients,choose_fork,make_rows
from sampler import TreeSamplerFacade


class FakeTokenizer:
    eos_token_id=999
    def decode(self,ids,**kwargs):return ''.join('EOS' if i==999 else f't{i}' for i in ids)


class FakeLLM:
    def __init__(self):self.calls=[]
    def generate(self,prompts,sampling_params,**kwargs):
        self.calls.append((prompts,sampling_params))
        values=[]
        for i,p in enumerate(prompts):
            is_branch=isinstance(p,dict)
            ids=[80+i,90+i,999] if is_branch else [11,12,13,14,15,999]
            ps=[-.2]*len(ids) if is_branch else [-.1,-.4,-2.,-.6,-.1,-.02]
            completion=NS(token_ids=ids,logprobs=[{t:NS(logprob=v)} for t,v in zip(ids,ps)],
                          text='fixture',finish_reason='stop',cumulative_logprob=sum(ps))
            values.append(NS(prompt_token_ids=p['prompt_token_ids'] if is_branch else [1,2,3],
                             prompt=p,outputs=[completion]))
        return values


def main(stdlib_only=False):
    checks=0
    # Exhaust every binary labeling and both real/fallback prefix configurations.
    for ys in itertools.product((0,1),repeat=8):
        for cuts in ((0,0),(2,0),(0,3),(2,3)):
            for arm in ('chain_cb','tree_vr','tree_cb'):
                minima=[.1*(i+1) for i in range(8)]
                rows=group_coefficients(list(ys),minima,[cuts[0]]*4+[cuts[1]]*4,arm)
                root=sum(ys)/8
                for i,r in enumerate(rows):
                    if arm!='chain_cb':
                        # Independent subtree sums, separate from the signal constructor.
                        parent=sum(ys[i//4*4:i//4*4+4])/4 if cuts[i//4] else root
                        assert math.isclose(r['suffix_credit'],2*ys[i]-root-parent,abs_tol=1e-12)
                        prefix=(2*parent-2*root)/2 if cuts[i//4] else 0
                        assert math.isclose(r['prefix_credit'],prefix,abs_tol=1e-12)
                    if not ys[i]:assert r['process_bonus']==0 and r['suffix_credit']<=0
                    assert abs(r['process_bonus'])<=.2
                if not any(ys):assert all(r['prefix_credit']==r['suffix_credit']==r['process_bonus']==0 for r in rows)
                checks+=1
    f=choose_fork([11,12,13,999],[-.1,-3.,-.2,-.01],['one','two','answer','EOS'],999)
    assert f['cut']==1
    assert choose_fork([999],[-.1],['EOS'],999)['cut']==0
    assert choose_fork([11,12,999],[-.1,-3.,-.1],['answer','x','EOS'],999)['cut']==0
    checks+=3
    owner=NS(experiment_arm='tree_cb',state=NS(global_step=288))
    llm=FakeLLM();proxy=TreeSamplerFacade(llm,owner,FakeTokenizer())
    params=NS(n=1,logprobs=0,max_tokens=2048)
    out=proxy.generate([str(i//8) for i in range(32)],params,lora_request=NS(lora_int_id=289),use_tqdm=False)
    assert len(out)==32 and len(llm.calls)==2
    assert len(llm.calls[0][0])==8 and len(llm.calls[1][0])==24
    assert all(p.max_tokens==2046 for p in llm.calls[1][1])
    assert all(r['prefix_length']==2 for r in proxy.last_trace['rows'])
    assert proxy.last_trace['unique_generation_tokens']==8*6+24*3
    assert proxy.last_trace['full_leaf_tokens']==8*6+24*5
    records=[]
    for i,r in enumerate(proxy.last_trace['rows']):
        records.append(dict(problem_id=str(i//8),completion_token_ids=r['full_token_ids'],
                            outcome_reward=[1,1,1,0,1,0,0,0][i%8],min_score=.1*(i%8+1)))
    made=make_rows(records,proxy.last_trace,'tree_cb')
    assert made[3]['outcome_component'][:2]==[.25,.25]
    assert made[3]['outcome_component'][2:]==[-1.25]*(len(records[3]['completion_token_ids'])-2)
    assert not any(made[3]['process_component'])
    checks+=8
    exact_control=False
    if not stdlib_only:
        import torch
        assert not torch.cuda.is_initialized()
        path=Path(__file__).resolve().parents[2]/'experiments/abc_seed17_resumed_v1_20261104/advantages.py'
        spec=importlib.util.spec_from_file_location('historical_bounded_check',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for ys in itertools.product((0,1),repeat=8):
            ms=[.1*(i+1) for i in range(8)]
            ref=mod.compute_advantages(ys,ms)['advantages'].tolist()
            py=group_coefficients(list(ys),ms,[0]*8,'chain_cb')
            assert all(math.isclose(a,r['suffix_credit']+r['process_bonus'],rel_tol=2e-6,abs_tol=2e-6) for a,r in zip(ref,py))
        exact_control=True
        assert not torch.cuda.is_initialized()
    result={'passed':True,'algebra_and_mock_sampler_checks':checks,'all_256_binary_patterns_checked':True,
            'historical_torch_control_checked':exact_control,'CUDA_initialized':False,
            'scope':'CPU math and mocked sampler only; actual models checked separately.'}
    print(result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stdlib-only',action='store_true');a=p.parse_args()
    main(a.stdlib_only)
