"""Independent float64 replay. Run remotely on CPU AFTER the finite run completes.

Does not import the training sampler/signal constructor and never loads a model.
"""
from pathlib import Path
import hashlib,json,sys,math
import numpy as np
import torch

ROOT=Path('/path/to/ece285_math_rl_project')
RUN=ROOT/'runs/Tree_credit_triplet_seed17_48_20261108'
AUDIT=ROOT/'audits/Tree_credit_triplet_seed17_48_20261108'
ARMS=('chain_cb','tree_vr','tree_cb')


def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.open() if s.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not torch.cuda.is_initialized()
    assert read(RUN/'controller.json')['stage']=='complete'
    cp=ROOT/'runs/ABC_seed17_resumed_V1_20261104/C_bounded/checkpoint-288'
    expected_source=read(AUDIT/'expected_source_files.json')
    assert len(expected_source)==8
    assert {n:digest(cp/n) for n in expected_source}==expected_source
    report={};hashes={}
    for arm in ARMS:
        p=RUN/arm;records=lines(p/'rollouts.jsonl');traces=lines(p/'tree_traces.jsonl')
        assert len(records)==1536 and len(traces)==48
        max_error=0.;max_ratio_error=0.;wrong_prefix_positive=0;wrong_suffix_positive=0
        count=0;fallbacks=0;branching_roots=0;prefill=0;expanded=0;unique=0
        settings=read(p/'build.json')['settings'];cap=settings['vllm_importance_sampling_cap']
        for batch,trace in enumerate(traces):
            step=288+batch;raw_path=p/f'raw_step_{step}.pt'
            raw=torch.load(raw_path,map_location='cpu',weights_only=False)
            hashes[str(raw_path.relative_to(ROOT))]=digest(raw_path)
            rr=records[batch*32:(batch+1)*32]
            assert trace['step']==step and trace['adapter_id']==step+1
            assert len(trace['rows'])==32
            mask=raw['completion_mask'].bool().numpy()
            if arm=='chain_cb':prefill+=int(raw['prompt_mask'].sum())
            target=np.zeros(mask.shape,dtype=np.float64)
            target_out=np.zeros_like(target);target_proc=np.zeros_like(target)
            if arm!='chain_cb':
                assert len(trace['roots'])==8 and trace['generation_rounds']==2
                initial=sum(len(r['initial_ids']) for r in trace['roots'])
                added=sum(r['suffix_generation_tokens'] for i,r in enumerate(trace['rows']) if i%4)
                assert initial+added==trace['unique_generation_tokens']
                assert sum(len(r['full_token_ids']) for r in trace['rows'])-trace['unique_generation_tokens']==3*sum(len(r['prefix_token_ids']) for r in trace['roots'])
                for r in trace['roots']:
                    assert len(r['prefix_token_ids'])==r['cut']
                    assert r['prefix_token_ids']==r['initial_ids'][:r['cut']]
                    fallbacks+=bool(r['fallback']);branching_roots+=bool(r['cut'])
                    prefill+=len(r['prompt_token_ids'])+3*(len(r['prompt_token_ids'])+r['cut'])
            for g in range(4):
                ys=np.array([r['outcome_reward'] for r in rr[g*8:g*8+8]],dtype=np.float64)
                ms=np.array([r['min_score'] for r in rr[g*8:g*8+8]],dtype=np.float64)
                root=ys.mean();mu=ms[ys==1].mean() if ys.sum() else 0.
                for j in range(8):
                    i=g*8+j;r=rr[i];meta=trace['rows'][i]
                    ids=raw['completion_ids'][i][raw['completion_mask'][i].bool()].tolist()
                    assert ids==r['completion_token_ids']==meta['full_token_ids']
                    n=len(ids);assert n==int(mask[i].sum())<=2048
                    cut=meta['prefix_length'] if arm!='chain_cb' else 0
                    if arm=='chain_cb':
                        a=(ys[j]-root)/(ys.std(ddof=1)+1e-4)
                        out=np.full(n,a)
                    else:
                        origin=trace['roots'][g*2+j//4]
                        prompt_ids=raw['prompt_ids'][i][raw['prompt_mask'][i].bool()].tolist()
                        assert prompt_ids==origin['prompt_token_ids'],'HF loss prompt differs from actual vLLM sampling prompt'
                        parent=ys[j//4*4:j//4*4+4].mean() if cut else root
                        out=np.r_[np.full(cut,parent-root),np.full(n-cut,2*ys[j]-root-parent)]
                        # parent-root already incorporates (GA+LA)/sqrt(4).
                        if cut:
                            assert ids[:cut]==origin['initial_ids'][:cut]
                            assert meta['sampling_logprobs'][:cut]==origin['initial_logprobs'][:cut]
                    delta=.2*ys[j]*(ms[j]-mu) if arm!='tree_vr' else 0.
                    proc=np.r_[np.zeros(cut),np.full(n-cut,delta)]
                    target_out[i,:n]=out;target_proc[i,:n]=proc;target[i,:n]=out+proc
                    if not ys[j]:
                        assert not proc.any()
                        wrong_suffix_positive+=int(((out+proc)[cut:]>1e-7).sum())
                        wrong_prefix_positive+=int(((out+proc)[:cut]>1e-7).sum())
                    if 'old_per_token_logps' in raw and raw['old_per_token_logps'] is not None:
                        hf=raw['old_per_token_logps'][i,:n].double().numpy()
                        sample=np.array(meta['sampling_logprobs'],dtype=np.float64)
                        expected_ratio=np.exp(np.minimum(hf-sample,math.log(cap)))
                        observed=raw['importance_sampling_ratio'][i,:n].double().numpy()
                        error=np.abs(expected_ratio-observed).max(initial=0)
                        max_ratio_error=max(max_ratio_error,float(error))
                    expanded+=n;count+=1
            unique+=trace['unique_generation_tokens']
            for name,expected in [('advantages',target),('outcome_component',target_out),('process_component',target_proc)]:
                actual=raw[name].double().numpy()
                err=float(np.abs(actual-expected).max(initial=0));max_error=max(max_error,err)
                assert err<=2e-6,(arm,step,name,err)
                assert not actual[~mask].any()
            assert not (raw['process_component'][[not r['outcome_reward'] for r in rr]]>0).any()
        assert wrong_suffix_positive==0
        assert max_ratio_error<2e-5
        progress=read(p/'progress.json')
        assert progress['generated_tokens']==unique and progress['expanded_leaf_tokens']==expanded
        if arm=='chain_cb':assert wrong_prefix_positive==0
        report[arm]={'batches':48,'responses_verified':count,'max_credit_float64_error':max_error,
                     'max_sampling_ratio_reconstruction_error':max_ratio_error,
                     'wrong_positive_shared_prefix_tokens':wrong_prefix_positive,'wrong_positive_exclusive_suffix_tokens':0,
                     'wrong_positive_process_tokens':0,'unique_generation_tokens':unique,
                     'expanded_leaf_training_tokens':expanded,'rollout_requested_prefill_tokens':prefill,
                     'nonempty_branching_roots':branching_roots,'fallback_roots':fallbacks}
    assert not torch.cuda.is_initialized()
    result={'passed':True,'CUDA_initialized':False,'source8_files_unchanged':True,'arms':report,'raw_tensor_sha256':hashes,
            'training_constructor_imported':False,'scope':'Independent CPU replay of all 144 batches/4608 responses, components, prefix provenance and sampling correction.'}
    out=AUDIT/'independent_tensor_acceptance.json'
    assert not out.exists(),'Completed independent acceptance must be preserved'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_tensor_sha256'},indent=2))


if __name__=='__main__':main()
