"""Frozen official PURE PRM; same saved responses, no policy or optimizer."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pure_integration_20261022'))
from common import atomic, read, read_jsonl, sha, verify_release
from alignment import canonical_view

TEMPLATE = '{}\n\nPlease reason step by step with steps separated by "\n\n", and put your final answer within \\boxed{{}}.'

def encode(tokenizer, question, steps, variant):
    prompt = question if variant == 'native_raw' else TEMPLATE.format(question)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    sep = tokenizer.encode('\n', add_special_tokens=False)
    assert len(sep) == 1
    positions = []
    for step in steps:
        ids.extend(tokenizer.encode(step, add_special_tokens=False)); ids.extend(sep)
        positions.append(len(ids)-1)
    assert positions and all(ids[j] == sep[0] for j in positions)
    assert len(ids) <= 4096, 'Context overflow; no truncation or score substitution'
    return dict(input_ids=ids,positions=positions,steps=steps,prm_input_tokens=len(ids))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--qwen',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    lock=(a.output/'scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(a.output/'status.json',dict(status='starting',pid=os.getpid(),new_generations=0,optimizer_steps=0))
    verify_release(); release_before=sha(ROOT/'experiments/pure_integration_20261022/release.json')
    rows, old=read_jsonl(a.input),read_jsonl(a.qwen)
    assert len(rows)==len(old)==288 and sum(not r['completion'].strip() for r in rows)==9
    assets=read(a.assets);assert assets['complete'] and assets['revision']=='b0cce6763f2317762c9cb46720c8fe7b07f44a88'
    for f in assets['files']: assert sha(a.model/f['file'])==f['sha256']
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from safetensors import safe_open
    torch.set_num_threads(2); torch.manual_seed(17)
    start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='right')
    model,loading=AutoModelForTokenClassification.from_pretrained(a.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True,num_labels=2)
    assert not loading['missing_keys'] and not loading['mismatched_keys']
    index=read(a.model/'model.safetensors.index.json')['weight_map'];state=model.state_dict()
    assert set(loading['unexpected_keys']) <= {'lm_head.weight'}
    assert set(state) == set(index)-set(loading['unexpected_keys'])
    checked=0
    for shard in sorted(set(index.values())):
        with safe_open(a.model/shard,framework='pt',device='cpu') as weights:
            for key in weights.keys():
                if key in state: assert torch.equal(state[key],weights.get_tensor(key));checked+=1
    del state
    model=model.to('cuda:0').requires_grad_(False).eval();model.config.use_cache=False
    versions={n:v._version for n,v in model.named_parameters()}
    def frozen():
        assert not model.training
        for n,v in model.named_parameters(): assert not v.requires_grad and v.grad is None and versions[n]==v._version
    frozen()
    atomic(a.output/'model_loading.json',dict(loading=loading,matched_tensors=checked,model_class=type(model).__name__,
        marker=tokenizer.encode('\n',add_special_tokens=False),frozen=True,num_labels=model.config.num_labels,
        torch_version=torch.__version__,attention='sdpa',dtype='bfloat16',probability_computation='float32 same as saved Qwen audit'))
    variants=['native_raw','official_rl_prompt']; encoded={v:[] for v in variants}; written=[]
    for i,(r,q) in enumerate(zip(rows,old)):
        h=hashlib.sha256(r['completion'].encode()).hexdigest();assert h==q['input_sha256'] and i==q['index']
        steps=canonical_view(r['completion'],True)['steps'] or ['']
        assert steps==q['native']['steps']
        for v in variants: encoded[v].append(encode(tokenizer,r['question'],steps,v))
        written.append(dict(index=i,input_sha256=h,empty=not r['completion'].strip(),variants={}))
    costs={};forward_start=time.perf_counter()
    with torch.inference_mode():
        for v in variants:
            torch.cuda.synchronize();t=time.perf_counter();padded=0
            for off in range(0,288,1):
                sub=encoded[v][off:off+1]
                inp=tokenizer.pad([dict(input_ids=x['input_ids']) for x in sub],padding=True,return_tensors='pt').to('cuda:0')
                assert inp.attention_mask.all() and len(sub)==1
                logits=model(input_ids=inp.input_ids,use_cache=False).logits
                assert logits.shape[-1]==2 and torch.isfinite(logits).all();padded+=inp.input_ids.numel()
                for j,e in enumerate(sub):
                    selected=logits[j,e['positions']];probs=selected.float().softmax(-1)
                    signed=probs[:,1]-probs[:,0]
                    torch.testing.assert_close(signed,2*probs[:,1]-1,atol=2e-7,rtol=1e-6)
                    # Separate tensor-concatenation oracle follows model-card syntax.
                    r=rows[off+j];prompt=r['question'] if v=='native_raw' else TEMPLATE.format(r['question'])
                    direct=torch.tensor(tokenizer.encode(prompt,add_special_tokens=False)).unsqueeze(0)
                    positions=[]
                    for step in e['steps']:
                        direct=torch.cat([direct,torch.tensor(tokenizer.encode(step,add_special_tokens=False)).long().unsqueeze(0),
                                          torch.tensor(tokenizer.encode('\n',add_special_tokens=False)).unsqueeze(0)],dim=-1)
                        positions.append(direct.size(-1)-1)
                    assert direct[0].tolist()==e['input_ids'] and positions==e['positions']
                    result={**e,'step_scores':probs[:,1].cpu().tolist(),'signed_scores':signed.cpu().tolist(),
                        'step_logits':selected.float().cpu().tolist(),
                        'model_card_bfloat16_signed_scores':(selected.softmax(-1)[:,1]-selected.softmax(-1)[:,0]).float().cpu().tolist(),
                        'independent_serialization_equal':True}
                    if written[off+j]['empty']:
                        dl=model(direct.to('cuda:0'),use_cache=False).logits[0,positions]
                        dp=dl.float().softmax(-1)[:,1].cpu().tolist()
                        error=max(abs(x-y) for x,y in zip(dp,result['step_scores']))
                        result.update(direct_single_probabilities=dp,direct_adapter_max_abs_error=error)
                        assert error==0, 'Identical single-example input must reproduce literal forward'
                        # Reproduce the old two-example batch as a separate numerical diagnostic.
                        pair_start=((off+j)//2)*2
                        pair=encoded[v][pair_start:pair_start+2]
                        pi=tokenizer.pad([dict(input_ids=x['input_ids']) for x in pair],padding=True,return_tensors='pt').to('cuda:0')
                        which=(off+j)-pair_start
                        assert pi.input_ids[which,:len(e['input_ids'])].tolist()==e['input_ids']
                        assert int(pi.attention_mask[which].sum())==len(e['input_ids'])
                        pl=model(**pi,use_cache=False).logits[which,e['positions']].float().softmax(-1)[:,1].cpu().tolist()
                        result.update(batch2_probabilities=pl,batch2_vs_single_max_abs_error=max(abs(x-y) for x,y in zip(pl,result['step_scores'])),
                            batch2_sign_crossing=any((x>.5)!=(y>.5) for x,y in zip(pl,result['step_scores'])),batch2_input_and_padding_verified=True)
                        del pi
                    written[off+j]['variants'][v]=result
                del inp,logits
                with (a.output/'incremental_scores.jsonl').open('a') as partial:
                    partial.write(json.dumps(dict(index=off,variant=v,result=written[off]['variants'][v]),ensure_ascii=False)+'\n')
                if (off+1)%32==0:
                    atomic(a.output/'status.json',dict(status='scoring',pid=os.getpid(),variant=v,completed=off+1,total=288,optimizer_steps=0))
                    print(f'{v} {off+1}/288 saved responses, no updates',flush=True)
            torch.cuda.synchronize()
            costs[v]=dict(seconds=time.perf_counter()-t,scored_responses=288,
                prm_input_tokens=sum(e['prm_input_tokens'] for e in encoded[v]),padded_tokens=padded)
    frozen();assert release_before==sha(ROOT/'experiments/pure_integration_20261022/release.json')
    with (a.output/'scored.jsonl').open('x') as f:
        for r in written:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    atomic(a.output/'result.json',dict(passed=True,rows=288,empty_rows=9,variants=variants,costs=costs,
        elapsed_seconds=time.perf_counter()-start,forward_seconds=time.perf_counter()-forward_start,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
        input_sha256=sha(a.input),qwen_results_sha256=sha(a.qwen),model_assets_sha256=sha(a.assets),
        model_revision=assets['revision'],matched_checkpoint_tensors=checked,primary_microbatch=1,
        primary_forward='Literal model-card single-example forward without padding',
        batch2_empty_sensitivity_recorded=True,extra_single_empty_forwards_per_variant=9,extra_batch2_pair_forwards_per_variant=9,
        cost_scope='Per-variant time includes nine independent single and nine paired empty diagnostics; primary token totals exclude diagnostics',
        official_source_commit='b174e508a4469663543cfcb366be78cf199d20bf',
        identical_canonical_steps=True,all_independent_serializations_equal=True,
        empty_terminal_step='EOS-only boundary from official RL; literal model-card steps=[] has no score',
        official_rl_prompt_scope='Official prompt string with fixed canonical steps; not literal actor-token split or truncation emulation',
        new_generations=0,optimizer_steps=0,policy_model_loaded=False,model_frozen=True,training_started=False))
    atomic(a.output/'status.json',dict(status='completed',pid=os.getpid(),optimizer_steps=0,new_generations=0))

if __name__=='__main__':main()
