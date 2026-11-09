"""Unwrapped vLLM replay of a disposable merged model. No training."""
import os,json,time,hashlib
from pathlib import Path
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
os.environ['VLLM_USE_V1']='1'
import torch
from signals import ROOT
from no_update import summarise,sha,dump
from vllm import LLM,SamplingParams

def main():
    out=ROOT/'audits/prefix_protection_20261107'
    assert not (out/'no_update_result.json').exists()
    data=json.loads((out/'fixed_probes.json').read_text());result=json.loads((out/'native_no_update_result.json').read_text())
    assets=json.loads((ROOT/'receipts/assets.json').read_text());started=time.time()
    llm=LLM(model=data['merged_model'],tokenizer=assets['model_path'],dtype='bfloat16',gpu_memory_utilization=.18,
            max_model_len=4096,max_num_seqs=8,max_num_batched_tokens=4096,enforce_eager=True,
            enable_prefix_caching=False,seed=285,distributed_executor_backend='uni')
    pp,cc=data['prompt_ids'],data['completion_ids']
    generated=llm.generate([{'prompt_token_ids':p+c} for p,c in zip(pp,cc)],SamplingParams(temperature=0.,max_tokens=1,prompt_logprobs=1),use_tqdm=False)
    values=[torch.tensor([o.prompt_logprobs[len(p)+j][token].logprob for j,token in enumerate(c)]) for o,p,c in zip(generated,pp,cc)]
    xs=llm.generate([{'prompt_token_ids':p} for p in pp[::2]],SamplingParams(temperature=0.,max_tokens=64),use_tqdm=False)
    probes=[{'token_ids':list(o.outputs[0].token_ids),'text':o.outputs[0].text} for o in xs]
    result.update(HF_native_vs_vLLM_merged=summarise([torch.tensor(v) for v in data['native_hf']],values),
                  vLLM_native_vs_vLLM_merged=summarise([torch.tensor(v) for v in data['native_vllm']],values),
                  merged_probe=probes,probe_ids_equal=[a['token_ids']==b['token_ids'] for a,b in zip(result['native_probe'],probes)],
                  merged_probe_seconds=time.time()-started,remaining=None,
                  disposable_merged_model_sha256=sha(Path(data['merged_model'])/'model.safetensors'))
    assert all(sha(Path(p))==v for p,v in result['protected_files'].items())
    dump('no_update_result.json',result)
    print(json.dumps({k:v for k,v in result.items() if 'probe' not in k or k.endswith('seconds')},indent=2),flush=True)

if __name__=='__main__':main()
