"""Fetch pinned public JSON/tokenizer metadata only; no model weights or GPU."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
MODELS=['Qwen/Qwen2.5-Math-1.5B','Qwen/Qwen2.5-Math-7B-Instruct','Qwen/Qwen2.5-Math-1.5B-Instruct']
FILES=['config.json','generation_config.json','tokenizer_config.json','vocab.json','merges.txt']

def get(url):
    # macOS system curl uses the installed trust store; Python's standalone
    # certificate bundle is incomplete on this host. TLS verification stays on.
    return subprocess.run(['/usr/bin/curl','--fail','--location','--silent',
                           '--show-error','--max-time','30',url],
                          capture_output=True,check=True).stdout

def fetch_model(model):
    info=json.loads(get('https://huggingface.co/api/models/'+model))
    revision=info['sha'];target=HERE/'model_metadata'/model.split('/')[-1]/revision
    target.mkdir(parents=True,exist_ok=True)
    def fetch(name):
        url='https://huggingface.co/'+model+'/resolve/'+revision+'/'+name
        content=get(url);path=target/name
        if path.exists():assert path.read_bytes()==content
        else:path.write_bytes(content)
        return name,{'url':url,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
    with ThreadPoolExecutor(max_workers=3) as pool:files=dict(pool.map(fetch,FILES))
    cfg=json.loads((target/'config.json').read_text());gen=json.loads((target/'generation_config.json').read_text());tok=json.loads((target/'tokenizer_config.json').read_text())
    return model,{'revision':revision,'directory':str(target),'files':files,
        'config':{k:cfg.get(k) for k in ['architectures','hidden_size','vocab_size','eos_token_id','bos_token_id','torch_dtype','tie_word_embeddings']},
        'generation':{k:gen.get(k) for k in ['eos_token_id','pad_token_id','bos_token_id']},
        'tokenizer':{k:tok.get(k) for k in ['eos_token','pad_token','bos_token','chat_template']},
        'added_tokens_decoder':tok.get('added_tokens_decoder',{})}

def main():
    values={};errors={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={model:pool.submit(fetch_model,model) for model in MODELS}
        for model,job in jobs.items():
            try:
                name,data=job.result();values[name]=data
                print(json.dumps({'model':name,'revision':data['revision'],'config':data['config'],'generation':data['generation'],'tokenizer_eos':data['tokenizer']['eos_token']},ensure_ascii=False),flush=True)
            except Exception as e:errors[model]=repr(e);print(json.dumps({'model':model,'error':repr(e)}),flush=True)
    comparisons={}
    if len(values)==3:
        for model in MODELS[1:]:
            a,b=values[MODELS[0]],values[model]
            comparisons[model]={
                'same_vocab_json_bytes':a['files']['vocab.json']['sha256']==b['files']['vocab.json']['sha256'],
                'same_merges_txt_bytes':a['files']['merges.txt']['sha256']==b['files']['merges.txt']['sha256'],
                'same_added_token_mapping':a['added_tokens_decoder']==b['added_tokens_decoder'],
                'same_tokenizer_eos':a['tokenizer']['eos_token']==b['tokenizer']['eos_token'],
                'same_model_vocab_size':a['config']['vocab_size']==b['config']['vocab_size']}
    out={'models':values,'errors':errors,'comparisons_to_base_student':comparisons,'model_weights_downloaded':False,'GPU_used':False}
    (HERE/'model_metadata_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'comparisons':comparisons,'errors':errors},ensure_ascii=False))

if __name__=='__main__':main()
