"""CPU tokenizer-only read-only alignment check for all fixed 50 B records."""
import hashlib,json,os,re,sys,time
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path(sys.argv[1]).resolve();DATA=Path(sys.argv[2]).resolve();OUT=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(ROOT/'training_b'))
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
from transformers import AutoTokenizer
from prm_reward import encode_native
asset=json.loads((ROOT/'receipts/B_prm_assets.json').read_text())
tok=AutoTokenizer.from_pretrained(asset['model_path'],local_files_only=True,padding_side='right')
records=json.loads(DATA.read_text());results=[]
for r in records:
 s=r['saved'];e=encode_native(tok,r['question'],s['completion'],4096)
 assert len(e['steps'])==len(s['step_scores'])==s['step_count']
 assert e['steps']==r['steps']
 assert len(e['input_ids'])==s['prm_input_tokens']
 assert min(s['step_scores'])==s['min_score']
 assert abs(s['reward']-(s['outcome_reward']+.2*s['min_score']))<1e-12
 normalized=[x.strip() for x in s['completion'].replace('\r\n','\n').split('\n\n') if x.strip()]
 result={'review_id':r['review_id'],'source_row':r['source_row'],'steps':len(e['steps']),
   'input_tokens':len(e['input_ids']),'prefix_positions':e['positions'],
   'score_tokens':[{'index':pos,'token_id':e['input_ids'][pos],'decoded':tok.decode([e['input_ids'][pos]])} for pos in e['positions']],
   'boundary_prefix_equality_verified':True,'score_count_and_min_verified':True,'reward_formula_verified':True,
   'CRLF_blank_lines':len(re.findall(r'\r\n[ \t]*\r\n',s['completion'])),
   'steps_if_CRLF_normalized':len(normalized),'raw_bytes_sha256':hashlib.sha256(s['completion'].encode()).hexdigest()}
 results.append(result)
expected=json.loads((ROOT/'runs/B_seed17_24M/execution_source_hashes.json').read_text())
unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in expected.items())
assert unchanged and not torch.cuda.is_initialized()
report={'complete':True,'records':50,'scored_positions':sum(x['steps'] for x in results),
 'all_checks_passed':True,'cuda_initialized':False,'B_source_hashes_unchanged':unchanged,
 'scope':'Tokenizer-only audit; no model forward, training, reward redefinition or source edits',
 'prm_revision':asset['model_revision'],'selection_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
 'CRLF_affected_records':sum(x['steps_if_CRLF_normalized']!=x['steps'] for x in results),'results':results}
OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'}))
