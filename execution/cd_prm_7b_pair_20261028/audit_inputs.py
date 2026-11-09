"""Independent, read-only input audit; does not import the new pipeline."""
from pathlib import Path
import subprocess,json

REMOTE=r'''
from pathlib import Path
import collections,json,hashlib,os
root=Path('/path/to/ece285_math_rl_project');out=root/'runs/CD_PRM7B_pair_seed17_20261028'
os.environ['HF_HOME']=str(root/'cache/huggingface');os.environ['HF_MODULES_CACHE']=str(root/'cache/modules')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.open() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
selected=rows(out/'buffer_questions.jsonl');anchors=rows(out/'anchors.jsonl');probe=rows(out/'probe.jsonl')
forbidden=set(read(out/'forbidden.json'));assert len(selected)==len({r['problem_id'] for r in selected})==512
assert not ({r['question_sha256'] for r in selected}&forbidden)
assert len(anchors)==len({r['anchor_id'] for r in anchors})==2048
assert len({tuple(r['input_ids']) for r in anchors})==2048
assert collections.Counter(r['label'] for r in anchors)=={0:1024,1:1024}
assert not ({r['question_sha256'] for r in anchors}&forbidden)
groups=collections.defaultdict(list)
for r in probe:groups[r['group_id']].append(r)
assert len(groups)==64 and all(len(v)==8 for v in groups.values())
assert all(len({r['problem_id'] for r in v})==1 for v in groups.values())
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(root/'models/qwen25_math_prm_7b',local_files_only=True)
marker=tok.encode('<extra_0>',add_special_tokens=False);assert len(marker)==1
for a in anchors:
 messages=[{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},{'role':'user','content':a['question']},{'role':'assistant','content':'<extra_0>'.join(a['steps'])+'<extra_0>'}]
 ids=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False),add_special_tokens=False)
 assert ids==a['input_ids'];assert [i for i,t in enumerate(ids) if t==marker[0]]==a['positions']
 assert len(a['positions'])==len(a['steps']) and len(ids)<=2048
result=dict(passed=True,questions=512,anchors=2048,anchor_labels={0:1024,1:1024},primary_probe_groups=64,
 independent_official_format_anchor_encodings_checked=2048,no_new_GPU_work=True,no_policy_update=True,
 source_hashes={n:sha(out/n) for n in ['buffer_questions.jsonl','anchors.jsonl','probe.jsonl','processbench.jsonl']})
if (out/'buffer_rollouts.jsonl').exists():
 b=rows(out/'buffer_rollouts.jsonl');counts=collections.Counter(r['problem_id'] for r in b)
 assert len(b)==2048 and set(counts)=={r['problem_id'] for r in selected} and set(counts.values())=={4}
 assert all(r['completion_tokens']==len(r['completion_token_ids'])<=2048 for r in b)
 result['complete_buffer_verified']=True;result['buffer_sha256']=sha(out/'buffer_rollouts.jsonl')
else:result['complete_buffer_verified']=False
if (out/'data_ready.json').exists():
 ready=read(out/'data_ready.json');bm=read(out/'batch_manifest.json');data=read(out/'encoded_training.json')
 assert sha(out/'batch_manifest.json')==ready['batch_manifest_sha256']
 assert sha(out/'encoded_training.json')==bm['encoded_sha256']
 assert sha(out/'buffer_rollouts.jsonl')==bm['buffer_sha256']
 assert sha(out/'anchors.jsonl')==bm['anchors_sha256']
 reconstructed=0
 for key in ['fail','success']:
  for a in data[key]:
   assert bool(a['reward'])==(key=='success')
   steps=[s.strip() for s in a['completion'].replace('\r\n','\n').split('\n\n') if s.strip()]
   messages=[{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},{'role':'user','content':a['question']},{'role':'assistant','content':'<extra_0>'.join(steps)+'<extra_0>'}]
   ids=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False),add_special_tokens=False)
   assert ids==a['input_ids'] and steps==a['steps']
   assert [i for i,t in enumerate(ids) if t==marker[0]]==a['positions']
   assert len(ids)<=4096;reconstructed+=1
 assert len(bm['batches'])==bm['steps']
 for key,items in data.items():
  assert all(len(b[key])==2 and all(0<=j<len(items) for j in b[key]) for b in bm['batches'])
  assert {j for b in bm['batches'] for j in b[key]}==set(range(len(items)))
 for key,label in [('anchor_positive',1),('anchor_negative',0)]:
  assert {a['anchor_id'] for a in data[key]}=={a['anchor_id'] for a in anchors if a['label']==label}
 assert bm['input_tokens']==sum(len(data[k][j]['input_ids']) for b in bm['batches'] for k in data for j in b[k])
 result.update(independent_weak_native_encodings_checked=reconstructed,all_shared_samples_covered=True,
  paired_batches=bm['steps'],shared_counts=bm['counts'],input_tokens_per_arm=bm['input_tokens'],
  shared_manifest_sha256=sha(out/'batch_manifest.json'))
print(json.dumps(result,ensure_ascii=False))
'''
if __name__=='__main__':
    cmd=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','USER@GPU_HOST','/path/to/ece285_math_rl_project/.venv/bin/python -']
    r=subprocess.run(cmd,input=REMOTE,text=True,capture_output=True,timeout=180)
    assert r.returncode==0,r.stderr[-3000:]
    d=json.loads(r.stdout);(Path(__file__).parent/'independent_input_audit.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d,ensure_ascii=False,indent=2))
