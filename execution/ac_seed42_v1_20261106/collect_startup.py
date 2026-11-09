"""Read-only collection and independent CPU verification of committed first updates."""
from pathlib import Path
from inspect_remote import SSH,HERE
import subprocess,json,hashlib,math,time
REMOTE=r'''
from pathlib import Path
import json,hashlib,time,psutil
r=Path('/path/to/ece285_math_rl_project');p=r/'runs/AC_seed42_V1_20261106';a=r/'audits/AC_seed42_V1_20261106'
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(f):return json.loads(f.read_text())
assert read(p/'acceptance_gate.json')['passed'],'Common two-update gate not yet completed'
files={};checks={}
for n in ['dispatch.json','started.json','common_initialization.json','acceptance_gate.json','controller.json']:
 files[n]=(p/n).read_text()
for arm in ['A','C_bounded']:
 for n in ['resume_build_2.json','optimizer_acceptance_2.json','resume_acceptance_2.json','stage_2_result.json']:
  files[arm+'/'+n]=(p/arm/n).read_text()
 cp=p/arm/'milestones/checkpoint-2';contract=read(cp/'pilot_checkpoint.json')
 checks[arm]={n:sha(cp/n)==h for n,h in contract['files_sha256'].items()};assert all(checks[arm].values())
 for n in ['pilot_checkpoint.json','completion_accounting.json','trainer_state.json']:
  files[arm+'/checkpoint-2/'+n]=(cp/n).read_text()
 with (p/arm/'rollouts.jsonl').open() as stream:prefix=[next(stream) for _ in range(64)]
 assert all(json.loads(s)['optimizer_step_before_update']<2 for s in prefix)
 files[arm+'/first2_rollouts.jsonl']=''.join(prefix)
files['admission.json']=(a/'preflight.json').read_text()
controller=read(p/'controller.json');ids={}
for kind,pid in [('controller',controller['pid']),('child',controller.get('child_pid'))]:
 if pid:
  try:
   v=psutil.Process(pid);ids[kind]={'pid':pid,'create_time':v.create_time(),'status':v.status(),'cmdline':v.cmdline()}
  except psutil.NoSuchProcess:ids[kind]={'pid':pid,'absent':True}
print(json.dumps({'files':files,'hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()},'checkpoint_file_checks':checks,'process_identities':ids,'unix':time.time()}))
'''

def main():
 result=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=REMOTE,text=True,capture_output=True)
 if result.returncode:raise RuntimeError((result.returncode,result.stderr[-2000:]))
 d=json.loads(result.stdout);folder=HERE/'startup_evidence';folder.mkdir(exist_ok=False)
 for n,s in d['files'].items():
  assert not Path(n).is_absolute() and '..' not in Path(n).parts
  assert hashlib.sha256(s.encode()).hexdigest()==d['hashes'][n]
  q=folder/n;q.parent.mkdir(exist_ok=True,parents=True);q.write_text(s)
 all_rows={};stats={};specs={}
 for arm in ['A','C_bounded']:
  rr=[json.loads(s) for s in d['files'][arm+'/first2_rollouts.jsonl'].splitlines()];all_rows[arm]=rr
  spec=json.loads(d['files'][arm+'/resume_build_2.json']);specs[arm]=spec
  assert spec['seed']==spec['settings']['seed']==spec['settings']['data_seed']==42
  assert spec['vllm_engine_seed']==0 and spec['VLLM_USE_V1']=='1'
  assert json.loads(d['files'][arm+'/resume_acceptance_2.json'])['actor_updated']
  err=0.;wrong_positive=0;nonzero=0;max_process=0.
  for k in range(0,64,8):
   group=rr[k:k+8];y=[x['outcome_reward'] for x in group];m=[x['min_score'] for x in group]
   mean=sum(y)/8;std=math.sqrt(sum((v-mean)**2 for v in y)/7);nc=sum(y)
   mc=sum(v*s for v,s in zip(y,m))/nc if nc else 0.
   for row,v,s in zip(group,y,m):
    outcome=(v-mean)/(std+1e-4);process=.2*v*(s-mc) if arm=='C_bounded' else 0.
    err=max(err,abs(row['advantage']-outcome-process));assert len(row['completion_token_ids'])==row['completion_tokens']
    assert abs(row['process_contribution']-process)<3e-7
    wrong_positive+=int(not v and row['advantage']>0);nonzero+=int(abs(row['process_contribution'])>1e-8);max_process=max(max_process,abs(row['process_contribution']))
  assert err<3e-7 and wrong_positive==0 and max_process<=.200001
  stats[arm]={'completed_updates':2,'rows':64,'tokens':sum(x['completion_tokens'] for x in rr),'wrong_positive':wrong_positive,'max_independent_advantage_error':err,'nonzero_process_rows':nonzero,'max_abs_process_contribution':max_process}
  account=json.loads(d['files'][arm+'/checkpoint-2/completion_accounting.json']);assert stats[arm]['tokens']==account['generated_completion_tokens'] and account['loss_rows']==64
 assert [x['problem_id'] for x in all_rows['A']]==[x['problem_id'] for x in all_rows['C_bounded']]
 assert specs['A']['initial_adapter_digest']==specs['C_bounded']['initial_adapter_digest']
 settings=[{k:v for k,v in s['settings'].items() if k not in ['output_dir','logging_dir','run_name']} for s in specs.values()];assert settings[0]==settings[1]
 first_batch_identical=all(a['completion_token_ids']==b['completion_token_ids'] for a,b in zip(all_rows['A'][:32],all_rows['C_bounded'][:32]))
 receipt={'passed':True,'snapshot_unix':d['unix'],'arms':stats,'same_fresh_initialization':True,'same_prompt_order':True,'same_training_settings':True,'first_generated_batch_token_ids_identical':first_batch_identical,'checkpoint_file_checks':d['checkpoint_file_checks'],'process_identities':d['process_identities'],'files_sha256':d['hashes'],'no_new_model_calls':True}
 (HERE/'startup_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({k:receipt[k] for k in ['passed','arms','same_fresh_initialization','same_prompt_order','first_generated_batch_token_ids_identical','process_identities']},indent=2))
if __name__=='__main__':main()
