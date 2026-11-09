"""Independent CPU recount, paired statistics and signal audit of frozen artifacts."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, math
import numpy as np

HERE=Path(__file__).resolve().parent
E=HERE/'completed_evidence'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
arms=['A','C_mix','C_bounded']
c=read(E/'comparison.json');assert c['passed'] and c['global_step']==288
schedule=read(E/'audit/preflight.json')['first_epoch_question_groups']
preds={};signals={};total=0
for step in [144,288]:
 contract=read(E/f'evaluation/{step}/contract.json')
 summary=read(E/f'evaluation/{step}/summary.json')
 assert summary['contract_sha256']==sha(E/f'evaluation/{step}/contract.json')
 assert contract['engine']=='V1' and contract['cap']==2048 and contract['temperature']==0 and contract['seed']==285
 settings=[]
 for arm in arms:
  settings.append({k:v for k,v in read(E/f'{arm}/resume_build_{step}.json')['settings'].items() if k not in ['output_dir','logging_dir','run_name']})
  r=read(E/f'{arm}/stage_{step}_result.json')
  assert r['passed'] and r['global_step']==step and r['PRM_frozen']
  assert r['adapter_sha256']==contract['adapters'][arm]['sha256']
  assert read(E/f'{arm}/resume_acceptance_{step}.json')['loss_input_exact']
  for ds,s in summary['summary'][arm].items():
   p=E/f'evaluation/{step}/{arm}/{ds}.jsonl'
   assert sha(p)==s['predictions_sha256']
   rr=rows(p);assert len(rr)==len({r['problem_id'] for r in rr})==s['n']
   assert sum(r['reward'] for r in rr)==s['correct']
   assert sum(r['generated_tokens'] for r in rr)==s['generated_tokens']
   assert all(len(r['completion_token_ids'])==r['generated_tokens']<=2048 and r['adapter_sha256']==contract['adapters'][arm]['sha256'] for r in rr)
   preds[(step,arm,ds)]=rr;total+=len(rr)
 assert all(s==settings[0] for s in settings)
 for ds in summary['summary']['A']:
  keys=[[(r['problem_id'],r['question_sha256'],r['gold']) for r in preds[(step,a,ds)]] for a in arms]
  assert keys[0]==keys[1]==keys[2]

for arm in arms:
 rr=rows(E/f'{arm}/compact_signals.jsonl');assert len(rr)==288*32
 assert sum(r['completion_tokens'] for r in rr)==c['accounting'][arm]['generated_completion_tokens']
 strata=defaultdict(lambda:Counter())
 for i in range(0,len(rr),8):
  group=rr[i:i+8];k=i//32
  assert all(r['optimizer_step_before_update']==k for r in group)
  assert len({r['problem_id'] for r in group})==1
  assert group[0]['problem_id']==schedule[k][(i%32)//8]
  correct=sum(r['outcome_reward'] for r in group)
  kind='all_wrong' if correct==0 else 'all_correct' if correct==8 else 'mixed'
  s=strata[kind];s['groups']+=1;s['rows']+=8
  for r in group:
   s['wrong']+=not r['outcome_reward'];s['wrong_positive']+=not r['outcome_reward'] and r['advantage']>0
   s['process_abs_sum']+=abs(r['process_contribution'])
   if arm=='C_bounded':
    assert abs(r['process_contribution'])<=.200001
    if not r['outcome_reward']:assert r['advantage']<=0 and abs(r['process_contribution'])<1e-7
 for k in range(288):assert [r['problem_id'] for r in rr[k*32:(k+1)*32:8]]==schedule[k]
 checks=rows(E/f'{arm}/generation_checks.jsonl')
 assert len(checks)==288 and all(r['second_normalizations']==0 for r in checks)
 assert sum(v['wrong_positive'] for v in strata.values())==c['accounting'][arm]['loss_wrong_positive']
 resources=rows(E/f'{arm}/resource_samples.jsonl')
 signals[arm]={'strata':dict(strata),'maximum_absolute_process_adjustment':max(abs(r['process_contribution']) for r in rr),
  'max_observed_process_GPU_GiB_since_resume':max(r.get('own_total_MiB',0) for r in resources)/1024,
  'common144_to288_training_hours':read(E/f'{arm}/stage_288_result.json')['seconds']/3600}

paired={};categories={}
for ds in ['math500','gsm8k']:
 n=len(preds[(288,'A',ds)]);indices=np.random.default_rng(285).integers(0,n,size=(10000,n),dtype=np.int32)
 paired[ds]={};categories[ds]={}
 for lhs,rhs in [('C_mix','A'),('C_bounded','A'),('C_bounded','C_mix')]:
  left=preds[(288,lhs,ds)];right=preds[(288,rhs,ds)]
  d=np.array([l['reward']-r['reward'] for l,r in zip(left,right)])
  ci=np.quantile(d[indices].mean(1)*100,[.025,.975]).tolist()
  wins=int((d>0).sum());losses=int((d<0).sum());nd=wins+losses
  p=min(1.,2*sum(math.comb(nd,i) for i in range(min(wins,losses)+1))/2**nd) if nd else 1.
  pair=lhs+' minus '+rhs
  v={'delta_pp':float(d.mean()*100),'wins':wins,'losses':losses,'CI95_pp':ci,'exact_mcnemar_p':p}
  expected=c['paired'][ds][pair]
  assert v['wins']==expected['wins'] and v['losses']==expected['losses']
  assert np.allclose(ci,expected['paired_question_bootstrap95_pp'])
  paired[ds][pair]=v
  cats={}
  for event in ['wins','losses']:
   wrong=[r if event=='wins' else l for l,r,x in zip(left,right,d) if (x>0 if event=='wins' else x<0)]
   cats[event+'_incorrect_side_status']=dict(Counter(r['status'] for r in wrong))
  both_stop=[float(x) for l,r,x in zip(left,right,d) if l['finish_reason']=='stop' and r['finish_reason']=='stop']
  cats['both_finish_normally']={'n':len(both_stop),'net_correct_difference':sum(both_stop),'note':'Post-hoc conditional description; not an unbiased treatment-effect estimate.'}
  categories[ds][pair]=cats

result={'passed':True,'prediction_rows_recounted':total,'training_signal_rows_checked':3*288*32,
 'same_question_order_and_updates':True,'same_eval_protocol':True,'prediction_hashes_and_recorded_adapter_hashes_verified':True,
 'remote_frozen_verifier_regraded_final_rows':c['acceptance']['regraded'],
 'local_scope':'Recount, signal audits, receipt/hash verification and independent paired calculations. Verifier regrading is the separate saved remote CPU audit; no new mathematical labels.',
 'signals':signals,'paired':paired,'error_status_decomposition':categories,'new_training':False,'new_generation':False,
 'limitations':c['limitations']+['Question-level intervals are unadjusted for multiple comparisons.','Known C PRM adaptation anchors have six conflicting-label input groups; kept fixed for this comparison.','No independent proof that answer-correct reasoning is valid; no cause attribution from advantage sign constraints alone.']}
(HERE/'independent_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':True,'prediction_rows':total,'training_rows':3*288*32,'signals':signals,'paired':paired,'error_categories':categories},ensure_ascii=False,indent=2))
