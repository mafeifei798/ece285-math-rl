"""Bounded independent algebra/verifier checks and CPU FP32 PRM replay; no training."""
import json,os,sys,time,math
from fractions import Fraction
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT,DATA,OUT=map(lambda p:Path(p).resolve(),sys.argv[1:4])
sys.path.insert(0,str(ROOT/'training'));sys.path.insert(0,str(ROOT/'training_b'))
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
import sympy as sp
from common import atomic_json
from reward import outcome,parsed_answer
from prm_reward import encode_native,group_diagnostics
from transformers import AutoTokenizer,AutoModelForTokenClassification
records=json.loads(DATA.read_text());byid={r['review_id']:r for r in records}
checks={};x=sp.symbols('x')
value=Fraction(1)
for n in range(1,14):value*=Fraction(n*(n+2),(n+4)**2)
checks['R01_product']=str(value);assert value==Fraction(3,161840)
checks['R05_c_count']=len({9*n for n in range(112)}|{9*n+2 for n in range(111)})
checks['R08_expanded_shift']=str(sp.expand((x+2)**3-5*(x+2)+7))
checks['R08_poly_from_roots_exists']=hasattr(sp,'poly_from_roots')
N=10**12-1;div=sp.divisors(N)
checks['R12_factorization']={str(k):v for k,v in sp.factorint(N).items()}
checks['R12_true_count']=len(div)-1
checks['R12_claimed_code_actual_count']=sum(all(N%(int(d)*k)!=0 for k in range(1,12)) for d in div)
assert checks['R12_true_count']==255 and checks['R12_claimed_code_actual_count']==0
checks['R13_expected']=str(Fraction(1,2)+Fraction(3,3)-Fraction(5,6))
t=sp.symbols('t',positive=True)
T=2*t/(1-t*t)
checks['R15_tan_difference']=str(sp.factor((T/2-t)/(1+t*T/2)))
assert checks['R15_tan_difference']=='t**3'
tau=[0]*2006
for d in range(1,2006):
 for n in range(d,2006,d):tau[n]+=1
running=0;odd=0
for n in range(1,2006):running+=tau[n];odd+=running%2
checks['R17_parity_counts']={'odd':odd,'even':2005-odd,'difference':abs(odd-(2005-odd))}
assert checks['R17_parity_counts']['difference']==25
A=[2**i for i in range(9)];B=list(range(20,301,20))
checks['R20_claimed_code_actual_min']=min(abs(a-b) for a in A for b in B)
checks['R23_original_value']=(3*4-5*2+4)*(7-4)
checks['R23_claimed_code_actual_value']=8*(-6)+4*31+2*(-35)+28
checks['R25_distance_time_speed']={'distance':102,'time':sum([.5,.5,.75,.25]),'speed':102/sum([.5,.5,.75,.25])}
checks['R29_derivative_at_1']=str(sp.diff(x*(x-1)*(x-2)/8,x).subs(x,1))
missing=[n for n in range(1,201) if (n+11)//12>n//8]
checks['S13_max_missing_n_below_200']=max(missing);assert max(missing)==15
checks['S09_P_plus_Q']=6+25
checks['S14_pairs']=[[26,25],[10,7]]
assert all(a*a-b*b==51 for a,b in checks['S14_pairs'])
constant=Fraction(1,6)*Fraction(2,5)*Fraction(3,4)*Fraction(4,3)*Fraction(5,2)
checks['S15_coefficient']=str(constant);assert constant==Fraction(1,6)
M=sp.Matrix([[sp.sqrt(19),91],[1,0]])**5
num=sp.expand(x*(M[1,0]*x+M[1,1])-(M[0,0]*x+M[0,1]))
checks['S18_F5_equivalence_remainder']=str(sp.simplify(sp.rem(num,sp.Poly(x*x-sp.sqrt(19)*x-91,x).as_expr(),x)))
assert checks['S18_F5_equivalence_remainder']=='0'
checks['S18_numerator_factor']=str(sp.factor(num,extension=sp.sqrt(19)))
checks['S20_radius_squared']=str(Fraction(9)+Fraction(25,4)+Fraction(9,4))
checks['R04_outcome_percent']=outcome('\\boxed{20\\%}','20',True)
checks['R04_outcome_number']=outcome('\\boxed{20}','20',True)
checks['R04_parse_gold']=str(parsed_answer('20'))
checks['R04_parse_percent']=str(parsed_answer('20\\%'))
assert checks['R04_outcome_percent']['reward']==0 and checks['R04_outcome_number']['reward']==1
checks['S06_solve_nonsymbol_target']=str(sp.solve(sp.det(sp.Matrix([[1,4,9],[3,x,sp.Symbol('y')],[3,sp.Symbol('y'),x]])),x+sp.Symbol('y')))
atomic_json(OUT/'independent_math.json',checks)

allrows=[json.loads(s) for s in (ROOT/'runs/B_seed17_6M/rollouts.jsonl').read_text().splitlines()]
advantages=[]
for r in records:
 gi=r['source_row']//8;g=allrows[gi*8:gi*8+8]
 ys=[z['outcome_reward'] for z in g];rs=[z['reward'] for z in g];valid=[z['valid_for_process_reward'] for z in g];ms=[z['min_score'] for z in g]
 av,diag=group_diagnostics(ys,rs,valid,ms,8)
 oracle=torch.tensor(ys);oa=(oracle-oracle.mean())/(oracle.std()+1e-4)
 j=r['source_row']%8
 advantages.append({'review_id':r['review_id'],'group_index':gi,'group_type':diag[0]['outcome_type'],
 'outcomes':ys,'process_scores':ms,'mixed_rewards':rs,'mixed_advantage':float(av[j]),
 'outcome_only_advantage_on_same_group':float(oa[j]),'note':'Same stored B group; not a new A policy rollout or a causal estimate of training impact'})
atomic_json(OUT/'sample_advantages.json',advantages)
print(json.dumps({'stage':'math_and_reward_replay_complete','math_checks':checks},ensure_ascii=False),flush=True)

assets=json.loads((ROOT/'receipts/B_prm_assets.json').read_text())
tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True,padding_side='right')
started=time.monotonic()
model=AutoModelForTokenClassification.from_pretrained(assets['model_path'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').eval()
model.requires_grad_(False)
@torch.inference_mode()
def forward(e):
 ids=torch.tensor([e['input_ids']]);logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits.float()
 return logits.softmax(-1)[0,e['positions'],1].tolist()
results=[]
for identifier in ['R02','R20','S08','S15','S19']:
 r=byid[identifier];e=encode_native(tok,r['question'],r['saved']['completion'],4096)
 p=forward(e);saved=r['saved']['step_scores'];k=min(range(len(saved)),key=lambda j:saved[j])
 prefix='\n\n'.join([r['question']]+e['steps'][:k+1])+'\n\n'
 ids=tok(prefix,add_special_tokens=True)['input_ids']
 pp=forward({'input_ids':ids,'positions':[len(ids)-1]})[0]
 assert abs(pp-p[k])<2e-5
 result={'review_id':identifier,'tokens':len(e['input_ids']),'saved_BF16_step_scores':saved,'CPU_FP32_step_scores':p,
 'max_score_difference':max(abs(a-b) for a,b in zip(saved,p)),
 'saved_min':min(saved),'replayed_min':min(p),'min_score_difference':abs(min(saved)-min(p)),
 'checked_min_step':k+1,'causal_prefix_vs_full_difference':abs(pp-p[k]),
 'same_high_low_conclusion':(min(saved)>=.5)==(min(p)>=.5)}
 results.append(result);atomic_json(OUT/'forward_replay.json',{'complete':False,'records':results})
 print(json.dumps({'stage':'forward_case_complete','review_id':identifier,'tokens':result['tokens'],'min_difference':result['min_score_difference'],'elapsed_seconds':time.monotonic()-started}),flush=True)
assert all(r['same_high_low_conclusion'] for r in results) and not torch.cuda.is_initialized()
atomic_json(OUT/'forward_replay.json',{'complete':True,'device':'CPU','dtype':'FP32','cuda_initialized':False,'records':results,
 'elapsed_seconds':time.monotonic()-started,'scope':'5 fixed records; deterministic prefix indexing and numeric replay, not retraining or 50-case exact BF16 bitwise reproduction'})
