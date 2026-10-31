"""CPU-only official PAPO parity, legacy equivalence and pinned TRL loss replay."""
import argparse,ast,hashlib,json,sys,math
from pathlib import Path
from collections import defaultdict,deque
from types import SimpleNamespace
from typing import Any,Union,Optional,Sequence
import numpy as np
import torch
from signals import SignalConfig,construct_signal
from diagnostics import summarize_signal
from trainer_adapter import SignalAuditMixin,TRL_SHA,UTILS_SHA
from papo_reference import compute_grpo_dual_objective_advantage

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).open() if x.strip()]
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def pinned_methods():
    vendor=ROOT/'experiments/b_baseline_corrected_20261021/vendor'
    source=vendor/'grpo_trainer_0_23_1.py';utils=vendor/'trl_utils_0_23_1.py'
    assert sha(source)==TRL_SHA and sha(utils)==UTILS_SHA
    env=dict(torch=torch,Any=Any,Union=Union,Optional=Optional,Sequence=Sequence)
    names={'split_tensor_dict','shuffle_sequence_dict','nanmin','nanmax','split_pixel_values_by_grid','unsplit_pixel_values_by_grid'}
    for node in ast.parse(utils.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'pinned_utils','exec'),env)
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GRPOTrainer')
    for n in cls.body:
        if isinstance(n,ast.FunctionDef) and n.name in ['_prepare_inputs','compute_loss','_compute_loss']:
            n.decorator_list=[]
            exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),'pinned_TRL_'+n.name,'exec'),env)
    gen=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_and_score_completions')
    start=next(i for i,n in enumerate(gen.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rewards' for t in n.targets) and 'nansum' in ast.unparse(n))
    end=next(i for i in range(start,len(gen.body)) if isinstance(gen.body[i],ast.If) and ast.unparse(gen.body[i].test)=="self.scale_rewards != 'none'")
    kernel=compile(ast.fix_missing_locations(ast.Module(body=gen.body[start:end+1],type_ignores=[])),'actual_TRL_reward_kernel','exec')
    return env,kernel

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/'receipt.json').exists(),'Completed acceptance must not be overwritten'
    torch.set_num_threads(2);torch.manual_seed(17)
    artifact=ROOT/'execution/cd_prm_7b_pair_20261028/artifacts'
    if not artifact.exists():artifact=ROOT/'runs/CD_PRM7B_pair_seed17_20261028'
    protected=[*sorted((ROOT/'experiments/c_bounded_pilot_20261029_v2').glob('*.*')),artifact/'group_advantages.jsonl']
    before={str(f):sha(f) for f in protected if f.is_file()}
    source=rows(protected[-1]);groups=[g for g in source if g['arm']=='C'];assert len(groups)==67
    y=[v for g in groups for v in g['outcomes']];m=[v for g in groups for v in g['min_scores']]
    saved={r['example_id']:r for name in ['probe.jsonl','illustrations.jsonl'] for r in rows(artifact/name)}
    cached={r['example_id']:r for r in rows(artifact/'scores_C.jsonl')}
    batch=[]
    for j,g in enumerate(groups):
        for ident,yy,mm in zip(g['example_ids'],g['outcomes'],g['min_scores']):
            r=saved[ident];s=cached[ident]
            batch.append(dict(r,source_problem_id=r['problem_id'],problem_id=f'group-{j}',outcome_reward=yy,min_score=mm,
                valid_for_process_reward=s['eligible'],step_count=len(s['step_scores'])))
    # Official function body, not a formula transcription, executes on CPU.
    yy=torch.tensor(y,dtype=torch.float32);mask=torch.ones(len(y),3)
    rewards=torch.zeros_like(mask);rewards[:,-1]=yy
    ids=np.repeat(np.arange(67),8);papo_errors=[]
    for coefficient in [.2,1.]:
        actual=construct_signal(SignalConfig('papo',coefficient=coefficient,outcome_epsilon=1e-6),y,m)
        official,returns,_=compute_grpo_dual_objective_advantage(rewards,mask,ids,np.array(m),lambda_process=coefficient)
        error=float((actual['advantages']-official[:,0]).abs().max());papo_errors.append(error)
        torch.testing.assert_close(actual['advantages'],official[:,0],atol=3e-7,rtol=2e-6)
        assert torch.equal(official,returns)
    controlled=construct_signal(SignalConfig('papo'),y,m)
    native=construct_signal(SignalConfig('papo',outcome_epsilon=1e-6),y,m)
    torch.testing.assert_close(controlled['process_component'],native['process_component'],atol=0,rtol=0)
    torch.testing.assert_close(controlled['advantages']-native['advantages'],controlled['outcome_component']-native['outcome_component'],atol=3e-7,rtol=2e-6)
    # Explicit edge cases: no PRM requested is valid for A, and <=1 correct
    # response has no within-correct process preference.
    edge=0
    for count in range(9):
        sy=[1]*count+[0]*(8-count);sm=[.5+i*.0000001 for i in range(8)]
        for mode in ['outcome','mixed','papo','bounded']:
            z=construct_signal(SignalConfig(mode),sy,None if mode=='outcome' else sm,
                process_scored=[False]*8 if mode=='outcome' else [True]*8)
            assert torch.isfinite(z['advantages']).all();edge+=1
            if mode!='mixed':assert not (z['advantages'][torch.tensor(sy)==0]>0).any()
            if mode in ['papo','bounded'] and count<=1:assert z['process_component'].abs().sum()<1e-7
    input_rejections=[]
    for name,call in [
        ('unknown_mode',lambda:SignalConfig('unknown')),
        ('negative_coefficient',lambda:SignalConfig('papo',coefficient=-1)),
        ('nonfinite_epsilon',lambda:SignalConfig('papo',outcome_epsilon=float('nan'))),
        ('nonbinary_outcome',lambda:construct_signal(SignalConfig('outcome'),[.5]+[0]*7)),
        ('nonfinite_outcome',lambda:construct_signal(SignalConfig('outcome'),[float('nan')]+[0]*7)),
        ('probability_out_of_range',lambda:construct_signal(SignalConfig('papo'),[0]*8,[1.1]*8)),
        ('unscored_correct_response',lambda:construct_signal(SignalConfig('bounded'),[1]+[0]*7,[0]*8,process_scored=[False]*8)),
        ('invalid_correct_answer',lambda:construct_signal(SignalConfig('outcome'),[1]+[0]*7,valid_output=[False]*8)),
        ('mixed_question_group',lambda:construct_signal(SignalConfig('outcome'),[0]*8,group_ids=list(range(8)))),
        ('changed_reward',lambda:construct_signal(SignalConfig('outcome'),[0]*8,callback_rewards=[1]*8))]:
        try:call()
        except (ValueError,AssertionError):input_rejections.append(name)
        else:raise AssertionError('Invalid input accepted: '+name)
    # Compare the existing bounded implementation without modifying/importing
    # its trainer or launching any model.
    import importlib.util
    spec=importlib.util.spec_from_file_location('prior_bounded',ROOT/'experiments/c_bounded_integration_20261029_v2/advantages.py')
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    bound=construct_signal(SignalConfig('bounded'),y,m)
    old=prior.compute_advantages(y,m)
    torch.testing.assert_close(bound['advantages'],old['advantages'],atol=0,rtol=0)
    methods,kernel=pinned_methods();loss_errors=[];negative=[];metric_results={};native_retained={}
    total_calls=0
    for mode in ['outcome','mixed','papo','bounded']:
        ledger=SimpleNamespace(records=[])
        class Base:
            _prepare_inputs=methods['_prepare_inputs'];compute_loss=methods['compute_loss'];_compute_loss=methods['_compute_loss']
            def _generate_and_score_completions(self,inputs):
                ledger.records=[dict(r,min_score=0. if mode=='outcome' else r['min_score'],
                    valid_for_process_reward=False if mode=='outcome' else r['valid_for_process_reward'],
                    reward=r['outcome_reward'] if mode=='outcome' else r['outcome_reward']+.2*r['min_score']) for r in inputs]
                state=dict(torch=torch,self=self,device=torch.device('cpu'),rewards_per_func=torch.tensor([r['reward'] for r in ledger.records],dtype=torch.float32).reshape(-1,1))
                exec(kernel,state);adv=state['advantages'];self.parent_advantage=adv
                self._logs['advantages'].extend(adv.tolist())
                cmask=(torch.arange(13)[None,:]<(torch.arange(len(inputs))%11+2)[:,None]).long()
                return dict(prompt_ids=torch.arange(len(inputs))[:,None],prompt_mask=torch.ones(len(inputs),1,dtype=torch.long),
                    completion_ids=torch.zeros(len(inputs),13,dtype=torch.long),completion_mask=cmask,advantages=adv,
                    num_items_in_batch=cmask.sum(),old_per_token_logps=torch.full(cmask.shape,-2.),ref_per_token_logps=torch.full(cmask.shape,-2.1))
            def _get_per_token_logps_and_entropies(self,model,ids,mask,width,**kw):
                return torch.full((len(ids),width),-2.+self.log_ratio),torch.full((len(ids),width),.7)
        cls=type('CPU_signal_'+mode,(SignalAuditMixin,Base),{});t=cls()
        t.model=SimpleNamespace(training=True);t.accelerator=SimpleNamespace(num_processes=1,gather=lambda x:x)
        t.args=SimpleNamespace(steps_per_generation=len(batch),delta=None);t.num_generations=8;t.num_iterations=1
        t.scale_rewards='group';t.reward_weights=torch.ones(1);t.use_liger_loss=False;t.use_vllm=True;t.vllm_importance_sampling_correction=False
        t.top_entropy_quantile=1.;t.beta=.04;t.importance_sampling_level='token';t.loss_type='grpo';t.epsilon_low=t.epsilon_high=.2
        t.current_gradient_accumulation_steps=len(batch);t._metrics={'train':defaultdict(list)};t._logs={'advantages':deque(maxlen=len(batch))}
        t._step=0;t._buffered_inputs=None;t.state=SimpleNamespace(global_step=0);t.initialize_signal_audit(SignalConfig(mode),ledger)
        for i in range(len(batch)):
            prepared=t._prepare_inputs(batch)
            for ratio in [0.,-.7,.5]:
                t.log_ratio=ratio;loss=t.compute_loss(t.model,prepared)
                now=torch.tensor(-2.+ratio);old_log=torch.tensor(-2.);ref=torch.tensor(-2.1)
                r=(now-old_log).exp();clipped=r.clamp(.8,1.2);a=prepared['advantages'];kl=(ref-now).exp()-(ref-now)-1
                expected=(-torch.minimum(r*a,clipped*a)+.04*kl).mean()/len(batch)
                torch.testing.assert_close(loss,expected,atol=2e-7,rtol=2e-6);loss_errors.append(float(abs(loss-expected)))
            if i==0:
                full=t.signal_last_output;probe={k:v[:8].clone() if torch.is_tensor(v) and v.ndim else v for k,v in full.items()}
                original=probe['advantages'];assert original.std()>0
                corrupt={'second_normalization':(original-original.mean())/(original.std()+1e-4),
                    'rescale':original*2,'wrong_advantage_order':original.flip(0)}
                for name,val in corrupt.items():
                    try:t.assert_signal_input({**probe,'advantages':val})
                    except RuntimeError:negative.append(mode+':'+name)
                    else:raise AssertionError('Guard accepted '+mode+':'+name)
                for name,bad in [('missing_provenance',{k:v for k,v in probe.items() if k!='signal_trace_id'}),
                    ('token_mismatch',{**probe,'completion_ids':probe['completion_ids']+1})]:
                    try:t.assert_signal_input(bad)
                    except RuntimeError:negative.append(mode+':'+name)
                    else:raise AssertionError('Guard accepted '+name)
        assert t.signal_loss_calls==536*3
        if mode in ['outcome','mixed']:assert t.signal_last_output['advantages'] is t.parent_advantage
        native_retained[mode]=t.signal_last_output['advantages'] is t.parent_advantage
        metric_results[mode]=t.signal_metrics[0];total_calls+=t.signal_loss_calls
    assert before=={str(f):sha(f) for f in protected if f.is_file()}
    assert not torch.cuda.is_initialized(),'CPU acceptance unexpectedly initialized CUDA'
    receipt=dict(passed=True,CPU_only=True,CUDA_initialized=False,groups=67,stored_rows=536,official_PAPO_function_executed=True,
        official_PAPO_commit='7d260f0a7a6e23d289e32bbc96f80408dddfa1d5',max_official_PAPO_error=max(papo_errors),
        legacy_bounded_bitwise_identical=True,native_baseline_tensor_identity_preserved=native_retained,
        exact_pinned_TRL_buffer_and_loss=True,loss_calls=total_calls,max_loss_formula_error=max(loss_errors),
        edge_cases_checked=edge,invalid_inputs_rejected=input_rejections,corrupted_inputs_rejected=negative,current_pilot_sources_preserved=True,
        metrics=metric_results,models_loaded=0,model_generation_calls=0,optimizer_steps=0,new_training_dispatch=False,
        fixture='Frozen real C probabilities/outcomes; synthetic actor token IDs/masks/logits. No real-model GPU or resume acceptance claimed.')
    save(out/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='metrics'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
