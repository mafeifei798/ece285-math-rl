"""Exact pinned TRL buffer/loss replay; saved C scores, synthetic actor logits."""
from collections import deque,defaultdict,Counter
from types import SimpleNamespace
import importlib.util,sys,math
import torch
from audit_common import ROOT,HERE,RUN,CD,rows,read,atomic,verify_release,sha
from c_trainer import CBoundedMixin
from c_rewards import RewardLedger
from cpu_reference import bounded_correct_process

def main():
    verify_release();torch.set_num_threads(2);torch.manual_seed(17)
    output=ROOT/'audits/C_bounded_integration_20261029_cpu';output.mkdir(exist_ok=True)
    assert not (output/'receipt.json').exists(),'Inspect existing CPU receipt'
    old=ROOT/'experiments/b_fix_integration_20261021';sys.path.insert(0,str(old))
    spec=importlib.util.spec_from_file_location('old_replay_pinned_helpers',old/'replay.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    methods,kernel,_=module.exact_methods()
    sources={r['example_id']:r for name in ['probe.jsonl','illustrations.jsonl'] for r in rows(CD/name)}
    scores={r['example_id']:r for r in rows(CD/'scores_C.jsonl')}
    groups=[r for r in rows(CD/'group_advantages.jsonl') if r['arm']=='C']
    batch=[{**sources[i],'problem':sources[i]['question'],'cached_C':scores[i]} for g in groups for i in g['example_ids']]
    assert len(batch)==536 and len(groups)==67
    class Cached:
        def score(self,questions,completions):
            selected=[r for r in batch if r['cached_C']['eligible']]
            assert questions==[r['question'] for r in selected]
            assert completions==[r['completion'].replace('\r\n','\n') for r in selected]
            out=[]
            for r in selected:
                ps=r['cached_C']['step_scores'];assert len(ps)==len([s for s in r['completion'].replace('\r\n','\n').split('\n\n') if s.strip()])
                out.append(dict(step_scores=ps,min_score=min(ps),step_count=len(ps),prm_input_tokens=0))
            return out,dict(prm_scored_completions=len(out),prm_input_tokens=0,prm_padded_tokens=0,prm_forward_seconds=0.)
    ledger=RewardLedger(Cached(),read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json'))
    class Base:
        _prepare_inputs=methods['_prepare_inputs'];compute_loss=methods['compute_loss'];_compute_loss=methods['_compute_loss']
        def _generate_and_score_completions(self,inputs):
            values=ledger.score_records(completions=[r['completion'] for r in inputs],
              completion_counts=[r['completion_tokens'] for r in inputs],terminated=[r['terminated'] for r in inputs],
              answer=[r['gold'] for r in inputs],problem_id=[r['problem_id'] for r in inputs],problem=[r['question'] for r in inputs])
            assert all(a['outcome_reward']==b['cached_C']['outcome'] for a,b in zip(ledger.records,inputs))
            state=dict(torch=torch,self=self,device=torch.device('cpu'),rewards_per_func=torch.tensor(values,dtype=torch.float32).reshape(-1,1))
            exec(kernel,state);self._logs['advantages'].extend(state['advantages'].tolist())
            counts=torch.tensor([r['completion_tokens'] for r in inputs]);width=max(int(counts.max()),1)
            mask=(torch.arange(width)[None,:]<counts[:,None]).long()
            return dict(prompt_ids=torch.arange(len(inputs))[:,None],prompt_mask=torch.ones(len(inputs),1,dtype=torch.long),
              completion_ids=torch.zeros(len(inputs),width,dtype=torch.long),completion_mask=mask,
              advantages=state['advantages'],num_items_in_batch=mask.sum(),old_per_token_logps=torch.full(mask.shape,-2.),ref_per_token_logps=torch.full(mask.shape,-2.1))
        def _get_per_token_logps_and_entropies(self,model,ids,mask,width,**kwargs):
            return torch.full((len(ids),width),-2.+self.log_ratio),torch.full((len(ids),width),.7)
    cls=type('CPUCBounded',(CBoundedMixin,Base),{});t=cls()
    t.model=SimpleNamespace(training=True);t.accelerator=SimpleNamespace(num_processes=1,gather=lambda x:x)
    t.args=SimpleNamespace(steps_per_generation=len(batch),delta=None);t.num_generations=8;t.num_iterations=1
    t.scale_rewards='group';t.reward_weights=torch.ones(1);t.use_liger_loss=False;t.use_vllm=True;t.vllm_importance_sampling_correction=False
    t.top_entropy_quantile=1.;t.beta=.04;t.importance_sampling_level='token';t.loss_type='grpo';t.epsilon_low=t.epsilon_high=.2
    t.current_gradient_accumulation_steps=len(batch);t._metrics={'train':defaultdict(list)};t._logs={'advantages':deque(maxlen=len(batch))}
    t._step=0;t._buffered_inputs=None;t.state=SimpleNamespace(global_step=0);t.initialize_c_bounded(ledger);t._c_phase='CPU_saved_text_logits_fixture'
    expected=[]
    for g in groups:
        rs=[scores[i] for i in g['example_ids']]
        expected.extend(bounded_correct_process([r['outcome'] for r in rs],[r['min_score'] if r['eligible'] else 0 for r in rs])['advantage'])
    observed={};negative=[];loss_errors=[]
    for step in range(len(batch)):
        prepared=t._prepare_inputs(batch);tag=int(prepared['c_bounded_trace_id'][0]);observed[tag]=float(prepared['advantages'][0])
        for ratio in [0.,-.7,.5]:
            t.log_ratio=ratio;loss=t.compute_loss(t.model,prepared)
            now=torch.tensor(-2.+ratio);old=torch.tensor(-2.);ref=torch.tensor(-2.1)
            c=(now-old).exp();clip=c.clamp(.8,1.2);a=prepared['advantages'];kl=(ref-now).exp()-(ref-now)-1
            want=(-torch.minimum(c*a,clip*a)+.04*kl).mean()/len(batch)
            torch.testing.assert_close(loss,want,atol=2e-7,rtol=2e-6);loss_errors.append(float(abs(loss-want)))
        if step==0:
            # Corrupt real nonconstant group values; none is allowed to reach loss.
            first=sorted(t.c_expected)[:8];full=t.c_last_output
            inputs={k:(v[:8].clone() if torch.is_tensor(v) and v.ndim else v) for k,v in full.items()}
            original=inputs['advantages'].clone()
            changes={'second_normalization':(original-original.mean())/(original.std()+1e-4),
                     'rescale':original*2,'swap_advantages_only':original.flip(0)}
            for name,changed in changes.items():
                bad={**inputs,'advantages':changed}
                try:t.assert_c_loss_input(bad)
                except RuntimeError:negative.append(name)
                else:raise AssertionError('Missed mutation '+name)
            bad={k:v for k,v in inputs.items() if k!='c_bounded_trace_id'}
            try:t.assert_c_loss_input(bad)
            except RuntimeError:negative.append('missing_tags')
            else:raise AssertionError('Missing tag undetected')
            bad={**inputs,'completion_ids':inputs['completion_ids'].clone()};bad['completion_ids'][0,0]+=1
            try:t.assert_c_loss_input(bad)
            except RuntimeError:negative.append('token_tag_misalignment')
            else:raise AssertionError('Token mismatch undetected')
    max_error=max(abs(observed[i]-expected[i]) for i in range(len(batch)));assert max_error<4e-7
    assert all(observed[i]<=0 for i,r in enumerate(batch) if r['cached_C']['outcome']==0)
    assert len(negative)==5 and t.c_loss_calls==536*3
    receipt=dict(passed=True,stored_rows=536,groups=67,group_types=dict(Counter(g['kind'] for g in groups)),
      cached_C_scores=True,original_repaired_verifier_reexecuted=True,exact_pinned_TRL_buffer_and_loss=True,
      loss_calls=t.c_loss_calls,max_FP64_reference_error=max_error,max_loss_formula_error=max(loss_errors),
      wrong_positive=0,post_combination_normalizations=0,loss_input_bitwise_preserved=True,negative_tests_rejected=negative,
      models_loaded=0,optimizer_steps=0,fixture='Actor tokens/logits are synthetic; lengths and C scores are real saved data. CPU replay batch536; live audit uses32.')
    atomic(output/'receipt.json',receipt);print(receipt,flush=True)

if __name__=='__main__':main()
