"""One live GRPO batch plus saved-text real-model loss replay, zero updates."""
import os,time,fcntl,traceback,copy,math,json
from pathlib import Path
from contextlib import contextmanager
import torch
from audit_common import ROOT,HERE,RUN,CD,read,rows,sha,atomic,event,verify_release,guard_root,protect,parameter_hash
from build import build_no_update

def adapter_hash(model):
    import hashlib
    from peft import get_peft_model_state_dict
    h=hashlib.sha256()
    for name,value in sorted(get_peft_model_state_dict(model).items()):
        t=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def slice_rows(output,start,end):
    return {k:(v[start:end] if torch.is_tensor(v) and v.ndim else v) for k,v in output.items()}

def main():
    assert os.sys.argv[1:]==['--authorized-no-update']
    guard_root();release=verify_release();assert not RUN.exists(),'Inspect existing PID and receipt; no duplicate dispatch'
    cpu=read(ROOT/'audits/C_bounded_integration_20261029_cpu_v2/receipt.json');assert cpu['passed']
    RUN.mkdir();lock=(RUN/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();stages={};trainer=None
    @contextmanager
    def stage(name):
        event(name);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.time()
        yield
        torch.cuda.synchronize();stages[name]=dict(seconds=time.time()-t,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30);atomic(RUN/'stages.json',stages)
    try:
        event('preservation_before');before=protect();atomic(RUN/'originals_before.json',before)
        lookup={r['problem_id']:r for r in rows(ROOT/'data/prepared/train.jsonl')}
        historical=rows(ROOT/'runs/B_corrected_seed17_6M_20261022/rollouts.jsonl')[:32]
        ids=[historical[i]['problem_id'] for i in range(0,32,8)];assert len(set(ids))==4
        questions=[lookup[i] for i in ids];expanded=[q for q in questions for _ in range(8)]
        atomic(RUN/'audit_contract.json',dict(formal_RL=False,PRM_adaptation=False,optimizer_steps=0,
            questions=ids,question_selection='same first4 as historical corrected A/B initialization, fixed before sampling',
            live_group_size=8,live_rows=32,exact_emitted_token_ids=True,actor_checkpoint='shared original1.5B base, fresh matched LoRA',
            saved_phase='Retokenized saved real response texts, not exact historical actor token IDs; integration-only',source_release_sha256=release))
        with stage('build_actor_vllm_and_frozen_C'):
            trainer,tok,ledger,scorer=build_no_update(RUN,questions)
        expected_adapter=read(ROOT/'runs/A_corrected_seed17_6M_20261022/run_config.json')['actor_adapter_initial_sha256']
        assert adapter_hash(trainer.model)==expected_adapter
        with stage('parameter_hashes_before'):
            actor_before=parameter_hash(trainer.model);prm_before=parameter_hash(scorer.model)
            atomic(RUN/'parameter_hashes_before.json',dict(actor=actor_before,C=prm_before))
        live_prepared=[]
        with stage('reuse_exact_live_tokens_rescore_and_TRL_buffer'):
            from types import MethodType
            origin=ROOT/'runs/C_bounded_no_update_20261029'
            assert read(origin/'failure.json')['traceback'].find('max(diffs')>=0
            assert read(origin/'independent_version.json')['source_release_sha256']=='579d4d0217c5224c1614adf6de5810b1d054bcda81099e638854831b29e625f5'
            prior=rows(origin/'live_rollouts_exact_tokens.jsonl');assert len(prior)==32
            full={k:(v.to(trainer.accelerator.device) if torch.is_tensor(v) else v)
                  for k,v in torch.load(origin/'live_signals.pt',map_location='cpu',weights_only=True).items()}
            assert [r['problem_id'] for r in prior]==[r['problem_id'] for r in expanded]
            token_lists=[r['completion_token_ids'] for r in prior]
            # Execute actual TRL reward callbacks on the existing exact samples.
            reward_values=trainer._calculate_rewards(expanded,[r['prompt_text'] for r in prior],
                [r['completion'] for r in prior],token_lists)
            mixed=(reward_values*trainer.reward_weights.to(reward_values.device)[None,:]).nansum(1).reshape(-1,8)
            full['advantages']=((mixed-mixed.mean(1,keepdim=True))/(mixed.std(1,keepdim=True)+1e-4)).flatten()
            trainer._logs['advantages'].extend(full['advantages'].cpu().tolist())
            with torch.no_grad(),trainer.accelerator.unwrap_model(trainer.model).disable_adapter():
                ref,_=trainer._get_per_token_logps_and_entropies(trainer.model,
                    torch.cat([full['prompt_ids'],full['completion_ids']],1),
                    torch.cat([full['prompt_mask'],full['completion_mask']],1),full['completion_ids'].size(1),batch_size=1)
            full['ref_per_token_logps']=ref
            original_generate=trainer._generate_and_score_completions
            def use_existing_exact_batch(self,inputs):
                assert [r['problem_id'] for r in inputs]==[r['problem_id'] for r in expanded]
                return self.attach_c_signal(full,ledger.records,phase='cached_exact_live_token_replay')
            trainer._generate_and_score_completions=MethodType(use_existing_exact_batch,trainer)
            try:
                first=trainer._prepare_inputs(expanded)
                live_prepared=[first]+[trainer._prepare_inputs(expanded) for _ in range(31)]
            finally:
                trainer._generate_and_score_completions=original_generate
            live_records=copy.deepcopy(ledger.records)
            for i,r in enumerate(live_records):
                exact=full['completion_ids'][i][full['completion_mask'][i].bool()].cpu().tolist()
                assert exact==r['completion_token_ids']==token_lists[i]
                assert tok.decode(exact,skip_special_tokens=True)==r['completion']
                pids=full['prompt_ids'][i][full['prompt_mask'][i].bool()].cpu().tolist()
                assert pids==prior[i]['exact_prompt_ids']
                r.update(exact_prompt_ids=pids,advantage=float(full['advantages'][i]),reused_generation_from=str(origin))
            with (RUN/'live_rollouts_exact_tokens.jsonl').open('w') as f:
                for r in live_records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            torch.save({k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in full.items()},RUN/'live_signals.pt')
            assert len({int(x['c_bounded_trace_id'][0]) for x in live_prepared})==32
            for x in live_prepared:trainer.assert_c_loss_input(x)
            atomic(RUN/'generation_reuse.json',dict(new_generation_calls=0,rows=32,original_token_IDs_retained=True,
                source=str(origin),source_tokens_sha256=sha(origin/'live_rollouts_exact_tokens.jsonl'),source_signals_sha256=sha(origin/'live_signals.pt'),
                reference_logps_recomputed=True,C_scores_recomputed_with_TF32_disabled=True,
                limitation='Full fresh vLLM generation tested in initial attempt; this attempt replays those exact actual token IDs through actual TRL reward callbacks, buffer and loss.'))
        def backward_phase(inputs_list):
            trainer.model.zero_grad(set_to_none=True);start=len(trainer.c_loss_trace);loss_errors=[]
            for inputs in inputs_list:
                loss=trainer.compute_loss(trainer.model,inputs);assert torch.isfinite(loss)
                # Initial zero LoRA has zero reference KL and ratio1. Check the
                # actual pinned model loss, not a substitute implementation.
                expected=-inputs['advantages'].mean()/32
                loss_errors.append(float(abs(loss.detach()-expected)))
                torch.testing.assert_close(loss.detach(),expected,atol=2e-5,rtol=2e-3)
                loss.backward()
            grads=[p.grad for p in trainer.model.parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
            gradient_abs_sum=sum(float(g.abs().sum()) for g in grads)
            assert gradient_abs_sum>0
            assert all(p.grad is None for p in trainer.model.parameters() if not p.requires_grad)
            scorer.assert_frozen();assert trainer.optimizer is None and trainer.lr_scheduler is None
            trainer.model.zero_grad(set_to_none=True)
            return dict(rows=len(inputs_list),max_actual_loss_vs_ratio1_formula_error=max(loss_errors),finite_actor_LoRA_gradients=True,
                actor_gradient_abs_sum=gradient_abs_sum,wrong_positive=sum(x['outcome']==0 and x['advantage_at_loss']>0 for x in trainer.c_loss_trace[start:]))
        with stage('live_real_actor_forward_backward_no_update'):
            live_loss=backward_phase(live_prepared)
        # Cover all-correct/all-wrong/mixed and the three old failures even if
        # one fresh sampled batch does not naturally contain every group type.
        old_groups=[r for r in rows(CD/'group_advantages.jsonl') if r['arm']=='C']
        chosen=[]
        for kind in ['all_correct','mixed']:
            gs=sorted([g for g in old_groups if g['dataset']=='probe' and g['kind']==kind],key=lambda g:str(g['group_id']))
            if kind=='all_correct':gs=[g for g in gs if max(g['min_scores'])-min(g['min_scores'])>1e-3]
            assert gs;chosen.append(gs[0])
        chosen.extend(g for g in old_groups if g['dataset']=='illustrations')
        assert len(chosen)==5
        raw={r['example_id']:r for name in ['probe.jsonl','illustrations.jsonl'] for r in rows(CD/name)}
        cached={r['example_id']:r for r in rows(CD/'scores_C.jsonl')}
        saved=[raw[i] for g in chosen for i in g['example_ids']];assert len(saved)==40
        with stage('saved_text_real_C_native_score_parity'):
            ledger.score_records(completions=[r['completion'] for r in saved],completion_counts=[r['completion_tokens'] for r in saved],
                terminated=[r['terminated'] for r in saved],answer=[r['gold'] for r in saved],problem_id=[r['problem_id'] for r in saved],problem=[r['question'] for r in saved])
            diffs=[]
            for original,scored in zip(saved,ledger.records):
                expected=cached[original['example_id']];assert scored['outcome_reward']==expected['outcome']
                assert len(scored['step_scores'])==len(expected['step_scores'])
                diffs.extend(abs(x-y) for x,y in zip(scored['step_scores'],expected['step_scores']))
            atomic(RUN/'score_parity.json',dict(max_step_error=max(diffs,default=0),TF32_disabled_for_C=True,rows=len(saved)))
            assert max(diffs,default=0)<2e-6
            saved_records=copy.deepcopy(ledger.records)
        device=trainer.accelerator.device
        prompts=[tok.encode(lookup[r['problem_id']]['prompt'],add_special_tokens=False)[-2048:] for r in saved]
        completions=[tok.encode(r['completion'],add_special_tokens=False)+([tok.eos_token_id] if r['terminated'] else []) for r in saved]
        assert max(len(p)+len(c) for p,c in zip(prompts,completions))<=4096
        pw=max(map(len,prompts));cw=max(max(map(len,completions)),1)
        pack=dict(prompt_ids=torch.full((40,pw),tok.pad_token_id,device=device,dtype=torch.long),prompt_mask=torch.zeros((40,pw),device=device,dtype=torch.long),
            completion_ids=torch.full((40,cw),tok.pad_token_id,device=device,dtype=torch.long),completion_mask=torch.zeros((40,cw),device=device,dtype=torch.long),
            advantages=torch.zeros(40,device=device,dtype=torch.float32))
        for i,(pi,ci) in enumerate(zip(prompts,completions)):
            pack['prompt_ids'][i,-len(pi):]=torch.tensor(pi,device=device);pack['prompt_mask'][i,-len(pi):]=1
            if ci:pack['completion_ids'][i,:len(ci)]=torch.tensor(ci,device=device);pack['completion_mask'][i,:len(ci)]=1
        pack['num_items_in_batch']=pack['completion_mask'].sum()
        with stage('saved_text_actual_reference_logps'):
            with torch.no_grad(),trainer.accelerator.unwrap_model(trainer.model).disable_adapter():
                ref,_=trainer._get_per_token_logps_and_entropies(trainer.model,torch.cat([pack['prompt_ids'],pack['completion_ids']],1),
                    torch.cat([pack['prompt_mask'],pack['completion_mask']],1),cw,batch_size=1)
            pack['ref_per_token_logps']=ref
        trainer.attach_c_signal(pack,saved_records,phase='saved_real_text_retokenized')
        with stage('saved_text_real_actor_forward_backward_no_update'):
            saved_loss=backward_phase([slice_rows(pack,i,i+1) for i in range(40)])
        assert not ((pack['advantages']>0)&torch.tensor([r['outcome_reward']==0 for r in saved_records],device=device)).any()
        for i,r in enumerate(saved):
            if r.get('known_error_case'):assert pack['advantages'][i]==0
        with stage('corrupted_signal_guard_tests'):
            inputs=slice_rows(pack,0,8);original=inputs['advantages'].clone();assert original.std()>0
            rejected=[]
            altered={'second_group_normalization':(original-original.mean())/(original.std()+1e-4),
                'rescale_after_builder':original*2,'shuffle_advantage_without_tokens':original.flip(0)}
            for name,value in altered.items():
                try:trainer.assert_c_loss_input({**inputs,'advantages':value})
                except RuntimeError:rejected.append(name)
                else:raise AssertionError('Guard missed '+name)
            missing={k:v for k,v in inputs.items() if k!='c_bounded_trace_id'}
            try:trainer.assert_c_loss_input(missing)
            except RuntimeError:rejected.append('missing_provenance')
            else:raise AssertionError('Missing provenance accepted')
            corrupt={**inputs,'completion_ids':inputs['completion_ids'].clone()};corrupt['completion_ids'][0,0]+=1
            try:trainer.assert_c_loss_input(corrupt)
            except RuntimeError:rejected.append('token_tag_mismatch')
            else:raise AssertionError('Token mismatch accepted')
            wrong=slice_rows(pack,16,17);assert trainer.c_expected[int(wrong['c_bounded_trace_id'][0])]['outcome']==0
            try:trainer.assert_c_loss_input({**wrong,'advantages':torch.full_like(wrong['advantages'],.01)})
            except RuntimeError:rejected.append('injected_wrong_positive')
            else:raise AssertionError('Injected wrong-positive accepted')
            for name,call in [('formal_RL_entry',lambda:trainer.train()),('optimizer_creation',lambda:trainer.create_optimizer())]:
                try:call()
                except RuntimeError:rejected.append(name)
                else:raise AssertionError('Unauthorized operation accepted')
        with stage('parameter_hashes_and_preservation_after'):
            actor_after=parameter_hash(trainer.model);prm_after=parameter_hash(scorer.model)
            assert actor_before==actor_after and prm_before==prm_after
            assert adapter_hash(trainer.model)==expected_adapter
            assert before==protect()
        assert trainer.optimizer is None and trainer.lr_scheduler is None and trainer.state.global_step==0
        assert len(trainer.c_loss_trace)==72 and all(x['bitwise_preserved'] for x in trainer.c_loss_trace)
        with (RUN/'actual_loss_input_trace.jsonl').open('w') as f:
            for r in trainer.c_loss_trace:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        atomic(RUN/'generation_checks.json',trainer.c_generation_checks)
        atomic(RUN/'parameter_preservation.json',dict(actor_before=actor_before,actor_after=actor_after,
            C_before=prm_before,C_after=prm_after,actor_and_C_unchanged=True,originals=before))
        result=dict(passed=True,status='complete_no_update',source_release_sha256=release,live_generated_rows=32,new_generation_calls_this_attempt=0,generation_reused_from='runs/C_bounded_no_update_20261029',
            live_generated_completion_tokens=sum(r['completion_tokens'] for r in live_records),exact_token_ID_rows_verified=32,
            saved_real_text_rows=40,saved_groups=[dict(dataset=g['dataset'],kind=g['kind'],group=g['group_id']) for g in chosen],
            native_C_score_max_error_vs_saved=max(diffs,default=0),live_loss=live_loss,saved_loss=saved_loss,
            actual_loss_rows=72,wrong_positive_at_actual_loss=0,loss_input_bitwise_preserved=True,
            actual_outcome_normalizations=1,post_combination_normalizations=0,legacy_mixed_advantages_discarded=True,
            negative_tests_rejected=rejected,optimizer_constructed=False,optimizer_steps=0,formal_RL=False,D_adaptation=False,
            actor_and_C_parameters_unchanged=True,original_ABCD_preserved=True,stages=stages,total_seconds=time.time()-started,
            limitations=['Exact tokens reused from first actual live generation after a scorer-only TF32 precision correction; no resampling. Not convergence or policy accuracy.',
                'Saved actor tokens were retokenized explicitly; only live rows claim exact emitted tokens.',
                'No optimizer states allocated; measured memory is no-update footprint, not full training peak.',
                'Shared GPU and audit-only vLLM cache reduction; timings do not establish dedicated-card throughput.',
                'The new signal does not itself fix C raw PRM ranking.'])
        atomic(RUN/'result.json',result);event('complete_no_update',passed=True)
    except Exception as exc:
        atomic(RUN/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()));event('failed',error=str(exc));raise

if __name__=='__main__':main()
