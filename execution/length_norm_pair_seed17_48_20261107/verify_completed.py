"""Independent local receipt, count, protocol and paired-statistic verification."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,random
HERE=Path(__file__).resolve().parent
E=HERE/'completed_evidence'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest=read(HERE/'completed_manifest.json')
    for name,v in manifest['files'].items():
        p=E/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
    comparison=read(E/'comparison.json');assert comparison['passed']
    no_update=read(E/'admission/no_update/result.json')
    assert no_update['passed'] and no_update['optimizer_steps']==0 and no_update['normalizations_after_scalar']==no_update['wrong_positive_tokens']==0
    assert no_update['actor_base_unchanged'] and no_update['actor_adapter_unchanged'] and no_update['PRM_frozen']
    questions=rows(E/'admission/validation.jsonl');ids=[q['problem_id'] for q in questions]
    assert len(ids)==len(set(ids))==500
    preds={a:rows(E/'evaluation'/a/'predictions.jsonl') for a in ['source','grpo','fixed_length']}
    metrics={}
    for a,rr in preds.items():
        assert [r['problem_id'] for r in rr]==ids
        for r,q in zip(rr,questions):
            assert r['question_sha256']==q['question_sha256'] and r['gold']==q['answer']
            assert r['reward'] in [0,1] and r['generated_tokens']==len(r['completion_token_ids'])<=2048
        metrics[a]={'n':len(rr),'correct':int(sum(r['reward'] for r in rr)),
                    'truncated':sum(r['finish_reason']!='stop' for r in rr),
                    'missing_or_invalid_box':sum(r['status']=='missing_or_invalid_box' for r in rr),
                    'tokens':sum(r['generated_tokens'] for r in rr),
                    'wrong_status_counts':dict(Counter(r['status'] for r in rr if not r['reward']))}
        for k in ['n','correct','truncated','missing_or_invalid_box','tokens']:
            assert metrics[a][k]==comparison['validation_summary'][a][k],(a,k)
    cfgs=[];orders=[]
    for a in ['grpo','fixed_length']:
        b=E/a;updates=rows(b/'update_audits.jsonl');gens=rows(b/'generation_checks.jsonl');rr=rows(b/'rollouts.jsonl')
        result=read(b/'result.json');acceptance=read(b/'first_update_acceptance.json')
        assert result['passed'] and result['additional_updates']==48 and result['global_step']==336
        assert result['actor_base_unchanged'] and result['PRM_frozen'] and result['source_checkpoint_unchanged']
        assert len(updates)==len(gens)==48 and len(rr)==1536
        assert [u['global_step'] for u in updates]==list(range(289,337))
        assert all(acceptance[k] for k in ['passed','adapter_exact','optimizer_exact','scheduler_exact','torch_cpu_rng_exact','torch_cuda_rng_exact','actual_optimizer_update','advantage_exact_at_loss'])
        assert sum(r['completion_tokens'] for r in rr)==comparison['training_costs'][a]['new_generation_tokens']
        orders.append([g['question_ids'] for g in gens])
        assert all(g['wrong_positive_tokens']==g['post_scalar_normalizations']==0 for g in gens)
        signal=rows(b/'signal_trace.jsonl');assert len(signal)==48
        assert all(g['variant']=='bounded' and g['normalizations_after_scalar']==g['wrong_positive_tokens']==0 for g in signal)
        reduction=rows(b/'loss_reductions.jsonl');assert len(reduction)==1536
        kind='grpo' if a=='grpo' else 'dr_grpo'
        assert all(g['loss_type']==kind and g['gradient_accumulation_steps']==32 and g['constant_length']==2048 and g['post_scalar_normalizations']==0 for g in reduction)
        cfg=read(b/'build.json')['settings'];assert cfg['loss_type']==kind and cfg['max_completion_length']==2048 and cfg['learning_rate']==5e-6
        cfgs.append({k:v for k,v in cfg.items() if k not in ['output_dir','logging_dir','run_name','loss_type']})
    assert cfgs[0]==cfgs[1]
    assert orders[0]==orders[1]==read(E/'admission/preflight.json')['question_groups']
    left,right=preds['fixed_length'],preds['grpo'];diff=[l['reward']-r['reward'] for l,r in zip(left,right)]
    wins=sum(x==1 for x in diff);losses=sum(x==-1 for x in diff);expected=comparison['paired_comparisons']['fixed_length_minus_grpo']
    assert wins==expected['wins'] and losses==expected['losses']
    assert math.isclose(100*sum(diff)/500,expected['difference_percentage_points'],abs_tol=1e-12)
    n=wins+losses;p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    assert math.isclose(p,expected['mcnemar_exact_two_sided_p'],rel_tol=1e-10,abs_tol=1e-14)
    # Independently seeded, standard-library bootstrap. Primary interval remains the prespecified remote estimate.
    rng=random.Random(20261108);boot=sorted(100*sum(rng.choices(diff,k=500))/500 for _ in range(10000))
    def quantile(q):
        t=(len(boot)-1)*q;i=int(t);f=t-i
        return boot[i]*(1-f)+boot[min(i+1,len(boot)-1)]*f
    ci=[quantile(.025),quantile(.975)]
    assert all(abs(a-b)<=.6 for a,b in zip(ci,expected['paired_bootstrap_95CI_pp']))
    cases=read(E/'validation_disagreements.json')
    assert len(cases)==wins+losses
    out={'passed':True,'manifest_files_verified':len(manifest['files']),'metrics':metrics,
         'same_checkpoint_seed_settings_and_order':True,'updates_per_arm':48,'wrong_positive_tokens':0,
         'extra_advantage_normalizations':0,'no_update_gate_passed':True,'no_optimizer_in_smoke':True,
         'paired_wins':wins,'paired_losses':losses,'independent_standard_library_bootstrap_95CI_pp':ci,
         'primary_paired_comparison':expected,'test_sets_used':False,
         'limits':'Single seed continuation and previously used validation; bootstrap covers question sampling only.'}
    (HERE/'completion_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
