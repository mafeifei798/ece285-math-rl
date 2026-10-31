"""CPU-only recount/regrade, paired statistics, costs and inspection pack."""
from common_eval import *
import sys,math,re,statistics,random,collections

def features(record):
    text=record['completion'];starts=[m.start() for m in re.finditer(r'\\boxed\s*\{',text)]
    early=False;after=False
    if starts:
        start=starts[-1];brace=text.find('{',start);depth=0;end=None
        for j in range(brace,len(text)):
            if text[j]=='{':depth+=1
            elif text[j]=='}':
                depth-=1
                if depth==0:end=j+1;break
        early=start<.25*len(text)
        after=end is not None and len(text[end:].strip())>=100
    return {'empty':not text.strip(),'short_wrong':record['generated_tokens']<=32 and not record['reward'],
        'short_correct':record['generated_tokens']<=32 and bool(record['reward']),
        'truncated':record['finish_reason']!='stop','no_extractable_answer':record.get('answer') is None,
        'boxes':len(starts),'early_last_box':early,'text_after_last_box':after,
        'no_double_newline':'\n\n' not in text.strip(),'CRLF':'\r\n' in text}

def summarize(rr):
    fs=[features(r) for r in rr];correct=sum(r['reward'] for r in rr)
    return {'n':len(rr),'correct':correct,'accuracy':correct/len(rr),
       'generated_tokens':sum(r['generated_tokens'] for r in rr),
       'mean_generated_tokens':statistics.mean(r['generated_tokens'] for r in rr),
       'median_generated_tokens':statistics.median(r['generated_tokens'] for r in rr),
       'status_counts':dict(collections.Counter(r['status'] for r in rr)),
       **{k:sum(bool(x[k]) for x in fs) for k in ['empty','short_wrong','short_correct','truncated','no_extractable_answer','early_last_box','text_after_last_box','no_double_newline','CRLF']},
       'multiple_boxes':sum(x['boxes']>1 for x in fs),'mean_boxes':statistics.mean(x['boxes'] for x in fs)}

def main():
    guard();release=release_check();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (RUN/'comparison.json').exists(),'Do not overwrite completed analysis'
    import torch
    assert not torch.cuda.is_initialized()
    import numpy as np
    sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'))
    from repairs import repaired_outcome
    contract=read(RUN/'contract.json');gen=read(RUN/'generation_complete.json');assert gen['passed']
    assert gen['contract_sha256']==sha(RUN/'contract.json') and contract['release_sha256']==release
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
    summary={a:{} for a in ARMS};by_dataset={};regraded=0
    for dataset in DATASETS:
        questions=rows(ROOT/'data/prepared'/f'{dataset}.jsonl');qmap={r['problem_id']:r for r in questions}
        by_dataset[dataset]={}
        for arm in ARMS:
            p=RUN/arm/f'{dataset}.jsonl';assert sha(p)==gen['summary'][arm][dataset]['predictions_sha256']
            rr=rows(p);assert len(rr)==len(qmap)==len({r['problem_id'] for r in rr})
            assert [r['problem_id'] for r in rr]==[q['problem_id'] for q in questions]
            for r in rr:
                q=qmap[r['problem_id']]
                assert r['question_sha256']==q['question_sha256'] and r['gold']==q['answer']
                assert r['adapter_sha256']==contract['states'][arm]['adapter_sha256']
                assert len(r['completion_token_ids'])==r['generated_tokens']<=2048
                judged=repaired_outcome(r['completion'],q['answer'],r['finish_reason']=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
                for k,v in judged.items():assert r[k]==v,('regrade_difference',arm,dataset,r['problem_id'],k)
                regraded+=1
            summary[arm][dataset]={**summarize(rr),**{k:gen['summary'][arm][dataset][k] for k in ['inference_seconds','grading_seconds','predictions_sha256']}}
            assert summary[arm][dataset]['correct']==gen['summary'][arm][dataset]['correct']
            by_dataset[dataset][arm]={r['problem_id']:r for r in rr}
    paired={};contrasts=[('A','initial'),('C_mix','A'),('C_bounded','A'),('C_bounded','C_mix'),('C_bounded','initial')]
    for dataset in DATASETS:
        ids=sorted(by_dataset[dataset]['initial']);n=len(ids);paired[dataset]={}
        rng=np.random.default_rng(285);indices=rng.integers(0,n,size=(10000,n),dtype=np.int32)
        for lhs,rhs in contrasts:
            d=np.array([by_dataset[dataset][lhs][k]['reward']-by_dataset[dataset][rhs][k]['reward'] for k in ids])
            samples=d[indices].mean(1)*100;wins=int((d>0).sum());losses=int((d<0).sum());discord=wins+losses
            pval=min(1.,2*sum(math.comb(discord,i) for i in range(min(wins,losses)+1))/2**discord) if discord else 1.
            paired[dataset][lhs+' minus '+rhs]={'delta_pp':float(d.mean()*100),'wins':wins,'losses':losses,
               'unchanged':n-discord,'paired_question_bootstrap95_pp':np.quantile(samples,[.025,.975]).tolist(),
               'exact_mcnemar_two_sided_p':pval,'bootstrap_replicates':10000,'bootstrap_seed':285}
        del indices
    training={};curves={}
    for arm in ARMS[1:]:
        final=read(PILOT/arm/'final_result.json');smoke=read(PILOT/arm/'smoke_result.json');progress=read(PILOT/arm/'progress.json')
        audits=rows(PILOT/arm/'update_audits.jsonl');assert len(audits)==48 and [x['global_step'] for x in audits]==list(range(1,49))
        training[arm]={'accounting':final['accounting'],'updates':final['global_step'],
           'successful_smoke_seconds':smoke['seconds'],'successful_continuation_seconds':final['seconds'],
           'successful_training_seconds_sum':smoke['seconds']+final['seconds'],
           'peak_allocator_reserved_GiB':max(smoke['peak_reserved_GiB'],progress['peak_reserved_GiB']),
           'peak_allocator_allocated_GiB':max(smoke['peak_allocated_GiB'],progress['peak_allocated_GiB']),
           'frozen_PRM':final['PRM_frozen'],'source_release_sha256':final['source_release_sha256']}
        curves[arm]=[{k:r[k] for k in ['global_step','generated_completion_tokens','correct_completions','generated_completions','truncated_completions','loss_wrong_positive','outcome_group_counts','process_abs_sum','outcome_abs_sum','elapsed_seconds','phase']} for r in audits]
    resources=rows(RUN/'resource_samples.jsonl');good=[r for r in resources if 'own_total_MiB' in r]
    cost={'evaluation_generated_tokens':sum(summary[a][n]['generated_tokens'] for a in ARMS for n in DATASETS),
       'inference_seconds_sum':sum(summary[a][n]['inference_seconds'] for a in ARMS for n in DATASETS),
       'grading_seconds_sum':sum(summary[a][n]['grading_seconds'] for a in ARMS for n in DATASETS),
       'generation_process_seconds':gen['elapsed_seconds'],'startup_seconds':gen['startup_seconds'],
       'own_GPU_memory_peak_MiB':max((r['own_total_MiB'] for r in good),default=None),
       'resource_samples':len(good),'sampling_seconds':15,
       'limits':'Shared GPU; sampled memory can miss transients. Process allocation excludes other jobs. Per-cell inference time includes contention, not isolated algorithm speed.',
       'new_training_tokens':0,'new_optimizer_updates':0,'new_PRM_calls':0}
    # Diagnostic pack: balanced same-question wins/losses; not a random prevalence estimate.
    pack=[];rng=random.Random(285)
    for dataset in DATASETS:
        qmap={r['problem_id']:r for r in rows(ROOT/'data/prepared'/f'{dataset}.jsonl')};p=by_dataset[dataset]
        strata={
         'bounded_gain':[k for k in qmap if p['C_bounded'][k]['reward']>p['A'][k]['reward']],
         'bounded_loss':[k for k in qmap if p['C_bounded'][k]['reward']<p['A'][k]['reward']],
         'mixed_loss':[k for k in qmap if p['C_mix'][k]['reward']<p['A'][k]['reward']]}
        for label,ids in strata.items():
            for ident in rng.sample(sorted(ids),min(4,len(ids))):
                pack.append({'dataset':dataset,'stratum':label,'problem_id':ident,'problem':qmap[ident]['problem'],
                   'gold':qmap[ident]['answer'],'outputs':{a:{k:v for k,v in p[a][ident].items() if k!='completion_token_ids'} for a in ARMS}})
    atomic(RUN/'case_review_pack.json',{'selection':'post-hoc stratified examples; not prevalence or human gold labels','cases':pack})
    atomic(RUN/'training_curves.json',curves)
    protected=unchanged()
    acceptance={'passed':True,'regraded':regraded,'expected':7276,'contracts_and_hashes_checked':True,
        'raw_token_count_checked':True,'per_arm_adapter_hash_checked':True,'protected_files_unchanged':protected,
        'CUDA_initialized':torch.cuda.is_initialized(),'optimizer_updates':0,'release_sha256':release}
    atomic(RUN/'acceptance.json',acceptance)
    result={'passed':True,'summary':summary,'paired':paired,'training':training,'cost':cost,
        'acceptance':acceptance,'protocol':contract['protocol'],'contract_sha256':sha(RUN/'contract.json'),
        'limitations':['One training seed; paired intervals cover question sampling only, not training-seed robustness.',
          'The current500-question validation and these test benchmarks are different sets; do not mix their scores.',
          'Known exact PRM800K question overlap audit is not a guarantee against all pretraining/paraphrase exposure.',
          'No checkpoint or prompt chosen using this final evaluation; historical benchmark results have previously been viewed.',
          'Short correct answers are not hacking; short wrong or early boxes are diagnostics, not proof of a failure mechanism.',
          'Correct final answer does not certify a valid proof; qualitative review is separate from frozen benchmark scores.',
          'Recorded training seconds cover successful smoke/continuation, not every earlier failed setup or environment repair.',
          'C-mix actual-minus-outcome advantage is not a uniquely separable process component.'],
        'new_training_started':False,'finished_unix':time.time()}
    atomic(RUN/'comparison.json',result);event('CPU_audit_complete',comparison=str(RUN/'comparison.json'))
    print(json.dumps({'passed':True,'summary':summary,'paired':paired,'cost':cost},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
