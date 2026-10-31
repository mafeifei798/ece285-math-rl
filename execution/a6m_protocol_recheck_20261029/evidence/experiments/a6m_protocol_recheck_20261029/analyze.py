"""Frozen-verifier recount and paired comparisons, entirely on CPU."""
from common_eval import *
import sys,math,statistics,collections,random

def summarize(rr):
    return {'n':len(rr),'correct':sum(r['reward'] for r in rr),
        'accuracy':sum(r['reward'] for r in rr)/len(rr),
        'generated_tokens':sum(r['generated_tokens'] for r in rr),
        'mean_generated_tokens':statistics.mean(r['generated_tokens'] for r in rr),
        'truncated':sum(r['finish_reason']!='stop' for r in rr),'empty':sum(not r['completion'].strip() for r in rr),
        'short_wrong':sum(r['generated_tokens']<=32 and not r['reward'] for r in rr),
        'no_extractable_answer':sum(r.get('answer') is None for r in rr),
        'status_counts':dict(collections.Counter(r['status'] for r in rr))}

def main():
    guard();release=release_check();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (RUN/'comparison.json').exists(),'Preserve completed analysis'
    import torch,numpy as np
    assert not torch.cuda.is_initialized()
    sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'))
    from repairs import repaired_outcome
    contract=read(RUN/'contract.json');generation=read(RUN/'generation_complete.json');assert generation['passed']
    assert generation['contract_sha256']==sha(RUN/'contract.json')
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
    summary={a:{} for a in ['initial','A_1M','A_6M_historical','A_6M_recheck']};paired={};reproduction={};packs=[];regraded=0
    for dataset in DATASETS:
        qs=rows(ROOT/'data/prepared'/f'{dataset}.jsonl')
        paths={'initial':PREVIOUS/'initial'/f'{dataset}.jsonl','A_1M':PREVIOUS/'A'/f'{dataset}.jsonl',
            'A_6M_historical':HISTORICAL/'evaluation'/f'{dataset}.jsonl','A_6M_recheck':RUN/'A_6M'/f'{dataset}.jsonl'}
        rr={a:rows(p) for a,p in paths.items()};n=len(qs)
        assert sha(paths['A_6M_recheck'])==generation['summary']['A_6M'][dataset]['predictions_sha256']
        for a,records in rr.items():
            assert len(records)==n and [r['problem_id'] for r in records]==[q['problem_id'] for q in qs]
            for r,q in zip(records,qs):
                assert r['question_sha256']==q['question_sha256'] and r['gold']==q['answer']
                if a=='A_6M_recheck':
                    assert len(r['completion_token_ids'])==r['generated_tokens']<=2048
                    assert r['adapter_sha256']==contract['states']['A_6M']['adapter_sha256']
                j=repaired_outcome(r['completion'],q['answer'],r['finish_reason']=='stop',question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
                for k,v in j.items():assert r[k]==v,('Frozen regrade differs',a,dataset,r['problem_id'],k)
                regraded+=1
            summary[a][dataset]={**summarize(records),'predictions_sha256':sha(paths[a])}
        old,new=rr['A_6M_historical'],rr['A_6M_recheck']
        reproduction[dataset]={'same_text':sum(a['completion']==b['completion'] for a,b in zip(old,new)),
            'same_text_and_token_count':sum(a['completion']==b['completion'] and a['generated_tokens']==b['generated_tokens'] for a,b in zip(old,new)),
            'same_reward':sum(a['reward']==b['reward'] for a,b in zip(old,new)),
            'same_status':sum(a['status']==b['status'] for a,b in zip(old,new)),
            'same_correct_count':summary['A_6M_historical'][dataset]['correct']==summary['A_6M_recheck'][dataset]['correct'],
            'n':n,'bitwise_greedy_repeat_not_assumed':True}
        rng=np.random.default_rng(285);indices=rng.integers(0,n,size=(10000,n),dtype=np.int32);paired[dataset]={}
        for rhs in ['A_6M_historical','A_1M','initial']:
            d=np.array([x['reward']-y['reward'] for x,y in zip(new,rr[rhs])]);win=int((d>0).sum());loss=int((d<0).sum());nd=win+loss
            p=min(1.,2*sum(math.comb(nd,i) for i in range(min(win,loss)+1))/2**nd) if nd else 1.
            paired[dataset]['A_6M_recheck minus '+rhs]={'delta_pp':float(d.mean()*100),'wins':win,'losses':loss,'unchanged':n-nd,
                'paired_question_bootstrap95_pp':np.quantile(d[indices].mean(1)*100,[.025,.975]).tolist(),
                'bootstrap_replicates':10000,'bootstrap_seed':285,'exact_mcnemar_two_sided_p':p}
        del indices
        # Short purposive examples for inspection, not a prevalence estimate.
        for label,sign in [('gain_over_1M',1),('loss_against_1M',-1),('changed_from_historical',0)]:
            candidates=[i for i,(a,b) in enumerate(zip(new,rr['A_1M'])) if (a['reward']-b['reward'])*sign>0] if sign else [i for i,(a,b) in enumerate(zip(new,old)) if a['reward']!=b['reward']]
            candidates.sort(key=lambda i:max(len(rr[a][i]['completion']) for a in rr))
            for i in candidates[:4]:packs.append({'dataset':dataset,'stratum':label,'problem_id':qs[i]['problem_id'],'problem':qs[i]['problem'],
                'gold':qs[i]['answer'],'outputs':{a:{k:v for k,v in records[i].items() if k!='completion_token_ids'} for a,records in rr.items()}})
    samples=[r for r in rows(RUN/'resource_samples.jsonl') if 'own_total_MiB' in r]
    unchanged_count=unchanged()
    acceptance={'passed':True,'new_predictions':1819,'new_prediction_raw_tokens_verified':True,'frozen_verifier_regraded_records':regraded,
        'protected_files_unchanged':unchanged_count,'CUDA_initialized':torch.cuda.is_initialized(),'new_optimizer_updates':0,'new_PRM_calls':0,
        'source_release_sha256':release,'contract_sha256':sha(RUN/'contract.json')}
    costs={'generated_tokens':sum(summary['A_6M_recheck'][d]['generated_tokens'] for d in DATASETS),
        'generation_seconds':generation['elapsed_seconds'],'startup_seconds':generation['startup_seconds'],
        'sampled_own_GPU_peak_MiB':max(r['own_total_MiB'] for r in samples),'memory_sampling_seconds':15,
        'shared_GPU':True,'new_training_tokens':0,'new_optimizer_updates':0,'new_PRM_calls':0}
    limits=['Only the already-trained historical A checkpoint is re-evaluated; current A_1M and initial results are reused without new generation.',
        'Same prompts, questions, gold answers, base model, verifier, inference and LoRA settings as the four-model evaluation; single-model ordering replaces four-model rotation.',
        'Historical 6M A and current 1M A are different training trajectories, not checkpoints from one uninterrupted run. Training vLLM memory settings and restart histories differ.',
        'Shared-GPU timing is not isolated throughput. Floating-point and scheduling changes can alter greedy output; exact text agreement is measured rather than assumed.',
        'Paired bootstrap intervals cover questions, not training-seed uncertainty. One training seed does not establish robust gains.',
        'No changed verifier, prompt selection, new training, PRM inference or checkpoint selection. Correct final answers do not certify valid derivations.']
    atomic(RUN/'case_review_pack.json',{'selection':'Purposive short paired examples; not prevalence/human labels','cases':packs})
    atomic(RUN/'acceptance.json',acceptance)
    result={'passed':True,'summary':summary,'paired':paired,'reproduction':reproduction,'cost':costs,'acceptance':acceptance,
        'protocol':PROTOCOL,'limitations':limits,'contract_sha256':sha(RUN/'contract.json'),'finished_unix':time.time(),'new_training':False}
    atomic(RUN/'comparison.json',result);event('CPU_audit_complete')
    print(json.dumps({'passed':True,'summary':summary,'paired':paired,'reproduction':reproduction,'cost':costs},ensure_ascii=False))
if __name__=='__main__':main()
