from common import *
import argparse
import numpy as np

def cohort_data(name):
    path=RUN/'evaluation'/name;repeat_paths=sorted(path.glob('repeat_*'))
    assert repeat_paths and all(read(p/'summary.json')['passed'] for p in repeat_paths)
    dataset='math500' if name=='endpoints_math' else 'gsm8k' if name=='endpoints_gsm' else 'validation'
    canonical=[(r['problem_id'],r['question_sha256'],r['answer']) for r in rows(ROOT/f'data/prepared/{dataset}.jsonl')]
    data={}
    for p in repeat_paths:
        for model in read(p/'summary.json')['summary']:
            rr=rows(p/model/'predictions.jsonl')
            assert [(r['problem_id'],r['question_sha256'],r['gold']) for r in rr]==canonical
            assert sha(p/model/'predictions.jsonl')==read(p/'summary.json')['summary'][model]['predictions_sha256']
            data.setdefault(model,[]).append(rr)
    return data

def means(data):
    out={}
    for model,repeats in data.items():
        ids=[[r['problem_id'] for r in rr] for rr in repeats];assert all(x==ids[0] for x in ids)
        arr=np.array([[r['reward'] for r in rr] for rr in repeats],dtype=float)
        out[model]={'n_questions':arr.shape[1],'repeats':len(repeats),'per_repeat_accuracy':arr.mean(1).tolist(),
            'mean_accuracy':float(arr.mean()),'same_checkpoint_repeat_binary_flips':int((arr[0]!=arr[-1]).sum()) if len(repeats)>1 else None,
            'per_question_mean':arr.mean(0).tolist()}
    return out

def paired(a,b):
    d=np.array(a)-np.array(b);rng=np.random.default_rng(2851108)
    boot=np.concatenate([d[rng.integers(0,len(d),(1000,len(d)))].mean(1) for _ in range(20)])
    return {'difference_pp':float(d.mean()*100),'paired_question_bootstrap95CI_pp':(np.quantile(boot,[.025,.975])*100).tolist(),
        'questions':len(d),'note':'Repeat means retain question dependence; interval excludes train-seed and full runtime-repeat variance.'}

def gate():
    base=means(cohort_data('baseline'));end=means(cohort_data('final'))
    delta0=base['C_bounded']['mean_accuracy']-base['A']['mean_accuracy']
    delta1=end['C_bounded']['mean_accuracy']-end['A']['mean_accuracy']
    checks={'gap_positive':delta1>0,'gap_at_least_baseline':delta1+1e-12>=delta0,
        'C_accuracy_not_below_baseline':end['C_bounded']['mean_accuracy']+1e-12>=base['C_bounded']['mean_accuracy'],
        'C_wins_each_repeat':all(c>a for c,a in zip(end['C_bounded']['per_repeat_accuracy'],end['A']['per_repeat_accuracy']))}
    result={'eligible':all(checks.values()),'checks':checks,'baseline_gap_pp':delta0*100,'final_gap_pp':delta1*100,
        'criterion_frozen_in_protocol':True,'uses_test_results':False,'dispatch_authorized_in_this_pipeline':False,
        'note':'Eligibility for a separately scoped seed42 continuation, not statistical confirmation.'}
    atomic(RUN/'seed42_eligibility.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gate-only',action='store_true');a=ap.parse_args()
    verify();source_unchanged()
    if a.gate_only:print(json.dumps(gate()));return
    stats={c:means(cohort_data(c)) for c in ['baseline','nine','final','endpoints_math','endpoints_gsm']}
    comparisons={}
    for c in ['baseline','nine','final']:
        comparisons[c+'_C_minus_A']=paired(stats[c]['C_bounded']['per_question_mean'],stats[c]['A']['per_question_mean'])
    for c in ['endpoints_math','endpoints_gsm']:
        for k in [288,576]:comparisons[c+f'_C_minus_A_at{k}']=paired(stats[c][f'C_bounded_{k}']['per_question_mean'],stats[c][f'A_{k}']['per_question_mean'])
        for arm in ARMS:comparisons[c+f'_{arm}_12M_minus6M']=paired(stats[c][arm+'_576']['per_question_mean'],stats[c][arm+'_288']['per_question_mean'])
    for arm in ARMS:
        for step in [432,576]:assert read(RUN/arm/f'stage_{step}_result.json')['passed']
        sig=rows(RUN/arm/'actual_signals.jsonl');assert len(sig)==288
        assert [x['step'] for x in sig]==list(range(288,576))
        for batch in sig:
            y=np.array(batch['outcomes']).reshape(-1,8);m=np.array(batch['minima']).reshape(-1,8)
            outcome=(y-y.mean(1,keepdims=True))/(y.std(1,ddof=1,keepdims=True)+1e-4)
            mu=np.divide((y*m).sum(1,keepdims=True),y.sum(1,keepdims=True),out=np.zeros((4,1)),where=y.sum(1,keepdims=True)!=0)
            expected=outcome if arm=='A' else outcome+.2*y*(m-mu)
            assert np.max(abs(expected.flatten()-np.array(batch['advantages'])))<2e-6
            assert not np.any(np.array(batch['advantages'])[y.flatten()==0]>0)
    protected_unchanged()
    atomic(RUN/'comparison.json',{'passed':True,'statistics':stats,'paired_comparisons':comparisons,'seed42_eligibility':gate(),
        'seed':17,'same_update_counts':True,'reward_loss_unchanged':True,'source_checkpoints_unchanged':True,
        'limits':['Controlled continuation with native-LoRA runtime; historical vLLM RNG not bitwise resumed.',
                  'Validation previously used; question bootstrap is not a training-seed interval.',
                  'Original6M test predictions preserved; newly common-scored endpoint baselines reported separately.',
                  'No diagnostic result alters reward/loss or checkpoint selection.']})

if __name__=='__main__':main()
