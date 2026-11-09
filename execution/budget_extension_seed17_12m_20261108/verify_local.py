"""Independent local manifest, averages and primary paired statistics."""
from pathlib import Path
import json,hashlib,math,random,statistics
HERE=Path(__file__).resolve().parent;E=HERE/'completed_evidence'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]
def quantile(s,q):
    i=(len(s)-1)*q;l=math.floor(i);h=math.ceil(i)
    return s[l]+(s[h]-s[l])*(i-l)
def paired(a,b):
    d=[x-y for x,y in zip(a,b)];rng=random.Random(2851109)
    bs=sorted(sum(rng.choices(d,k=len(d)))/len(d)*100 for _ in range(20000))
    return {'difference_pp':statistics.mean(d)*100,'independent_paired_question_bootstrap95CI_pp':[quantile(bs,.025),quantile(bs,.975)]}
def main():
    manifest=read(HERE/'completed_manifest.json')
    for name,m in manifest['files'].items():
        b=(E/name).read_bytes();assert len(b)==m['bytes'] and hashlib.sha256(b).hexdigest()==m['sha256']
    c=read(E/'comparison.json');cpu=read(E/'admission/independent_acceptance.json')
    assert c['passed'] and cpu['passed'] and not cpu['CUDA_initialized'] and cpu['predictions_regraded']==15276
    reconstructed={}
    for cohort in c['statistics']:
        reconstructed[cohort]={}
        base=E/'evaluation'/cohort
        for model,reference in c['statistics'][cohort].items():
            rr=[rows(p/model/'predictions.jsonl') for p in sorted(base.glob('repeat_*')) if (p/model/'predictions.jsonl').exists()]
            assert len(rr)==reference['repeats']
            assert all([r['problem_id'] for r in x]==[r['problem_id'] for r in rr[0]] for x in rr)
            vals=[sum(x[i]['reward'] for x in rr)/len(rr) for i in range(len(rr[0]))]
            assert vals==reference['per_question_mean'] and abs(statistics.mean(vals)-reference['mean_accuracy'])<1e-12
            reconstructed[cohort][model]=vals
    comparisons={}
    for cohort in ['endpoints_math','endpoints_gsm']:
        for label,a,b in [('C_minus_A_at576','C_bounded_576','A_576'),('A_12M_minus6M','A_576','A_288'),('C_bounded_12M_minus6M','C_bounded_576','C_bounded_288')]:
            key=cohort+'_'+label;value=paired(reconstructed[cohort][a],reconstructed[cohort][b])
            ref=c['paired_comparisons'][key]
            assert abs(value['difference_pp']-ref['difference_pp'])<1e-10
            value['original_interval_pp']=ref['paired_question_bootstrap95CI_pp']
            value['interval_endpoint_max_MC_difference_pp']=max(abs(x-y) for x,y in zip(value['independent_paired_question_bootstrap95CI_pp'],ref['paired_question_bootstrap95CI_pp']))
            comparisons[key]=value
    b=c['statistics']['baseline'];f=c['statistics']['final']
    db=b['C_bounded']['mean_accuracy']-b['A']['mean_accuracy'];df=f['C_bounded']['mean_accuracy']-f['A']['mean_accuracy']
    eligible=df>0 and df+1e-12>=db and f['C_bounded']['mean_accuracy']+1e-12>=b['C_bounded']['mean_accuracy'] and all(x>y for x,y in zip(f['C_bounded']['per_repeat_accuracy'],f['A']['per_repeat_accuracy']))
    assert eligible==c['seed42_eligibility']['eligible']
    out={'passed':True,'manifest_files_checked':len(manifest['files']),'evaluation_predictions':15276,
        'new_training_batches':sum(x['batches'] for x in cpu['arms'].values()),'new_training_responses':sum(x['responses'] for x in cpu['arms'].values()),
        'averages_independently_recomputed':True,'seed42_gate_independently_recomputed':True,
        'primary_endpoint_comparisons':comparisons,
        'limits':'Independent Monte Carlo endpoints may differ. Report boundary sensitivity; question bootstrap excludes train-seed and complete runtime variation.'}
    (HERE/'completion_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
