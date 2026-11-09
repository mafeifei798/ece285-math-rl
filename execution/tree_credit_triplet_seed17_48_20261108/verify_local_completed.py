"""Independent delivery/hash and standard-library paired-statistics verification."""
from pathlib import Path
import hashlib,json,math,random

HERE=Path(__file__).resolve().parent
OUT=HERE/'completed_evidence'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.open() if s.strip()]
def percentile(values,q):
    i=(len(values)-1)*q;lo=math.floor(i);hi=math.ceil(i)
    return values[lo]*(hi-i)+values[hi]*(i-lo) if hi!=lo else values[lo]


def paired(left,right):
    assert [r['problem_id'] for r in left]==[r['problem_id'] for r in right]
    differences=[int(l['reward'])-int(r['reward']) for l,r in zip(left,right)]
    n=len(differences);wins=differences.count(1);losses=differences.count(-1)
    rng=random.Random(2851108)
    sample=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n*100 for _ in range(20000))
    discord=wins+losses
    p=min(1.,2*sum(math.comb(discord,k) for k in range(min(wins,losses)+1))/(2**discord)) if discord else 1.
    return dict(difference_percentage_points=sum(differences)/n*100,wins=wins,losses=losses,
                paired_bootstrap_95CI_pp=[percentile(sample,.025),percentile(sample,.975)],
                mcnemar_exact_two_sided_p=p,n=n)


def main():
    manifest=read(HERE/'completed_manifest.json')
    for name,meta in manifest['files'].items():
        b=(OUT/name).read_bytes()
        assert len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256'],name
    c=read(OUT/'comparison.json');assert c['passed']
    independent=read(OUT/'admission/independent_tensor_acceptance.json')
    mechanisms=read(OUT/'admission/mechanism_diagnostics.json')
    assert read(OUT/'admission/verifier_replay.json')['passed']
    assert independent['passed'] and mechanisms['passed'] and not independent['CUDA_initialized']
    data={a:rows(OUT/'evaluation'/a/'predictions.jsonl') for a in ('source','chain_cb','tree_vr','tree_cb')}
    for a,rr in data.items():
        assert len(rr)==len({r['problem_id'] for r in rr})==500
        assert sum(r['reward'] for r in rr)==c['validation_summary'][a]['correct']
        assert hashlib.sha256((OUT/'evaluation'/a/'predictions.jsonl').read_bytes()).hexdigest()==c['validation_summary'][a]['predictions_sha256']
    checks={}
    for a,b in [('tree_cb','chain_cb'),('tree_vr','chain_cb'),('tree_cb','tree_vr')]:
        value=paired(data[a],data[b]);ref=c['paired_comparisons'][f'{a}_minus_{b}']
        assert value['wins']==ref['wins'] and value['losses']==ref['losses']
        assert math.isclose(value['difference_percentage_points'],ref['difference_percentage_points'],abs_tol=1e-10)
        assert math.isclose(value['mcnemar_exact_two_sided_p'],ref['mcnemar_exact_two_sided_p'],abs_tol=1e-10)
        # Independent Monte Carlo seed; endpoints can differ by one question.
        assert all(abs(x-y)<=.200001 for x,y in zip(value['paired_bootstrap_95CI_pp'],ref['paired_bootstrap_95CI_pp']))
        checks[f'{a}_minus_{b}']=value
    result={'passed':True,'manifest_files_verified':len(manifest['files']),
            'raw_batches_independently_verified':sum(v['batches'] for v in independent['arms'].values()),
            'raw_responses_independently_verified':sum(v['responses_verified'] for v in independent['arms'].values()),
            'independent_local_standard_library_statistics':checks,
            'evaluation_dataset':'previously used validation500','test_sets_used':False,
            'limits':'Single-seed48-update continuation; question bootstrap excludes train-seed/evaluation-run numerical uncertainty.'}
    (HERE/'completion_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
