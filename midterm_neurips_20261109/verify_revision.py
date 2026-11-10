"""CPU-only checks for the clarity revision: costs, counts and power estimate."""
from pathlib import Path
from statistics import NormalDist
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
inputs = {}

def read(rel):
    p = ROOT / rel
    inputs[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())

def main():
    costs = {}
    for seed, directory in [(17,'abc_seed17_resumed_v1_20261104'),(42,'ac_seed42_v1_20261106')]:
        d=read(f'execution/{directory}/completed_evidence/comparison.json')
        assert d['passed'] and d['global_step']==288
        costs[str(seed)]={}
        for arm, a in d['accounting'].items():
            costs[str(seed)][arm]={'PRM_input_tokens':a['prm_input_tokens'],
                'PRM_forward_seconds':a['prm_forward_seconds']}
            if seed==42:
                costs[str(seed)][arm]['full_run_seconds']=d['training_seconds'][arm]
            else:
                costs[str(seed)][arm]['continuation_only_seconds']=d['resumed_training_seconds_since_20261104'][arm]
        if seed==42:
            assert all(d['paired'][ds]['C_bounded minus A']['paired_question_bootstrap95_pp'][0]<0<d['paired'][ds]['C_bounded minus A']['paired_question_bootstrap95_pp'][1]
                       for ds in ['math500','gsm8k'])
    assert costs['42']['A']['full_run_seconds']/3600>2.28
    assert round(costs['42']['C_bounded']['full_run_seconds']/3600,2)==2.50
    assert costs['17']['C_mix']['PRM_input_tokens']==5534256
    assert costs['17']['C_bounded']['PRM_input_tokens']==5544633
    pb=read('execution/cd_prm_7b_pair_20261028/artifacts/summary.json')['processbench']
    counts={}
    for arm,valid,error in [('B',58,87),('C',56,94),('D',9,45)]:
        assert pb[arm]['valid_count']==64 and pb[arm]['error_count']==128
        assert math.isclose(pb[arm]['valid_acceptance']*64,valid)
        assert math.isclose(pb[arm]['first_error_accuracy']*128,error)
        counts[arm]={'valid_accepted':valid,'valid_total':64,'first_error_hits':error,'error_total':128}
    z=NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(.8)
    power={'method':'Normal approximation to a paired binary difference; assumed discordance 0.10.',
        'two_sided_alpha':.05,'power':.8,'n500_MDE_pp':100*z*math.sqrt(.1/500),
        'n1319_MDE_pp':100*z*math.sqrt(.1/1319),
        'limitation':'Question-level approximation, not a guarantee of detecting small effects or seed variability.'}
    requirements=read('execution/midterm_requirements_20261109/requirements.json')
    result={'passed':True,'GPU_used':False,'model_calls':0,'training_updates':0,
        'costs':costs,'ProcessBench_counts':counts,'power_estimate':power,
        'requirements':requirements,'inputs_sha256':inputs}
    (HERE/'clarity_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'GPU_used':False,'n500_MDE_pp':power['n500_MDE_pp']}))

if __name__=='__main__':main()
