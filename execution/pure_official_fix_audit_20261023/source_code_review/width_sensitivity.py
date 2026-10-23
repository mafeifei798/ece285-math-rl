"""CPU-only fixed-trajectory PURE baseline denominator sensitivity.

Run from any directory with Python 3. No imports from torch or project training code.
N changes the official code's baseline denominator only. All existing rewards,
step ends, valid-token masks and groups stay fixed; no longer outputs are generated.
"""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'execution/pure_official_prm_offline_20261022'
OUT = Path(__file__).resolve().parent
ARMS = ('P_VR','QWEN_PRM','PURE_NATIVE','PURE_RL_PROMPT')


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = dict(inputs=ROOT/'execution/pure_empty_boundary_20261022/inputs.jsonl',
        qwen=OLD/'remote/qwen_single/scored.jsonl',pure=OLD/'remote/scoring_single/scored.jsonl',
        prior=OLD/'per_response_metrics.jsonl')
    inputs,qwen,pure,prior = [load(paths[k]) for k in ('inputs','qwen','pure','prior')]
    assert len(inputs)==len(qwen)==len(pure)==len(prior)==288
    batches = collections.defaultdict(list)
    for i,row in enumerate(inputs):
        batches[row['batch_id']].append(i)
    fixed_batches, skipped = {}, []
    for bid,ix in batches.items():
        if any('alignment_error' in qwen[i] for i in ix):
            skipped.append(bid)
            continue
        assert len(ix)==32
        fixed_batches[bid]=ix
    assert skipped==['B:210'] and len(fixed_batches)==8
    details=[];errors=[]
    for arm in ARMS:
        for bid,ix in fixed_batches.items():
            row_returns=[]
            for i in ix:
                length=len(qwen[i]['reconstructed_ids'])
                sparse={}
                if arm=='P_VR':sparse[length-1]=inputs[i]['outcome']
                else:
                    ps=qwen[i]['native']['step_scores'] if arm=='QWEN_PRM' else pure[i]['variants'][
                        'native_raw' if arm=='PURE_NATIVE' else 'official_rl_prompt']['step_scores']
                    rs=[2*p-1 for p in ps]
                    z=[-r/.1 for r in rs];ex=[math.exp(x-max(z)) for x in z];den=math.fsum(ex)
                    for pos,r,w in zip(qwen[i]['alignment']['actor_positions'],rs,ex):sparse[pos]=.5*r*w/den
                ret=[0.]*length;value=0.
                for t in reversed(range(length)):
                    value+=sparse.get(t,0.)
                    ret[t]=value
                row_returns.append(ret)
            for width in (2048,7168,8192):
                raw=[]
                for j,i in enumerate(ix):
                    peers=[k for k,p in enumerate(ix) if p!=i and inputs[p]['problem_id']==inputs[i]['problem_id']]
                    assert len(peers)==7
                    base=math.fsum(math.fsum(row_returns[k]) for k in peers)/(7*width)
                    raw.append([value-base for value in row_returns[j]])
                valid=[x for r in raw for x in r]
                mu=math.fsum(valid)/len(valid)
                var=math.fsum((x-mu)**2 for x in valid)/(len(valid)-1)
                for j,i in enumerate(ix):
                    w=(raw[j][-1]-mu)/math.sqrt(var+1e-8)
                    if width==2048:
                        r=prior[i]['variants'][arm]
                        errors.append(max(abs(w-r['whitened_terminal_advantage']),
                            abs(raw[j][-1]-r['raw_RLOO_terminal_advantage'])))
                    if not inputs[i]['completion'].strip():
                        details.append(dict(arm=arm,width=width,index=i,batch_id=bid,
                            raw=raw[j][-1],white=w,valid_tokens=len(valid),response_tokens=len(raw[j])))
    assert len(errors)==1024 and max(errors)<1e-5
    aggregate={}
    for arm in ARMS:
        aggregate[arm]={}
        for width in (2048,7168,8192):
            rows=[x for x in details if x['arm']==arm and x['width']==width]
            assert len(rows)==8
            aggregate[arm][str(width)]=dict(empty_cases=8,positive_raw=sum(x['raw']>0 for x in rows),
                positive_white=sum(x['white']>0 for x in rows),min_raw=min(x['raw'] for x in rows),
                max_raw=max(x['raw'] for x in rows),min_white=min(x['white'] for x in rows),
                max_white=max(x['white'] for x in rows))
    result=dict(passed=True,new_generations=0,optimizer_steps=0,model_loaded=False,torch_imported=False,
        widths=[2048,7168,8192],replayed_batches=list(fixed_batches),skipped_batches=skipped,
        immutable_inputs={k:dict(path=str(p),sha256=digest(p)) for k,p in paths.items()},
        script_sha256=digest(Path(__file__)),width2048_reference_max_abs_error=max(errors),
        same_existing_valid_token_masks=True,same_existing_rewards_and_groups=True,
        aggregate=aggregate,details=details,
        scope='Only change the denominator N in pinned verl RLOO on saved trajectories. No longer responses generated; not a full long-context training/evaluation result.',
        formula='baseline_i = sum_{k!=i,same_prompt} sum_{valid t} return[k,t] / ((G-1)*N); whiten once over unchanged valid tokens',
        vr_caveat='P-VR here follows pinned verl shared-token-baseline implementation; paper Eq7 uses a separate response-level VR baseline.')
    target=OUT/'width_sensitivity.json'
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(aggregate=aggregate,max_reference_error=max(errors),saved=str(target)),indent=2))


if __name__=='__main__':main()
