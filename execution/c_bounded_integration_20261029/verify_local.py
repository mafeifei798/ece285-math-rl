"""Recompute accepted real-loss advantages from delivered JSON, CPU only."""
from pathlib import Path
import json,hashlib,statistics as st

BASE=Path(__file__).resolve().parent
P=BASE/'artifacts'
def read(n):return json.loads((P/n).read_text())
def rows(n):return [json.loads(s) for s in (P/n).open() if s.strip()]

def main():
    result=read('result.json');trace=rows('actual_loss_input_trace.jsonl')
    bytag={r['tag']:r for r in trace};assert len(trace)==len(bytag)==72
    errors=[]
    for start in range(0,72,8):
        group=[bytag[i] for i in range(start,start+8)]
        y=[r['outcome'] for r in group];m=[r['process_min'] for r in group]
        correct=[p for v,p in zip(y,m) if v];center=st.fmean(correct) if correct else 0.
        for r,v,p in zip(group,y,m):
            expected=(v-st.fmean(y))/(st.stdev(y)+1e-4)+.2*v*(p-center)
            errors.append(abs(expected-r['advantage_at_loss']))
            assert r['advantage_at_loss']==r['advantage'] and r['bitwise_preserved']
            assert v or r['advantage_at_loss']<=0
            assert abs(r['process_component'])<=.2+1e-6
    assert max(errors)<4e-7 and result['optimizer_steps']==0 and not result['optimizer_constructed']
    assert hashlib.sha256((P/'source_release.json').read_bytes()).hexdigest()==result['source_release_sha256']
    saved=read('parameter_preservation.json')
    assert saved['actor_before']==saved['actor_after'] and saved['C_before']==saved['C_after']
    assert read('independent_final_acceptance.json')['live_and_saved_token_mask_fingerprints_verified']==72
    print(json.dumps(dict(passed=True,rows=72,wrong=sum(r['outcome']==0 for r in trace),wrong_positive=0,
        max_FP64_difference=max(errors),optimizer_steps=0,scope='CPU verification of delivered evidence; no model loads'),indent=2))

if __name__=='__main__':main()
