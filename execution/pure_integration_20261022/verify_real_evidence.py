"""Independent CPU acceptance of actual model tensors after no-update run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/pure_integration_20261022'))
from common import read, read_jsonl, atomic, sha
from offline_check import independent_reference
from signals import task_rewards
from trainer import PureIntegration

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run
    result=read(run/'result.json');preserve=read(run/'preservation.json')
    assert result['passed'] and result['optimizer_steps']==0
    assert preserve['passed'] and preserve['actor_before']==preserve['actor_after'] and preserve['prm_before']==preserve['prm_after']
    rows=read_jsonl(run/'generated_exact_ids.jsonl');scores=read_jsonl(run/'scored_alignment.jsonl')
    tensors=torch.load(run/'signals.pt',map_location='cpu',weights_only=True)
    assert len(rows)==len(scores)==32
    assert len(set(r['problem_id'] for r in rows))==4
    for start in range(0,32,8):assert len(set(r['problem_id'] for r in rows[start:start+8]))==1
    # Post-GPU interface correction only removes an unnecessary P-VR dependency
    # on PRM metadata. Verify latest entry point on the saved real GPU tensors.
    vr_rows=[dict(r,outcome=s['outcome']) for r,s in zip(rows,scores)]
    vr_without_prm,_=PureIntegration.prepare(object(),'P_VR',vr_rows,None,
        tensors['signals']['P_VR']['old_logps'],tensors['reference_logps'])
    torch.testing.assert_close(vr_without_prm.advantages,tensors['signals']['P_VR']['advantages'],atol=2e-6,rtol=2e-6)
    lengths=[len(r['completion_ids']) for r in rows]
    steps=sum(len(r['native']['step_scores']) for r in scores)
    previous=run.parent/'PURE_no_update_20261022_v2'
    shifted=0
    if run.name.endswith('_v3'):
        assert sha(run/'generated_exact_ids.jsonl')==sha(previous/'generated_exact_ids.jsonl')
        prior_scores=read_jsonl(previous/'scored_alignment.jsonl')
        for new,old in zip(scores,prior_scores):
            assert new['alignment']['placement']=='nonfinal step-separator token; final EOS or actual last token'
            torch.testing.assert_close(torch.tensor(new['native']['step_scores']),torch.tensor(old['native']['step_scores']),atol=1e-6,rtol=1e-6)
            shifted+=sum(a!=b for a,b in zip(new['alignment']['actor_positions'],old['alignment']['actor_positions']))
        assert shifted==67
    for i,(r,s) in enumerate(zip(rows,scores)):
        native,alignment=s['native'],s['alignment']
        assert native['raw_sha256']==hashlib.sha256(r['completion'].encode()).hexdigest()
        assert len(native['positions'])==len(native['steps'])==len(native['step_scores'])==len(alignment['actor_positions'])
        assert alignment['actor_positions']==sorted(set(alignment['actor_positions']))
        assert all(0<=j<lengths[i] for j in alignment['actor_positions'])
        assert alignment['actor_positions'][-1]==lengths[i]-1
        for step,span in zip(native['steps'],native['spans']):
            assert r['completion'][span['raw_start']:span['raw_end']].replace('\r\n','\n')==step
    checks={}
    for arm in ['P_VR','P_PRM']:
        t=tensors['signals'][arm]
        assert t['advantages'].shape==(32,2048)
        expected_task=task_rewards(arm,lengths,[s['outcome'] for s in scores],
            [s['native']['step_scores'] for s in scores],[s['alignment']['actor_positions'] for s in scores])
        torch.testing.assert_close(t['task_rewards'],expected_task,atol=2e-6,rtol=2e-6)
        delta=tensors['reference_logps']-t['old_logps']
        kl=(delta.exp()-delta-1).clamp(-10,10)*t['mask']
        torch.testing.assert_close(t['kl'],kl,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(t['rewards'],expected_task-.001*kl,atol=2e-6,rtol=2e-6)
        expected_adv,expected_returns=independent_reference(t['rewards'],t['mask'],[i//8 for i in range(32)])
        torch.testing.assert_close(t['advantages'],expected_adv,atol=5e-6,rtol=5e-6)
        torch.testing.assert_close(t['returns'],expected_returns,atol=2e-6,rtol=2e-6)
        loss=read(run/(arm+'_loss_audit.json'))
        assert loss['optimizer_steps']==0 and loss['whitening_calls']==1 and loss['loss_calls']==32
        assert loss['valid_tokens']==sum(lengths)
        assert all(r['advantage_retained'] and r['normalization_calls']==1 for r in loss['loss_trace'])
        values=t['advantages'][t['mask'].bool()]
        checks[arm]=dict(max_advantage_error=float((t['advantages']-expected_adv).abs().max()),
            mean=float(values.mean()),unbiased_std=float(values.std()),task_reward_sum=float(t['task_rewards'].sum()),
            gradient_norm=loss['gradient_norm'],whitening_calls=1,loss_rows=32)
    edge=read(run/'native_edge_forwards.json')
    peaks=read(run/'nvml_peaks.json')
    ownpeak=max(p['own_process_bytes'] for p in peaks.values())
    assert ownpeak>0
    summary=dict(passed=True,standalone_P_VR_without_PRM_verified=True,separator_position_corrections=shifted,
        actual_generation_reused=run.name.endswith('_v3'),real_rows=len(rows),real_tokens=sum(lengths),real_steps=steps,
        terminated=sum(r['terminated'] for r in rows),truncated=sum(not r['terminated'] for r in rows),
        original_parameter_hashes_unchanged=True,optimizer_steps=0,checks=checks,
        measured_peak_own_process_GiB=ownpeak/1024**3,
        measured_peak_torch_allocated_GiB=max(s['peak_allocated_bytes'] for s in result['stages'].values())/1024**3,
        measured_peak_device_total_GiB=max(p['device_used_bytes'] for p in peaks.values())/1024**3,
        empty_response_positive_probability=edge['scores'][0]['step_scores'],
        edge_forward_probabilities=[s['step_scores'] for s in edge['scores']],
        run_result_sha256=sha(run/'result.json'),real_signal_file_sha256=sha(run/'signals.pt'))
    atomic(HERE/'real_acceptance.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
