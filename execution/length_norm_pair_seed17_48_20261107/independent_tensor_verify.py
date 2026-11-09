"""CPU verifier to execute remotely only AFTER completion; no training imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib
import torch

ROOT=Path('/path/to/ece285_math_rl_project')
RUN=ROOT/'runs/Length_norm_pair_seed17_48_20261107'


def main():
    assert json.loads((RUN/'controller.json').read_text())['stage']=='complete'
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    expected_source=globals().get('EXPECTED_SOURCE');assert expected_source,'Use verify_remote_completed.py to supply the pre-run source hashes'
    cp=ROOT/'runs/ABC_seed17_resumed_V1_20261104/C_bounded/checkpoint-288'
    for name,h in expected_source.items():assert hashlib.sha256((cp/name).read_bytes()).hexdigest()==h,('Source checkpoint changed',name)
    recovery=json.loads((RUN/'recovery_launch_receipt.json').read_text())
    recovery_path=Path(recovery['command'][-1]);assert recovery_path.is_relative_to(ROOT/'experiments')
    assert hashlib.sha256(recovery_path.read_bytes()).hexdigest()==recovery['recovery_controller_sha256']
    results={}
    for arm in ['grpo','fixed_length']:
        rr=[json.loads(s) for s in (RUN/arm/'rollouts.jsonl').read_text().splitlines()]
        assert len(rr)==1536
        error=0.;token_rows=0
        for step in range(288,336):
            group=rr[(step-288)*32:(step-287)*32]
            assert all(r['optimizer_step_before_update']==step for r in group)
            raw=torch.load(RUN/arm/f'raw_step_{step}.pt',map_location='cpu',weights_only=False)
            adv=raw['advantages'];mask=raw['completion_mask'].bool()
            assert adv.dtype==torch.float32 and adv.shape==mask.shape
            y=torch.tensor([r['outcome_reward'] for r in group],dtype=torch.float64).view(4,8)
            m=torch.tensor([r['min_score'] for r in group],dtype=torch.float64).view(4,8)
            centered=y-y.mean(1,keepdim=True)
            sd=((centered.square().sum(1,keepdim=True))/7).sqrt()
            outcome=centered/(sd+1e-4)
            cm=(m*y).sum(1,keepdim=True)/y.sum(1,keepdim=True).clamp_min(1)
            scalar=(outcome+.2*y*(m-cm)).flatten()
            expected=scalar[:,None].expand_as(adv)*mask
            error=max(error,float((adv.double()-expected).abs().max()))
            assert not adv[~mask].any() and not (adv[y.flatten()==0]>0).any()
            for i,r in enumerate(group):
                assert raw['completion_ids'][i][mask[i]].tolist()==r['completion_token_ids']
                assert int(mask[i].sum())==r['completion_tokens'];token_rows+=1
        assert error<2e-5
        results[arm]={'rows':token_rows,'raw_batches':48,'max_original_bounded_advantage_error':error,
                      'raw_tokens_and_masks_match_JSON':True,'wrong_positive_tokens':0,'padding_advantage_zero':True}
    assert not torch.cuda.is_initialized()
    return {'passed':True,'arms':results,'CUDA_initialized':False,'GPU_work':False,
            'source_checkpoint_files_verified':len(expected_source),'all_source_checkpoint_files_unchanged':True,'recovery_controller_unchanged':True,
            'scope':'Independent float64 bounded-advantage recomputation and raw emitted token/mask check. No regrading or model calls.'}


if __name__=='__main__':print(json.dumps(main()))
