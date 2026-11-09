"""Finish only the previously authorized frozen-answer scoring and CPU analysis."""
from pathlib import Path
import sys
import os
import json
import time
import hashlib
import subprocess
import fcntl
import traceback
import math

ROOT = Path('/path/to/ece285_math_rl_project')
SOURCE = ROOT/'experiments/cd_prm_7b_pair_20261028'
RUN = ROOT/'runs/CD_PRM7B_pair_seed17_20261028'
OUT = RUN/'wiring_recovery'
sys.path.insert(0,str(SOURCE))
from common import read, sha, atomic, verify_release, protect


def main():
    lock=(OUT/'postprocess.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'postprocess.json').exists(), 'Inspect existing postprocessing before restarting'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    reference=read(OUT/'fp32_reference_v2/gradient_replay.json')
    replay=read(OUT/'gradient_replay.json')
    assert reference['strict_identity_check_passed'] and reference['all_parameters_unchanged']
    assert reference['optimizer_updates']==replay['optimizer_updates']==0
    assert reference['model_release_sha256']==replay['model_release_sha256']==verify_release()
    old=read(RUN/'controller.json');assert old['status']=='failed'
    assert old['completed_stages'][-2:]==['adapt_C','adapt_D']
    assert not Path('/proc',str(old['pid'])).exists()
    init={a:read(RUN/a/'initialization.json') for a in ['C','D']}
    for key in ['initial_adapter_hash','prm_revision','batch_manifest_sha256','steps','config_sha256','trainable_parameters']:
        assert init['C'][key]==init['D'][key],key
    completion={a:read(RUN/a/'completion.json') for a in ['C','D']}
    for a,v in completion.items():
        assert v['steps_completed']==512 and v['policy_optimizer_updates']==0 and v['frozen_parameters_unchanged']
        assert v['adapter_sha256']==sha(RUN/a/'adapter/adapter_model.safetensors')
    first={a:json.loads(next((RUN/a/'metrics.jsonl').open())) for a in ['C','D']}
    assert first['C']['losses']==first['D']['losses']
    assert math.isclose(first['C']['total_loss']-first['D']['total_loss'],first['C']['losses']['success'],abs_tol=1e-12)
    reassessment=dict(original_BF16_strict_coordinate_check=read(RUN/'paired_wiring.json'),
      deterministic_FP32_reference=reference, repeated_BF16_no_update=replay,
      conclusion='Original coordinate-wise test is not robust to observed BF16 backward nondeterminism. Strict identity passed unchanged tolerance in deterministic FP32 reference. No training, loss, or saved adapter changed.',
      unverified='No claim that every numerical kernel or every later gradient is bitwise deterministic.',
      read_only_scoring_min_free_MiB=24000,
      memory_admission_reason='Measured PRM training peak15.232GiB; read-only evaluation uses no optimizer/backward. Require24GiB free, without modifying original training config.')
    atomic(OUT/'reassessment.json',reassessment)
    receipt=dict(status='running',pid=os.getpid(),started_unix=time.time(),source_release_sha256=verify_release(),
      completed_stages=[],active_child=None,policy_optimizer_updates=0,PRM_optimizer_updates=0,
      source_sha256=sha(Path(__file__)),original_controller_unchanged=True)
    atomic(OUT/'postprocess.json',receipt)
    try:
        for name,args,gpu in [('score_B',['score.py','B'],True),('score_C',['score.py','C'],True),('score_D',['score.py','D'],True),('analyze',['analyze.py'],False)]:
            verify_release()
            if gpu:
                deadline=time.time()+7200
                while int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<24000:
                    receipt.update(status='waiting_for_memory',next_stage=name);atomic(OUT/'postprocess.json',receipt)
                    assert time.time()<deadline;time.sleep(30)
            with (OUT/(name+'.log')).open('a') as log:
                child=subprocess.Popen([sys.executable,'-B','-u',str(SOURCE/args[0]),*args[1:]],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                receipt.update(status='running',active_child=dict(stage=name,pid=child.pid),stage_started_unix=time.time());atomic(OUT/'postprocess.json',receipt)
                code=child.wait(timeout=7200)
            assert code==0,(name,code)
            receipt['completed_stages'].append(name);receipt['active_child']=None;atomic(OUT/'postprocess.json',receipt)
        # Preserve the failed original check, attach the independent reassessment.
        summary=read(RUN/'summary.json');summary['wiring_reassessment']=reassessment
        summary['limitations'].append('BF16 backward was not bitwise reproducible; the original strict coordinate test failed, while a deterministic FP32 no-update reference passed its unchanged tolerance. The failed test and both audits are preserved.')
        atomic(RUN/'summary.json',summary)
        text='\n\nGradient verification recovery: original BF16 coordinate-wise test FAILED and remains preserved. Same-input BF16 repeated backward showed numerical variation. Deterministic FP32 no-update reference PASSED the original tolerance. No optimization or loss change was performed during recovery. See wiring_recovery/reassessment.json.\n'
        with (RUN/'RESULTS.txt').open('a') as f:f.write(text)
        import html
        with (RUN/'review.html').open('a') as f:f.write('<pre>'+html.escape(text)+'</pre>')
        receipt.update(status='complete',finished_unix=time.time(),preservation=protect(),
          summary_sha256=sha(RUN/'summary.json'),scores={a:read(RUN/f'scoring_{a}.json') for a in ['B','C','D']})
        atomic(OUT/'postprocess.json',receipt)
    except Exception as e:
        receipt.update(status='failed',failure=str(e),traceback=traceback.format_exc(),failed_unix=time.time());atomic(OUT/'postprocess.json',receipt);raise


if __name__=='__main__':main()
