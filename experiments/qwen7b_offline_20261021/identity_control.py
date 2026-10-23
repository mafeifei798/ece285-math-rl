"""CPU negative control: two identical score copies must yield zero differences."""
import json
from pathlib import Path
import shutil
import sys
import torch
from core import atomic_json


def main():
    root,out=map(lambda x:Path(x).resolve(),sys.argv[1:3])
    control=out/'analysis_identity_control'
    if control.exists():raise RuntimeError('Preserve existing identity-control result')
    control.mkdir()
    for name in ['input_manifest.json','reviewed_records.json','encoding_h4.json','receipt_h4.json','scores_h4.jsonl']:
        shutil.copyfile(out/name,control/name)
    for old,new in [('encoding_h4.json','encoding_qwen7b.json'),('receipt_h4.json','receipt_qwen7b.json'),('scores_h4.jsonl','scores_qwen7b.jsonl')]:
        shutil.copyfile(out/old,control/new)
    (control/'README.txt').write_text('IDENTITY NEGATIVE CONTROL ONLY. Both scorer inputs are the same H4 scores. These are not Qwen predictions. CPU algebra only. Not part of the measured model comparison.\n')
    torch.set_num_threads(1)
    import trl_kernel
    original=trl_kernel.actual_trl_advantages
    def cpu(values,source_path,device=None):return original(values,source_path,device='cpu')
    trl_kernel.actual_trl_advantages=cpu
    import analyze
    sys.argv=['analyze.py',str(root),str(control)]
    analyze.main()
    summary=json.loads((control/'summary.json').read_text())
    for key in ['model_only_raw','model_only_repaired']:
        result=summary['comparisons'][key]
        for metric in ['reward_order_changed_groups','advantage_sign_flips','advantage_changed_rows_gt_0p01','max_abs_advantage_change']:
            assert result[metric]==0,(key,metric,result[metric])
    random=summary['review']['uniform_random']
    assert random['clean_solutions']==7 and random['erroneous_solutions']==20
    assert random['h4']['clean_solutions_falsely_rejected_at_0p5']==6
    assert random['h4']['erroneous_solutions_fully_passed_at_0p5']==3
    assert random['h4']==random['qwen7b']
    atomic_json(out/'identity_control_QA.json',{'passed':True,'CPU_only':True,
        'two_identical_score_sets_produce_zero_reward_rank_advantage_differences':True,
        'prior_random_review_denominators_reproduced':{'clean':7,'errors':20,'clean_rejected':6,'errors_fully_passed':3},
        'control_is_not_Qwen_predictions':True,'new_training':False})


if __name__=='__main__':main()
