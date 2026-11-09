"""Post-completion CPU verification. Does not import the training constructor."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch
ROOT=Path('/path/to/ece285_math_rl_project')
RUN=ROOT/'runs/Budget_extension_seed17_12M_20261108'
AUDIT=ROOT/'audits/Budget_extension_seed17_12M_20261108'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not torch.cuda.is_initialized() and read(RUN/'controller.json')['stage']=='complete'
    assert read(RUN/'comparison.json')['passed']
    out=AUDIT/'independent_acceptance.json';assert not out.exists()
    source=read(AUDIT/'source_checkpoints.json')
    for arm,files in source.items():
        for n,h in files.items():assert sha(ROOT/'runs/ABC_seed17_resumed_V1_20261104'/arm/'checkpoint-288'/n)==h
    schedule=read(AUDIT/'preflight.json')['first_epoch_question_groups'];hashes={};arms={}
    assets=read(ROOT/'receipts/assets.json')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
    for arm in ['A','C_bounded']:
        path=RUN/arm;rr=rows(path/'rollouts.jsonl');assert len(rr)==9216
        error=0.;tokens=0;optional_ratio_batches=0
        for offset,step in enumerate(range(288,576)):
            file=path/f'raw_signals_step-{step}.pt';raw=torch.load(file,map_location='cpu',weights_only=False)
            hashes[str(file.relative_to(ROOT))]=sha(file)
            sub=rr[offset*32:offset*32+32]
            assert [x['problem_id'] for x in sub[::8]]==schedule[step]
            mask=raw['completion_mask'].bool();ids=raw['completion_ids'];actual=raw['advantages'].double().numpy()
            assert actual.shape==(32,) and raw['advantages'].dtype==torch.float32
            y=np.array([r['outcome_reward'] for r in sub],dtype=np.float64).reshape(4,8)
            m=np.array([r['min_score'] for r in sub],dtype=np.float64).reshape(4,8)
            outcome=(y-y.mean(1,keepdims=True))/(y.std(1,ddof=1,keepdims=True)+1e-4)
            mu=np.divide((m*y).sum(1,keepdims=True),y.sum(1,keepdims=True),out=np.zeros((4,1)),where=y.sum(1,keepdims=True)!=0)
            expected=outcome if arm=='A' else outcome+.2*y*(m-mu)
            error=max(error,float(abs(expected.ravel()-actual).max()));assert error<2e-6
            assert not (actual[y.ravel()==0]>0).any()
            assert raw.get('old_per_token_logps') is None and raw.get('importance_sampling_ratio') is None
            for i,r in enumerate(sub):
                full=ids[i][mask[i]].tolist();assert full==r['completion_token_ids']
                assert len(full)==r['completion_tokens'] and 0<len(full)<=2048
                assert r['terminated']==(tok.eos_token_id in full)
                if r['terminated']:assert full[-1]==tok.eos_token_id
                assert int(raw['pilot_trace_id'][i])==r['tag']==step*32+i
                assert float(actual[i])==r['advantage']
                tokens+=len(full)
        source_cp=ROOT/'runs/ABC_seed17_resumed_V1_20261104'/arm/'checkpoint-288'
        old=read(source_cp/'completion_accounting.json');final=read(path/'stage_576_result.json')['accounting']
        assert final['generated_completion_tokens']-old['generated_completion_tokens']==tokens
        assert final['loss_rows']-old['loss_rows']==9216
        for target in [432,576]:
            assert read(path/f'first_update_acceptance_{target}.json')['passed']
            cp=path/f'milestones/checkpoint-{target}';contract=read(cp/'pilot_checkpoint.json')
            for n,h in contract['files_sha256'].items():assert sha(cp/n)==h
        arms[arm]={'batches':288,'responses':9216,'new_generated_tokens':tokens,'max_float64_advantage_error':error,
            'source_states_unchanged':True,'optional_sampling_ratio_reconstruction_applicable':False,
            'note':'Sampling correction disabled and no cached old log-probabilities; no claim of ratio reconstruction.'}
    sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'));from repairs import repaired_outcome
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json');verified=0;repeat_stats={}
    for cohort in ['baseline','nine','final','endpoints_math','endpoints_gsm']:
        dataset='math500' if cohort=='endpoints_math' else 'gsm8k' if cohort=='endpoints_gsm' else 'validation'
        data=rows(ROOT/f'data/prepared/{dataset}.jsonl');byid={r['problem_id']:r for r in data}
        model_rows={}
        for repeat in sorted((RUN/'evaluation'/cohort).glob('repeat_*')):
            contract=read(repeat/'contract.json');assert contract['dataset_sha256']==sha(ROOT/f'data/prepared/{dataset}.jsonl')
            for model,meta in read(repeat/'summary.json')['summary'].items():
                path=repeat/model/'predictions.jsonl';assert sha(path)==meta['predictions_sha256'];rr=rows(path)
                assert len(rr)==len(data) and [r['problem_id'] for r in rr]==[r['problem_id'] for r in data]
                for r in rr:
                    q=byid[r['problem_id']];assert q['question_sha256']==r['question_sha256'] and q['answer']==r['gold']
                    assert len(r['completion_token_ids'])==r['generated_tokens']
                    judged=repaired_outcome(r['completion'],r['gold'],r['finish_reason']=='stop',question=q['problem'],problem_id=r['problem_id'],unit_metadata=units)
                    assert all(judged[k]==r[k] for k in ['reward','status','answer'])
                verified+=len(rr);model_rows.setdefault(model,[]).append(rr)
        repeat_stats[cohort]={}
        for model,repeats in model_rows.items():
            arr=np.array([[r['reward'] for r in rr] for rr in repeats])
            repeat_stats[cohort][model]={'n_questions':arr.shape[1],'repeats':len(repeats),
                'accuracy_per_repeat':arr.mean(1).tolist(),'mean_accuracy':float(arr.mean()),
                'binary_flips_between_two_repeats':int((arr[0]!=arr[1]).sum()) if len(repeats)==2 else None}
    assert verified==15276
    children=read(RUN/'children.json')
    frozen=next(x for x in children if x['stage']=='freeze_seed42_eligibility')
    first_test=min(x['started_unix'] for x in children if x['stage'].startswith('common_endpoints_'))
    assert frozen['finished_unix']<first_test
    frozen_lines=[s for s in (RUN/'freeze_seed42_eligibility.log').read_text().splitlines() if s.startswith('{')]
    assert json.loads(frozen_lines[-1])==read(RUN/'seed42_eligibility.json')
    protected=read(AUDIT/'protected_before.json')
    for name,h in protected.items():assert sha(ROOT/name)==h
    assert not torch.cuda.is_initialized()
    out.write_text(json.dumps({'passed':True,'CUDA_initialized':False,'arms':arms,'predictions_regraded':verified,
        'fixed_verifier_matches':True,'repeatability':repeat_stats,'protected_files_verified':len(protected),
        'seed42_gate_log_frozen_before_tests':True,
        'raw_tensor_sha256':hashes,'training_constructor_imported':False},indent=2)+'\n')
    print(json.dumps({'passed':True,'arms':arms,'predictions_regraded':verified,'protected_files_verified':len(protected)}))
if __name__=='__main__':main()
