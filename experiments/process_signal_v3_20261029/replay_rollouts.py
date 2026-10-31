"""Compare four signal rules on one immutable scored rollout file, CPU only."""
import argparse,json,hashlib
from pathlib import Path
from signals import SignalConfig,construct_signal
from diagnostics import summarize_signal

C_SHA='4917be38966c8438f58e283eb06c7fa818fed2bb13b726152078c19af013c356'
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.input.resolve();output=a.output.resolve()
    if source==output or output.exists():raise ValueError('Do not overwrite input or prior replay')
    records=[json.loads(x) for x in source.open() if x.strip()]
    if not records or len(records)%8:raise ValueError('Only complete stored groups accepted')
    config=json.loads((source.parent/'run_config.json').read_text())
    if config.get('C_adapter_sha256')!=C_SHA:raise ValueError('This replay requires the fixed adapted C scorer provenance')
    for r in records:
        for key in ['problem_id','outcome_reward','min_score','valid_for_process_reward','completion','completion_tokens','terminated']:
            if key not in r:raise ValueError('Missing rollout field: '+key)
        if r['outcome_reward'] and not r['valid_for_process_reward']:
            raise ValueError('No PRM score for a correct response; do not treat unscored A outputs as zero-score C outputs')
    results={}
    y=[r['outcome_reward'] for r in records];m=[r['min_score'] for r in records]
    for mode in ['outcome','mixed','papo','bounded']:
        z=construct_signal(SignalConfig(mode),y,None if mode=='outcome' else m,
            process_scored=None if mode=='outcome' else [r['valid_for_process_reward'] for r in records],
            group_ids=[r['problem_id'] for r in records])
        results[mode]=summarize_signal(z,records)
    receipt=dict(scope='counterfactual signal-only replay of the SAME stored responses, not four trained policies',
        input=str(source),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scorer_C_adapter_sha256=C_SHA,rows=len(records),metrics=results,
        counted_completion_tokens=sum(r['completion_tokens'] for r in records),
        stored_PRM_input_tokens=sum(r.get('prm_input_tokens',0) for r in records),
        model_calls=0,optimizer_steps=0,source_unmodified=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='metrics'},indent=2))

if __name__=='__main__':main()
