"""Recheck all four validation prediction files with the frozen answer verifier."""
from pathlib import Path
import sys,json,hashlib,time

ROOT=Path('/path/to/ece285_math_rl_project');RUN=ROOT/'runs/Tree_credit_triplet_seed17_48_20261108'
AUDIT=ROOT/'audits/Tree_credit_triplet_seed17_48_20261108'
sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'))
from repairs import repaired_outcome


def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.open() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    contract=read(RUN/'evaluation/contract.json');start=time.time()
    assert contract['dataset']=='validation' and contract['temperature']==0 and contract['max_tokens']==2048
    path=ROOT/'data/prepared/validation.jsonl'
    assert sha(path)==contract['dataset_sha256']
    assert sha(ROOT/'experiments/reward_repair_v3_20261021/repairs.py')==contract['repair_sha256']
    units_path=ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json'
    assert sha(units_path)==contract['unit_metadata_sha256']
    units=read(units_path);data=lines(path);byid={r['problem_id']:r for r in data}
    mismatches=[];counts={}
    for arm in ('source','chain_cb','tree_vr','tree_cb'):
        rr=lines(RUN/'evaluation'/arm/'predictions.jsonl');assert len(rr)==500
        assert [r['problem_id'] for r in rr]==[q['problem_id'] for q in data]
        correct=0
        for r in rr:
            q=byid[r['problem_id']]
            assert r['gold']==q['answer'] and r['question_sha256']==q['question_sha256']
            assert r['generated_tokens']==len(r['completion_token_ids'])<=2048
            value=repaired_outcome(r['completion'],q['answer'],r['finish_reason']=='stop',
                    question=q['problem'],problem_id=q['problem_id'],unit_metadata=units)
            correct+=value['reward']
            if (value['reward'],value['status'])!=(r['reward'],r['status']):
                mismatches.append({'arm':arm,'problem_id':q['problem_id'],'original':{'reward':r['reward'],'status':r['status']},'replay':value})
        counts[arm]={'n':len(rr),'correct':correct}
    out={'passed':not mismatches,'records_rechecked':2000,'mismatches':mismatches,'counts':counts,
         'seconds':time.time()-start,'source_verifier_unchanged':True,'predictions_modified':False}
    target=AUDIT/'verifier_replay.json';assert not target.exists()
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    assert out['passed'],'Preserve mismatches and inspect; do not silently replace original rewards'


if __name__=='__main__':main()
