"""Outcome-only comparator: same verifier/status/IDs, no PRM model or forward."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'))
from repairs import repaired_outcome,valid_for_prm


class OutcomeLedger:
    def __init__(self,units):
        self.unit_metadata=units;self.records=[];self.cost={}

    def callback(self,eos_token_id,owner):
        def outcome(prompts,completions,completion_ids,answer,problem_id,problem,**unused):
            self.records=[];self.cost={'prm_input_tokens':0,'prm_forward_seconds':0.,'prm_model_loaded':False}
            for p,c,ids,gold,qid,q in zip(prompts,completions,completion_ids,answer,problem_id,problem):
                terminated=eos_token_id in ids
                r=repaired_outcome(c,gold,terminated,question=q,problem_id=qid,unit_metadata=self.unit_metadata)
                self.records.append(dict(optimizer_step_before_update=owner['trainer'].state.global_step,
                    problem_id=qid,gold=gold,completion=c,completion_tokens=len(ids),terminated=terminated,
                    status=r['status'],answer=r['answer'],outcome_reward=r['reward'],
                    valid_for_process_reward=valid_for_prm(r),step_scores=[],min_score=0.,step_count=0,
                    prm_input_tokens=0,raw_no_double_newline='\n\n' not in c.strip(),
                    reward=r['reward'],completion_token_ids=list(ids),prompt_text=p))
            return [r['outcome_reward'] for r in self.records]
        return outcome
