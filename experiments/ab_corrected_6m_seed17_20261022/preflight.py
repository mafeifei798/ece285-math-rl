"""Read-only source/data/verifier preflight; no model construction."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from pair_utils import HERE, ROOT, read, sha, read_jsonl, verify_sources, atomic


def main():
    release=verify_sources();plan=read(HERE/"plan.json")
    for p in HERE.glob("*.py"):ast.parse(p.read_text())
    old=ast.parse((ROOT/"training/evaluate_policy.py").read_text())
    new=ast.parse((HERE/"evaluate.py").read_text())
    def configs(tree):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {"LLM","SamplingParams"}]
    assert configs(old)==configs(new),"Evaluation generation configuration changed"
    from train_arm import OutcomeLedger
    from trainer import trainer_class
    cls=trainer_class()
    units=read(ROOT/"audits/reward_repair_v3_20261021/unit_metadata.json")
    train={r["problem_id"]:r for r in read_jsonl(ROOT/"data/prepared/train.jsonl")}
    data=read_jsonl(ROOT/"runs/A_seed17/rollouts.jsonl")
    # Includes all nine known A training false negatives and initial mixed groups.
    indices=list(range(32))+[2096,4822,5365,8513,8517,8519,8749,8750,8751]
    batch=[data[i] for i in indices]
    class IDs:
        def __init__(self,r):self.row=r
        def __len__(self):return self.row["completion_tokens"]
        def __contains__(self,key):return self.row["terminated"]
    ledger=OutcomeLedger(units)
    callback=ledger.callback(42,{"trainer":SimpleNamespace(state=SimpleNamespace(global_step=0))})
    values=callback(prompts=[train[r["problem_id"]]["prompt"] for r in batch],
        completions=[r["completion"] for r in batch],completion_ids=[IDs(r) for r in batch],
        answer=[r["gold"] for r in batch],problem_id=[r["problem_id"] for r in batch],
        problem=[train[r["problem_id"]]["problem"] for r in batch])
    assert values[-9:]==[1.]*9
    assert values[:32]==[r["reward"] for r in batch[:32]]
    assert not any(r["step_scores"] for r in ledger.records) and ledger.cost["prm_scored_completions"]==0
    manifest=read(ROOT/"data/prepared/manifest.json")
    for name in ["train.jsonl","math500.jsonl","gsm8k.jsonl"]:
        assert sha(ROOT/"data/prepared"/name)==manifest["files"][name]["sha256"]
    receipt=dict(passed=True,source_files=len(release["source_hashes"]),installed_class=cls.__name__,
        A_real_record_regressions=len(batch),A_known_false_negatives_corrected=9,
        evaluation_generation_config_unchanged=True,shared_corrected_verifier=True,
        original_reward_and_loss=True,models_constructed=0,optimizer_steps=0)
    output=ROOT/"audits"/plan["experiment"]/"preflight.json"
    atomic(output,receipt);print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":main()
