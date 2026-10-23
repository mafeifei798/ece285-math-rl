"""Counterfactual reward/rank/advantage replay; all actor outputs stay fixed."""
import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import sys
from repairs import *
from repairs import proposed_bounded_advantages
from replay_offline import summarize_difference
from trl_kernel import actual_trl_advantages, save_provenance
from v2core import midranks


def kind(ys):
    return "all_correct" if sum(ys)==8 else "all_wrong" if sum(ys)==0 else "mixed"


def main():
    cli=argparse.ArgumentParser();cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--inputs",type=Path);args=cli.parse_args();out=args.out
    assert not (out/"replay_complete.json").exists(),"Already replayed: do not repeat"
    import torch
    torch.set_num_threads(4)
    plan=json.loads((out/"preparation.json").read_text());inputs=args.inputs or Path(plan["inputs"])
    scoring=json.loads((out/"scoring_complete.json").read_text());assert scoring["complete"]
    assert sha256(out/"paired_scores.jsonl")==scoring["paired_scores_sha256"]
    assert sha256(out/"unit_metadata.json")==plan["unit_metadata_sha256"]
    source=inputs/plan["sources"]["B7_train"]["relative"]
    assert sha256(source)==plan["sources"]["B7_train"]["sha256"]
    data=read_jsonl(source);labels=read_jsonl(out/"outcomes/B7_train.jsonl")
    paired={r["source_row"]:r for r in read_jsonl(out/"paired_scores.jsonl")}
    assert set(plan["changed_process_rows"]).issubset(paired)
    trl=Path(importlib.util.find_spec("trl").submodule_search_locations[0])/"trainer/grpo_trainer.py"
    save_provenance(trl,out/"trl_tensor_kernel.json")
    arrays={}
    for variant,(new_units,score_kind) in {"logged":(False,"logged"),"matched_legacy":(False,"old"),
            "units_only":(True,"old"),"crlf_only":(False,"normalized"),"combined":(True,"normalized")}.items():
        rows=[]
        for i,r in enumerate(data):
            label=labels[i]["new" if new_units else "old"];valid=valid_for_prm(label)
            if valid:
                s=paired[i][score_kind] if i in paired and score_kind!="logged" else r
                p=s["step_scores"]
            else:p=[]
            value=mixed_reward(label,p)
            if variant=="logged":assert abs(value-r["reward"])<1e-12
            rows.append({"source_row":i,"problem_id":r["problem_id"],"status":label["status"],
                "outcome_reward":label["reward"],"process_scored":valid,"min_score":min(p) if valid else None,
                "step_count":len(p),"reward":value,
                "score_origin":"paired_"+score_kind if i in paired and score_kind!="logged" else "historical_frozen_cache"})
        rewards,advantages=actual_trl_advantages([r["reward"] for r in rows],trl,"cpu")
        assert torch.isfinite(advantages).all()
        for r,v,a in zip(rows,rewards.tolist(),advantages.tolist()):r.update(reward_fp32=v,advantage=a)
        for start in range(0,len(rows),8):
            group=rows[start:start+8];ranks=midranks([r["reward_fp32"] for r in group])
            for r,rank in zip(group,ranks):r["reward_rank"]=rank
        arrays[variant]=rows;write_jsonl(out/"variants"/(variant+".jsonl"),rows)
    all_groups=set(range(len(data)//8))
    comparisons={}
    for name,old,new in [("backend_padding_control","logged","matched_legacy"),
        ("units_only","matched_legacy","units_only"),("crlf_only","matched_legacy","crlf_only"),
        ("combined_matched","matched_legacy","combined"),("combined_vs_historical","logged","combined")]:
        comparisons[name]=summarize_difference(arrays[old],arrays[new],all_groups)
    strata={}
    for outcome_type in ["mixed","all_correct","all_wrong"]:
        ids=[]
        for start in range(0,len(data),8):
            if kind([r["outcome_reward"] for r in arrays["matched_legacy"][start:start+8]])==outcome_type:ids.extend(range(start,start+8))
        strata[outcome_type]=summarize_difference(arrays["matched_legacy"],arrays["combined"],all_groups,ids)
    groups={}
    for variant in ["matched_legacy","combined"]:
        stats={}
        for start in range(0,len(data),8):
            group=arrays[variant][start:start+8];k=kind([r["outcome_reward"] for r in group])
            stats.setdefault(k,[]).extend(group)
        groups[variant]={k:{"groups":len(v)//8,"mean_abs_advantage":sum(abs(r["advantage"]) for r in v)/len(v)} for k,v in stats.items()}
    combined=arrays["combined"]
    outcomes=[r["outcome_reward"] for r in combined]
    minima=[r["min_score"] if r["process_scored"] else 0. for r in combined]
    proposed,process=proposed_bounded_advantages(outcomes,minima)
    guard={"status":"OFFLINE PROPOSAL ONLY; no trainer uses it", "formula":"A_outcome_GRPO+0.2*(m-mean_group(m))",
        "invalid_process_minimum":0,"group_size":8,"coefficient":.2,
        "absolute_process_component_bound":.2*7/8,"maximum_observed_absolute_process_component":float(process.abs().max()),
        "does_not_repair_PRM_ranking":True,"stats":{}}
    for k in ["mixed","all_correct","all_wrong"]:
        indices=[]
        for start in range(0,len(data),8):
            if kind(outcomes[start:start+8])==k:indices.extend(range(start,start+8))
        a=proposed[indices];old=[combined[i]["advantage"] for i in indices]
        guard["stats"][k]={"groups":len(indices)//8,"current_mean_abs_advantage":sum(abs(v) for v in old)/len(old),
            "candidate_mean_abs_advantage":float(a.abs().mean()),"candidate_max_abs_advantage":float(a.abs().max())}
        if k=="mixed":
            guard["stats"][k]["candidate_outcome_sign_violations"]=sum((float(proposed[i])>0)!=(outcomes[i]==1) for i in indices)
    assert guard["maximum_observed_absolute_process_component"]<=.175001
    assert guard["stats"]["mixed"]["candidate_outcome_sign_violations"]==0
    atomic_json(out/"guardrail_diagnostic.json",guard)
    case_rows=[42,806,1531,2017,2347,3049,4133,4472,4623,5353,5682,5946,6006,6350,6460,6626,7446,7512,7829,8745]
    cases=[]
    for order,i in enumerate(case_rows,1):
        entry={"sample_id":f"B{order:02d}","source_row":i,"variants":{k:v[i] for k,v in arrays.items()},
            "proposed_bounded_advantage":float(proposed[i])}
        cases.append(entry)
    atomic_json(out/"case_impacts.json",cases)
    a_data=read_jsonl(inputs/plan["sources"]["A_train"]["relative"]);a_labels=read_jsonl(out/"outcomes/A_train.jsonl")
    aa=[]
    for version in ["old","new"]:
        reward,adv=actual_trl_advantages([r[version]["reward"] for r in a_labels],trl,"cpu")
        aa.append([{"source_row":i,"problem_id":r["problem_id"],"outcome_reward":r[version]["reward"],
            "process_scored":False,"min_score":None,"step_count":0,"reward":r[version]["reward"],
            "reward_fp32":float(reward[i]),"advantage":float(adv[i])} for i,r in enumerate(a_labels)])
    a_impact=summarize_difference(aa[0],aa[1],set(range(len(a_data)//8)))
    atomic_json(out/"A_unit_impact.json",a_impact)
    for name,receipt in json.loads((out/"protection_manifest.json").read_text()).items():assert sha256(inputs/name)==receipt["sha256"]
    result={"complete":True,"rows":len(data),"groups":len(data)//8,"comparisons":comparisons,
        "combined_by_original_outcome_type":strata,"group_distributions":groups,"A_unit_impact":a_impact,
        "guardrail":guard,"source_inputs_unchanged":331,"new_training":False,"new_actor_samples":0,
        "PRM_parameter_updates":0,"scope":"Learning signals on fixed responses; not new trained-policy accuracy",
        "notes":["0.2, min aggregation, validity mask and original GRPO normalization unchanged in bug-fix variants.",
            "Backend/padding control is separate from matched input repair effects.",
            "More step boundaries can reduce min scores mechanically; score/rank changes alone are not improvements.",
            "The proposed bounded objective is separate and not enabled in these variants."]}
    atomic_json(out/"replay_complete.json",result)
    print(json.dumps({"comparisons":comparisons,"group_distributions":groups,"guardrail":guard},indent=2))


if __name__=="__main__":main()
