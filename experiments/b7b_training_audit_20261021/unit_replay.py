"""Narrow existing typed-percent repair; preserve all original rewards/results."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

root = Path.cwd()
out = root / "audits/b7b_training_audit_20261021"
meta = json.loads((out / "unit_metadata_compact.json").read_text())
sys.path.insert(0,str(root / "experiments/reward_repair_v2_20261021"))
from v2core import outcome_v2
read_rows = lambda p:[json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
lookup = {}
for dataset in ["train","math500","gsm8k"]:
    lookup.update({r["problem_id"]:r for r in read_rows(root / "data/prepared" / (dataset+".jsonl"))})
arms={"A":"A_seed17","B_old":"B_seed17_6M","B7":"B_Qwen7B_seed17_6M"}
sources=[]
for arm,name in arms.items():
    sources.append((arm,"train",root/"runs"/name/"rollouts.jsonl"))
    for summary in (root/"runs"/name).rglob("evaluation_summary.json"):
        for dataset in ["math500","gsm8k"]:
            sources.append((arm,dataset,summary.parent/(dataset+".jsonl")))
result={"scope":"Secondary offline verifier correction only; cannot undo any already-executed training update.",
        "policy":"Use previously fixed train question-unit metadata and question-only reviewed eval metadata. No substring percent inference from model outputs. math500/24 explicitly forbids a percent sign and is excluded from this repair.",
        "sources":{},"changes":[]}
for arm,dataset,path in sources:
    print("checking",arm,dataset,flush=True)
    data=read_rows(path)
    old_total=sum(r.get("outcome_reward",r["reward"]) for r in data)
    delta=0
    checked=0
    for i,r in enumerate(data):
        m=meta.get(r["problem_id"])
        if not m or m["answer_type"] not in {"percent_value","percentage_points"}:continue
        q=lookup[r["problem_id"]]
        assert hashlib.sha256(q["problem"].encode()).hexdigest()==m["question_text_sha256"]
        assert hashlib.sha256(r["gold"].encode()).hexdigest()==m["gold_text_sha256"]
        terminated=r["terminated"] if "terminated" in r else r["finish_reason"]=="stop"
        new=outcome_v2(r["completion"],r["gold"],terminated,question=q["problem"],metadata=m)
        old=r.get("outcome_reward",r["reward"])
        checked+=1
        if new["reward"]!=old:
            delta+=new["reward"]-old
            result["changes"].append({"arm":arm,"dataset":dataset,"source_row":i,"problem_id":r["problem_id"],
                "old":old,"new":new["reward"],"old_status":r["status"],"new_status":new["status"],
                "answer":r["answer"],"gold":r["gold"]})
    result["sources"][arm+"/"+dataset]={"n":len(data),"unit_rows_checked":checked,"old_correct":old_total,
        "new_correct":old_total+delta,"delta":delta,"old_accuracy":old_total/len(data),"new_accuracy":(old_total+delta)/len(data)}
    print(arm,dataset,json.dumps(result["sources"][arm+"/"+dataset]),flush=True)
result["completed"]=True
(out/"unit_replay.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
