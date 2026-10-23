"""Paired frozen Qwen7B PRM inference, restricted to affected native microbatches."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from repairs import *


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--resume",action="store_true")
    args=cli.parse_args();out=args.out
    assert not (out/"scoring_complete.json").exists(),"Already scored: do not repeat"
    lock=(out/"inference.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((out/"preparation.json").read_text())
    inputs=Path(plan["inputs"])
    for item in plan["sources"].values():assert sha256(inputs/item["relative"])==item["sha256"]
    for name,receipt in json.loads((out/"protection_manifest.json").read_text()).items():
        assert sha256(inputs/name)==receipt["sha256"],name
    existing=list((out/"paired_batches").glob("*.json")) if (out/"paired_batches").exists() else []
    assert not existing or args.resume,"Partial inference exists; explicit resume required"
    devices=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    active=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    busy={line.split(',')[0].strip() for line in active.splitlines()}
    idle=[line.split(',')[0].strip() for line in devices.splitlines() if line.split(',')[1].strip() not in busy]
    assert idle,"No idle GPU: do not interrupt other workers"
    os.environ.update(CUDA_VISIBLE_DEVICES=idle[0],HF_HOME=str(ROOT/"cache/huggingface"),
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info()[0]>55*1024**3
    sys.path.insert(0,str(ROOT/"experiments/b_qwen7b_6m_20261021"))
    from frozen_prm import FrozenQwenPRM
    from core import encode_qwen
    start=time.monotonic()
    scorer=FrozenQwenPRM(ROOT/"models/qwen25_math_prm_7b",receipt_dir=out/"loader",microbatch_size=4)
    data=read_jsonl(inputs/plan["sources"]["B7_train"]["relative"])
    questions={r["problem_id"]:r["problem"] for r in read_jsonl(inputs/"data/prepared/train.jsonl")}
    plan_sha=sha256(out/"preparation.json")
    costs={"forward_calls":0,"unpadded_tokens":0,"padded_tokens":0,"repeat_calls":0}
    replayed=[];maximum_historical=0.;unchanged_max=0.;repeat_max=0.;position_checks=0
    torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def forward(encoded,length,repeat=False):
        x=scorer.tokenizer.pad([{"input_ids":e["input_ids"]} for e in encoded],padding="max_length",max_length=length,return_tensors="pt").to(scorer.device)
        output=scorer.model(**x,use_cache=False)
        logits=output.logits if hasattr(output,"logits") else output[0]
        assert logits.shape[:2]==x.input_ids.shape and logits.shape[-1]==2 and torch.isfinite(logits).all()
        result=[logits[i,e["positions"]].float().softmax(-1)[:,1].cpu().tolist() for i,e in enumerate(encoded)]
        costs["forward_calls"]+=1;costs["unpadded_tokens"]+=sum(len(e["input_ids"]) for e in encoded)
        costs["padded_tokens"]+=x.input_ids.numel();costs["repeat_calls"]+=int(repeat)
        return result

    for batch_id,block in enumerate(plan["affected_microbatches"]):
        path=out/"paired_batches"/f"{batch_id:04d}.json"
        if path.exists():
            b=json.loads(path.read_text());assert b["plan_sha256"]==plan_sha and b["indices"]==block["indices"]
            for k in costs:costs[k]+=b["costs"][k]
        else:
            before=dict(costs);old=[];new=[]
            for i in block["indices"]:
                r=data[i];q=questions[r["problem_id"]]
                a=encode_qwen(scorer.tokenizer,q,r["completion"],normalize=False)
                z=encode_qwen(scorer.tokenizer,q,r["completion"],normalize=True)
                assert len(a["steps"])==r["step_count"]==len(r["step_scores"])
                assert len(a["input_ids"])==r["prm_input_tokens"]
                assert len(a["positions"])==len(a["steps"]) and len(z["positions"])==len(z["steps"])
                assert max(len(a["input_ids"]),len(z["input_ids"]))<=scorer.max_length
                old.append(a);new.append(z)
            # Same partners, ordering, BF16 backend, and padding shape isolate the input change.
            length=max(len(e["input_ids"]) for e in old+new)
            old_scores=forward(old,length);new_scores=forward(new,length)
            repeated=forward(old,length,True) if batch_id%128==0 else old_scores
            records=[]
            for i,a,z,pa,pz,pr in zip(block["indices"],old,new,old_scores,new_scores,repeated):
                same=a["input_ids"]==z["input_ids"] and a["positions"]==z["positions"]
                delta=max(abs(x-y) for x,y in zip(pa,pz)) if same else None
                if same:assert delta<=1e-6
                for text,span in zip(z["steps"],z["spans"]):
                    assert data[i]["completion"][span["raw_start"]:span["raw_end"]].replace("\r\n","\n")==text
                records.append({"source_row":i,"input_changed":not same,"matched_unchanged_max":delta,
                    "historical_max_step_difference":max(abs(x-y) for x,y in zip(pa,data[i]["step_scores"])),
                    "repeat_max_difference":max(abs(x-y) for x,y in zip(pa,pr)),
                    "old":{"step_scores":pa,"step_count":len(pa),"min_score":min(pa),"input_tokens":len(a["input_ids"])},
                    "normalized":{"step_scores":pz,"step_count":len(pz),"min_score":min(pz),"input_tokens":len(z["input_ids"]),
                        "raw_sha256":z["raw_sha256"],"spans":z["spans"]}})
            b={"plan_sha256":plan_sha,"indices":block["indices"],"rows":records,"padding_length":length,
                "costs":{k:costs[k]-before[k] for k in costs}}
            atomic_json(path,b)
        replayed.extend(b["rows"])
        maximum_historical=max(maximum_historical,max(r["historical_max_step_difference"] for r in b["rows"]))
        unchanged_max=max(unchanged_max,max((r["matched_unchanged_max"] or 0) for r in b["rows"]))
        repeat_max=max(repeat_max,max(r["repeat_max_difference"] for r in b["rows"]))
        position_checks+=sum(r["old"]["step_count"]+r["normalized"]["step_count"] for r in b["rows"])
        if batch_id%32==0 or batch_id+1==len(plan["affected_microbatches"]):
            state={"completed_microbatches":batch_id+1,"total_microbatches":len(plan["affected_microbatches"]),
                "paired_rows":len(replayed),"costs":costs,"wall_seconds":time.monotonic()-start}
            atomic_json(out/"scoring_progress.json",state);print(json.dumps(state),flush=True)
    scorer.assert_frozen();torch.cuda.synchronize()
    assert len({r["source_row"] for r in replayed})==len(replayed)==plan["paired_scoring_rows"]
    for name,receipt in json.loads((out/"protection_manifest.json").read_text()).items():
        assert sha256(inputs/name)==receipt["sha256"],name
    write_jsonl(out/"paired_scores.jsonl",sorted(replayed,key=lambda x:x["source_row"]))
    atomic_json(out/"scoring_complete.json",{"complete":True,"no_training":True,"actor_rollouts":0,
        "PRM_parameter_updates":0,"model_frozen":True,"physical_gpu":idle[0],"plan_sha256":plan_sha,
        "model_revision":"0610740060112df12585d00a1c5f4624d2f59051","paired_rows":len(replayed),
        "affected_microbatches":len(plan["affected_microbatches"]),"native_positions_checked":position_checks,
        "maximum_historical_step_difference":maximum_historical,"maximum_unchanged_input_difference":unchanged_max,
        "maximum_repeat_difference":repeat_max,"costs":costs,"wall_seconds":time.monotonic()-start,
        "peak_allocated_GiB":torch.cuda.max_memory_allocated()/1024**3,
        "peak_reserved_GiB":torch.cuda.max_memory_reserved()/1024**3,"original_331_inputs_unchanged":True,
        "paired_scores_sha256":sha256(out/"paired_scores.jsonl")})
    print("Frozen paired scoring complete",len(replayed),flush=True)


if __name__=="__main__": main()
