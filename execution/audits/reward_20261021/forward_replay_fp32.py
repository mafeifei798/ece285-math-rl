"""Bounded CPU-only audit of saved PRM scores, padding, and causal prefix indexing."""
import json
import os
from pathlib import Path
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
ROOT, OUT = map(lambda p: Path(p).resolve(), sys.argv[1:3])
OUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(x) for x in (ROOT / "runs/B_seed17_6M/rollouts.jsonl").read_text().splitlines()]
train = {r["problem_id"]: r for r in map(json.loads, (ROOT / "data/prepared/train.jsonl").read_text().splitlines())}
assets = json.loads((ROOT / "receipts/B_prm_assets.json").read_text())

# Fixed strata chosen before forward computation; bounds protect the active GPU jobs.
ix = []
for target_y, lo, hi, single in [(1,0,300,False),(0,0,300,False),(1,300,600,False),(0,300,600,False),
                                (1,0,500,True),(0,0,500,True),(1,1000,1600,False),(0,1000,1600,False)]:
    candidates = [i for i,r in enumerate(rows) if r["valid_for_process_reward"] and r["outcome_reward"] == target_y
                  and lo <= r["prm_input_tokens"] < hi and (r["step_count"] == 1) == single and i not in ix]
    if candidates: ix.append(candidates[0])
ix.extend(i for i in [78,2095] if i not in ix)
tokenizer = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True, padding_side="right")
start = time.monotonic()
model = AutoModelForTokenClassification.from_pretrained(assets["model_path"], local_files_only=True,
                  torch_dtype=torch.float32, attn_implementation="sdpa").to("cpu").eval()
model.requires_grad_(False)
assert not torch.cuda.is_initialized()
print(json.dumps({"stage":"loaded_cpu", "seconds":time.monotonic()-start, "indices":ix}), flush=True)

encoded = []
for i in ix:
    row = rows[i]
    question = train[row["problem_id"]]["problem"]
    steps = [x.strip() for x in row["completion"].split("\n\n") if x.strip()]
    texts = ["\n\n".join([question]+steps[:j])+"\n\n" for j in range(1,len(steps)+1)]
    prefixes = [tokenizer(t, add_special_tokens=True)["input_ids"] for t in texts]
    full = prefixes[-1]
    assert all(full[:len(p)] == p for p in prefixes)
    assert len(full) == row["prm_input_tokens"]
    assert len(steps) == row["step_count"]
    encoded.append({"i":i,"ids":full,"positions":[len(p)-1 for p in prefixes],"prefixes":prefixes})


@torch.inference_mode()
def forward(batch):
    inputs = tokenizer.pad([{"input_ids":e["ids"]} for e in batch], padding=True, return_tensors="pt")
    logits = model(**inputs,use_cache=False).logits.float()
    assert torch.isfinite(logits).all()
    prob = logits.softmax(-1)[...,1]
    return [prob[i,e["positions"]].tolist() for i,e in enumerate(batch)]


report = {"prm_revision":assets["model_revision"],"device":"cpu","threads":4,"dtype":"float32",
          "note":"A bounded numerical plumbing check, not a process-accuracy benchmark; CPU FP32 and saved GPU BF16 kernels differ; CPU lacks native BF16.",
          "cases":[],"config": {"id2label":model.config.id2label,"label2id":model.config.label2id,
          "max_position_embeddings":model.config.max_position_embeddings,"padding_side":tokenizer.padding_side}}
batch_scores = []
for j in range(0,len(encoded),4):
    batch_scores.extend(forward(encoded[j:j+4]))
    print(json.dumps({"stage":"batch_done","n":len(batch_scores),"elapsed":time.monotonic()-start}),flush=True)
for e, bs in zip(encoded,batch_scores):
    individual = forward([e])[0]
    # At least one nonterminal prefix per multistep trace, to test future-token leakage.
    k = max(0, len(e["prefixes"])//2-1)
    prefix = e["prefixes"][k]
    ps = forward([{"ids":prefix,"positions":[len(prefix)-1]}])[0][0]
    saved = rows[e["i"]]["step_scores"]
    item = {"row_index":e["i"],"problem_id":rows[e["i"]]["problem_id"],"tokens":len(e["ids"]),
            "steps":len(saved),"saved_gpu_scores":saved,"batch_cpu_scores":bs,"single_cpu_scores":individual,
            "max_cpu_batch_vs_saved_gpu":float(np.max(np.abs(np.asarray(bs)-saved))),
            "max_cpu_batch_vs_single":float(np.max(np.abs(np.asarray(bs)-individual))),
            "checked_prefix_step":k,"prefix_vs_full_abs":abs(ps-individual[k])}
    report["cases"].append(item)
    (OUT/"forward_replay.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"stage":"case_done",**{k:v for k,v in item.items() if "scores" not in k},"elapsed":time.monotonic()-start}),flush=True)
report["complete"] = True
report["cuda_initialized"] = torch.cuda.is_initialized()
report["elapsed_seconds"] = time.monotonic()-start
report["max_cpu_batch_vs_saved_gpu"] = max(c["max_cpu_batch_vs_saved_gpu"] for c in report["cases"])
report["max_cpu_batch_vs_single"] = max(c["max_cpu_batch_vs_single"] for c in report["cases"])
report["max_prefix_vs_full_abs"] = max(c["prefix_vs_full_abs"] for c in report["cases"])
(OUT/"forward_replay.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k != "cases"}),flush=True)
