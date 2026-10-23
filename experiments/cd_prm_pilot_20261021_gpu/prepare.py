"""Pinned pilot selection, decontaminated public anchors, frozen actor generation."""
import collections
import json
import os
from pathlib import Path
import subprocess
import time

from pilot_common import *

ANCHOR_REPO = "HuggingFaceH4/prm800k-trl-dedup"
ANCHOR_REVISION = "fd8cba17d8214bfbabf6e7e10f29da4e6614dd6c"
ANCHOR_SHA256 = "d9e39ca727f8000c158f3baf51c7fe0c175f86146ad9250c2965f48172006864"


def select():
    OUT.mkdir(exist_ok=True)
    if (OUT / "selection_manifest.json").exists():
        m = json_read(OUT / "selection_manifest.json")
        assert m["config_sha256"] == file_hash(HERE / "config.json")
        for name, sha in m["files"].items():
            assert file_hash(OUT / name) == sha
        return
    train = read_jsonl(ROOT / "data/prepared/train.jsonl")
    byid = {r["problem_id"]: r for r in train}
    bpath = ROOT / "runs/B_seed17_6M/rollouts.jsonl"
    b = read_jsonl(bpath)
    groups = collections.defaultdict(list)
    for gi in range(len(b)//8):
        rows = b[8*gi:8*(gi+1)]
        assert len({r["problem_id"] for r in rows}) == 1
        yes = sum(r["outcome_reward"] for r in rows)
        kind = "all_wrong" if yes == 0 else ("all_correct" if yes == 8 else "mixed")
        groups[kind].append(gi)
    selected = set()
    for kind, indices in sorted(groups.items()):
        selected.update(sorted(indices, key=lambda i: rank("B-group-"+kind, i))[:CONFIG["probe_groups_per_outcome_type"]])
    selected.update(i//8 for i in CONFIG["known_error_rows"])
    probe = []
    for gi in sorted(selected):
        for i in range(gi*8, (gi+1)*8):
            row = b[i]
            probe.append({**row, "row_index": i, "group_index": gi,
                          "question": byid[row["problem_id"]]["problem"],
                          "known_error_case": i in CONFIG["known_error_rows"],
                          "illustrative_group": gi in {i//8 for i in CONFIG["known_error_rows"]}})
    forbidden = {question_key(r["question"]) for r in probe}
    for name in ["validation", "math500", "gsm8k"]:
        forbidden.update(r["question_sha256"] for r in read_jsonl(ROOT/f"data/prepared/{name}.jsonl"))
    pb_all = json_read(ROOT / "data/raw/processbench_math.json")
    forbidden.update(question_key(r["problem"]) for r in pb_all)
    candidates = [r for r in train if r["question_sha256"] not in forbidden]
    selected_train = sorted(candidates, key=lambda r: rank("pilot-buffer", r["problem_id"]))[:CONFIG["buffer_questions"]]
    assert len(selected_train) == CONFIG["buffer_questions"]
    pb_strata = collections.defaultdict(list)
    for r in pb_all:
        key = "answer_correct_valid" if r["label"] == -1 else ("answer_correct_invalid" if r["final_answer_correct"] else "answer_wrong_invalid")
        pb_strata[key].append(r)
    pb = []
    for key in CONFIG["processbench_strata"]:
        for r in sorted(pb_strata[key], key=lambda r: rank("pilot-processbench", r["id"]))[:CONFIG["processbench_rows_per_stratum"]]:
            assert all("\n\n" not in s for s in r["steps"])
            pb.append({**r, "pilot_stratum": key})
    write_jsonl(OUT / "buffer_questions.jsonl", selected_train)
    write_jsonl(OUT / "probe_B6M.jsonl", probe)
    write_jsonl(OUT / "processbench_subset.jsonl", pb)
    atomic_json(OUT / "forbidden_anchor_question_hashes.json", sorted(forbidden))
    m = {"config_sha256": file_hash(HERE / "config.json"), "selection_unix": time.time(),
         "B_rollout_source_sha256": file_hash(bpath), "known_cases_excluded_from_adaptation": True,
         "probe_groups": len(selected), "probe_rows": len(probe), "processbench_rows": len(pb),
         "selection": "Hash selection within fixed outcome strata; known-error groups disclosed separately",
         "files": {n:file_hash(OUT/n) for n in ["buffer_questions.jsonl", "probe_B6M.jsonl", "processbench_subset.jsonl", "forbidden_anchor_question_hashes.json"]}}
    atomic_json(OUT / "selection_manifest.json", m)
    emit("selection_complete", buffer_questions=len(selected_train), probe_groups=len(selected), processbench=len(pb))


def anchors():
    if (OUT / "anchor_manifest.json").exists():
        m=json_read(OUT/"anchor_manifest.json")
        assert file_hash(OUT/"anchors.jsonl") == m["anchors_sha256"]
        return
    source = OUT / "public_anchor_source.parquet"
    if not source.exists():
        url = f"https://hf-mirror.com/datasets/{ANCHOR_REPO}/resolve/{ANCHOR_REVISION}/data/train-00000-of-00001.parquet"
        subprocess.run(["curl","--fail","--location","--retry","3","--connect-timeout","20","--max-time","600","--output",str(source)+".partial",url],check=True)
        assert file_hash(str(source)+".partial") == ANCHOR_SHA256
        Path(str(source)+".partial").replace(source)
    assert file_hash(source) == ANCHOR_SHA256
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from prm_reward import encode_native
    tok = AutoTokenizer.from_pretrained(ROOT/"models/H4-PRM-1.5B-0.2"/CONFIG["prm_revision"],local_files_only=True)
    forbidden=set(json_read(OUT/"forbidden_anchor_question_hashes.json"))
    # Only original MATH-train questions can supply new anchors: this also excludes full MATH test.
    allowed={r["question_sha256"] for r in read_jsonl(ROOT/"data/prepared/train.jsonl")}
    candidates={0:{},1:{}}
    for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
        for r in batch.to_pylist():
            qh=question_key(r["prompt"])
            if qh in forbidden or qh not in allowed: continue
            assert len(r["completions"]) == len(r["labels"])
            # Store only unique complete native prefixes, never truncate a step.
            for j,label in enumerate(r["labels"]):
                text="\n\n".join(r["completions"][:j+1])
                ident=hashlib.sha256((r["prompt"]+"\0"+text).encode()).hexdigest()
                entry={"anchor_id":ident,"question":r["prompt"],"question_sha256":qh,
                       "completion":text,"label":int(label),"source_index":r["index"],"source_step":j}
                candidates[int(label)].setdefault(ident,entry)
        # Selection is among the complete source file, not first accepted rows.
    result=[]
    exclusions=collections.Counter()
    for label in [0,1]:
        ordered=sorted(candidates[label].values(),key=lambda r:rank("pilot-anchor",r["anchor_id"]))
        seen_questions=set()
        for r in ordered:
            if r["question_sha256"] in seen_questions: continue
            try:
                encoded=encode_native(tok,r["question"],r["completion"],CONFIG["anchor_max_tokens"])
            except ValueError:
                exclusions["length_or_boundary"]+=1
                continue
            r["input_tokens"]=len(encoded["input_ids"])
            result.append(r); seen_questions.add(r["question_sha256"])
            if len(seen_questions)==CONFIG["anchor_prefixes"]//2: break
        assert len(seen_questions)==CONFIG["anchor_prefixes"]//2
    write_jsonl(OUT/"anchors.jsonl",result)
    atomic_json(OUT/"anchor_manifest.json",{"repository":ANCHOR_REPO,"revision":ANCHOR_REVISION,
         "source_sha256":ANCHOR_SHA256,"anchors_sha256":file_hash(OUT/"anchors.jsonl"),
         "count":len(result),"labels":dict(collections.Counter(r["label"] for r in result)),
         "forbidden_overlap":0,"scope":"128 class-balanced native prefixes; at most one per question per label; original MATH train only",
         "eligible_unique_prefixes":{str(y):len(candidates[y]) for y in [0,1]},"excluded":dict(exclusions)})
    emit("anchors_complete",count=len(result))


def generate():
    if (OUT/"buffer_manifest.json").exists():
        m=json_read(OUT/"buffer_manifest.json")
        assert m["buffer_sha256"] == file_hash(OUT/"buffer_rollouts.jsonl")
        return
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,StoppingCriteria,StoppingCriteriaList
    from reward import outcome
    questions=read_jsonl(OUT/"buffer_questions.jsonl")
    path=ROOT/"models/Qwen2.5-Math-1.5B"/CONFIG["initial_actor_revision"]
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side="left")
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation="sdpa").to(CONFIG["device"]).eval()
    model.requires_grad_(False)
    assert torch.cuda.is_initialized()
    torch.manual_seed(CONFIG["generation_seed"])
    # One fixed batch of 16 prompts, expanded to 64 continuations; no policy adaptation.
    inputs=tokenizer([r["prompt"] for r in questions],padding=True,add_special_tokens=False,return_tensors="pt").to(CONFIG["device"])
    prompt_width=inputs.input_ids.shape[1]
    started=time.monotonic()
    class Heartbeat(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            n=input_ids.shape[1]-prompt_width
            if n==1 or n%32==0:
                emit("initial_actor_generation",decode_steps=n,max_steps=CONFIG["max_new_tokens"],batch_size=input_ids.shape[0],elapsed_seconds=time.monotonic()-started)
            return False
    emit("initial_actor_loaded",dtype="float32",device=CONFIG["device"],prompt_width=prompt_width)
    with torch.inference_mode():
        output=model.generate(**inputs,do_sample=True,temperature=CONFIG["temperature"],top_p=CONFIG["top_p"],top_k=CONFIG["top_k"],
          num_return_sequences=CONFIG["samples_per_question"],max_new_tokens=CONFIG["max_new_tokens"],pad_token_id=tokenizer.pad_token_id,
          stopping_criteria=StoppingCriteriaList([Heartbeat()]),use_cache=True)
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    rows=[]
    for i,tokens in enumerate(output[:,prompt_width:].tolist()):
        q=questions[i//CONFIG["samples_per_question"]]
        end=next((j for j,t in enumerate(tokens) if t in eos),None)
        terminated=end is not None
        ids=tokens[:end+1] if terminated else tokens
        text=tokenizer.decode(ids,skip_special_tokens=True)
        result=outcome(text,q["answer"],terminated)
        rows.append({"sample_id":f"initial-{i}","problem_id":q["problem_id"],"question":q["problem"],
          "question_sha256":q["question_sha256"],"completion":text,"completion_tokens":len(ids),
          "terminated":terminated,"gold":q["answer"],**result})
    write_jsonl(OUT/"buffer_rollouts.jsonl",rows)
    valid=[r for r in rows if r["status"] in {"exact","symbolic"}]
    counts=collections.Counter(int(r["reward"]) for r in valid)
    m={"buffer_sha256":file_hash(OUT/"buffer_rollouts.jsonl"),"rows":len(rows),"valid":len(valid),
       "valid_outcomes":dict(counts),"status":dict(collections.Counter(r["status"] for r in rows)),
       "completion_tokens":sum(r["completion_tokens"] for r in rows),"elapsed_seconds":time.monotonic()-started,
       "actor_revision":CONFIG["initial_actor_revision"],"dtype":"float32","device":CONFIG["device"],"cuda_initialized":torch.cuda.is_initialized(),
       "generation_config":model.generation_config.to_dict(),"sampling_seed":CONFIG["generation_seed"],
       "no_selection_on_prm_scores":True}
    atomic_json(OUT/"buffer_manifest.json",m)
    assert counts[0] > 0 and counts[1] > 0, "Both outcome classes required; do not silently change buffer"
    emit("buffer_complete",**{k:v for k,v in m.items() if k!="generation_config"})


if __name__=="__main__":
    cpu_setup()
    OUT.mkdir(exist_ok=True)
    atomic_json(OUT/"B_guard_before.json",guard())
    select()
    anchors()
    generate()
    atomic_json(OUT/"B_guard_after_prepare.json",guard())
