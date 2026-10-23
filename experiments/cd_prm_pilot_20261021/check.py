"""Score identical held-out records with B/C/D; replay fixed GRPO advantages."""
import collections
import gc
import time
import numpy as np
from pilot_common import *


def records():
    result=[]
    for r in read_jsonl(OUT/"probe_B6M.jsonl"):
        result.append({**r,"id":f"B-row-{r['row_index']}","kind":"rollout", "eligible":r["valid_for_process_reward"]})
    for r in read_jsonl(OUT/"processbench_subset.jsonl"):
        result.append({**r,"kind":"processbench","question":r["problem"],"completion":"\n\n".join(r["steps"]),"eligible":True})
    return result


def score(arm):
    import torch
    from transformers import AutoTokenizer,AutoModelForTokenClassification
    from peft import PeftModel
    from prm_reward import encode_native
    output=OUT/f"scores_{arm}.jsonl"
    receipt=OUT/f"scoring_{arm}.json"
    if receipt.exists():
        info=json_read(receipt)
        assert info["scores_sha256"]==file_hash(output)
        return
    assert not output.exists(),"Incomplete scores retained; inspect before resuming"
    data=records()
    path=ROOT/"models/H4-PRM-1.5B-0.2"/CONFIG["prm_revision"]
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side="right")
    model=AutoModelForTokenClassification.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation="sdpa")
    if arm!="B":model=PeftModel.from_pretrained(model,OUT/arm/"adapter",is_trainable=False)
    model.eval().requires_grad_(False)
    result={}; encoded=[]
    for r in data:
        if not r["eligible"]:
            result[r["id"]]={"id":r["id"],"kind":r["kind"],"scores":[],"min_score":0.0,"score_status":"invalid_rollout_zero"}
            continue
        try:e=encode_native(tok,r["question"],r["completion"],CONFIG["prm_max_tokens"])
        except ValueError as exc:
            assert r["kind"]=="processbench",str(exc)
            result[r["id"]]={"id":r["id"],"kind":r["kind"],"scores":[],"min_score":None,"score_status":"excluded_native_length_or_boundary","reason":str(exc)}
            continue
        if r["kind"]=="processbench":assert len(e["positions"])==len(r["steps"])
        encoded.append({**e,"id":r["id"],"kind":r["kind"]})
    encoded.sort(key=lambda e:(len(e["input_ids"]),e["id"]))
    started=time.monotonic(); tokens=0
    @torch.inference_mode()
    def forward(batch):
        inputs=tok.pad([{"input_ids":r["input_ids"]} for r in batch],padding=True,return_tensors="pt")
        logits=model(**inputs,use_cache=False).logits.float()
        assert torch.isfinite(logits).all()
        prob=logits.softmax(-1)[...,1]
        return [prob[i,r["positions"]].tolist() for i,r in enumerate(batch)]
    for offset in range(0,len(encoded),4):
        batch=encoded[offset:offset+4]
        scores=forward(batch)
        for r,p in zip(batch,scores):
            result[r["id"]]={"id":r["id"],"kind":r["kind"],"scores":p,"min_score":min(p),"steps":len(p),"input_tokens":len(r["input_ids"]),"score_status":"scored"}
            tokens+=len(r["input_ids"])
        if offset%16==0 or offset+4>=len(encoded):
            emit("scoring",arm=arm,completed=min(offset+4,len(encoded)),total=len(encoded),input_tokens=tokens,elapsed_seconds=time.monotonic()-started)
    repeat=next(e for e in encoded if e["id"]=="B-row-2095")
    rep=forward([repeat])[0]
    error=float(np.max(np.abs(np.array(rep)-result[repeat["id"]]["scores"])))
    assert error<1e-4,error
    write_jsonl(output,[result[r["id"]] for r in data])
    info={"complete":True,"arm":arm,"device":"cpu","dtype":"float32","cuda_initialized":torch.cuda.is_initialized(),
          "scores_sha256":file_hash(output),"count":len(data),"scored":len(encoded),"input_tokens":tokens,
          "repeated_single_vs_batch_max_difference":error,"elapsed_seconds":time.monotonic()-started,
          "adapter_sha256":None if arm=="B" else file_hash(OUT/arm/"adapter/adapter_model.safetensors")}
    assert not info["cuda_initialized"]
    atomic_json(receipt,info)
    del model
    gc.collect()


def advantage(y,m,valid):
    import torch
    from prm_reward import group_diagnostics
    r=[float(a)+.2*float(b) if v else 0.0 for a,b,v in zip(y,m,valid)]
    a,diagnostics=group_diagnostics(y,r,valid,m,8)
    return np.asarray(r),a.numpy(),diagnostics


def mechanism(rows,values,weights=None):
    if weights is None:weights=np.ones(len(rows))
    correct=np.array([r["final_answer_correct"] for r in rows],bool)
    valid=np.array([r["label"]==-1 for r in rows],bool)
    minimum=np.array([v["min_score"] for v in values])
    prediction=np.array([next((i for i,p in enumerate(v["scores"]) if p<.5),-1) for v in values])
    labels=np.array([r["label"] for r in rows])
    pos=valid&correct;neg=(~valid)&correct
    compare=(minimum[pos,None]>minimum[None,neg]).astype(float)+.5*(minimum[pos,None]==minimum[None,neg])
    wp=weights[pos];wn=weights[neg]
    auc=float((compare*wp[:,None]*wn[None,:]).sum()/(wp.sum()*wn.sum())) if wp.sum()*wn.sum()>0 else float("nan")
    acceptance=float((weights[valid]*(prediction[valid]==-1)).sum()/weights[valid].sum()) if weights[valid].sum()>0 else float("nan")
    localization=float((weights[~valid]*(prediction[~valid]==labels[~valid])).sum()/weights[~valid].sum()) if weights[~valid].sum()>0 else float("nan")
    return {"answer_correct_validity_auc":auc,"valid_chain_acceptance":acceptance,"first_error_localization":localization,
            "harmonic_mean":2*acceptance*localization/(acceptance+localization) if acceptance+localization>0 else 0.0,
            "mean_score_valid":float(np.average(minimum[valid],weights=weights[valid])) if weights[valid].sum()>0 else float("nan"),
            "mean_score_invalid":float(np.average(minimum[~valid],weights=weights[~valid])) if weights[~valid].sum()>0 else float("nan")}


def summarize():
    rows=read_jsonl(OUT/"probe_B6M.jsonl")
    pb=read_jsonl(OUT/"processbench_subset.jsonl")
    scores={arm:{r["id"]:r for r in read_jsonl(OUT/f"scores_{arm}.jsonl")} for arm in ["B","C","D"]}
    y=np.array([r["outcome_reward"] for r in rows]);v=np.array([r["valid_for_process_reward"] for r in rows])
    all_results={};case_results=[];group_results=[]
    for arm in scores:
        m=np.array([scores[arm][f"B-row-{r['row_index']}"]["min_score"] for r in rows])
        reward,adv,diagnostics=advantage(y,m,v)
        all_results[arm]={"m":m,"reward":reward,"advantage":adv,"diagnostics":diagnostics}
    for i,r in enumerate(rows):
        if r["known_error_case"]:
            case_results.append({"row_index":r["row_index"],"problem_id":r["problem_id"],"question":r["question"],
                "completion":r["completion"],"gold":r["gold"],"answer":r["answer"],
                "saved_B_gpu_min":r["min_score"],"same_cpu_backend":{a:{"min_score":float(z["m"][i]),"reward":float(z["reward"][i]),"advantage":float(z["advantage"][i]),
                "group_rank_among_valid":int(sum(z["m"][j]>z["m"][i] for j in range(i//8*8,i//8*8+8) if v[j])+1)} for a,z in all_results.items()}})
    for gi in range(len(rows)//8):
        sl=slice(gi*8,gi*8+8)
        info={"source_group_index":rows[gi*8]["group_index"],"problem_id":rows[gi*8]["problem_id"],
              "type":all_results["B"]["diagnostics"][gi]["outcome_type"],"illustrative":rows[gi*8]["illustrative_group"],
              "outcomes":y[sl].tolist(),"valid":v[sl].tolist(),"row_indices":[r["row_index"] for r in rows[sl]],"arms":{}}
        for a,z in all_results.items():
            info["arms"][a]={"min_scores":z["m"][sl].tolist(),"rewards":z["reward"][sl].tolist(),"advantages":z["advantage"][sl].tolist()}
        info["C_D_sign_agreement"]=float(np.mean(np.sign(np.where(np.abs(all_results["C"]["advantage"][sl])<=1e-6,0,all_results["C"]["advantage"][sl]))==np.sign(np.where(np.abs(all_results["D"]["advantage"][sl])<=1e-6,0,all_results["D"]["advantage"][sl]))))
        info["C_D_rms"]=float(np.sqrt(np.mean((all_results["C"]["advantage"][sl]-all_results["D"]["advantage"][sl])**2)))
        group_results.append(info)
    proxy={}
    for a,z in all_results.items():
        pair=[]
        for gi,g in enumerate(group_results):
            if g["illustrative"]: continue
            ix=np.arange(gi*8,gi*8+8);pos=ix[(y[ix]==1)&v[ix]];neg=ix[(y[ix]==0)&v[ix]]
            pair.extend(float(z["m"][i]>z["m"][j])+.5*float(z["m"][i]==z["m"][j]) for i in pos for j in neg)
        proxy[a]={"answer_correct_vs_wrong_pair_accuracy_proxy":float(np.mean(pair)) if pair else None,"pairs":len(pair),
                  "note":"Outcome correctness is not process correctness; descriptive proxy, excluding illustrative groups."}
    pb_eligible=[r for r in pb if all(scores[a][r["id"]]["score_status"]=="scored" for a in scores)]
    values={a:[scores[a][r["id"]] for r in pb_eligible] for a in scores}
    point={a:mechanism(pb_eligible,values[a]) for a in scores}
    # Question-cluster paired resampling; same sampled weights in all three arms.
    keys=sorted({question_key(r["problem"]) for r in pb_eligible})
    key_index={k:i for i,k in enumerate(keys)}
    positions=np.array([key_index[question_key(r["problem"])] for r in pb_eligible])
    rng=np.random.default_rng(285); boots={a:[] for a in scores}
    for _ in range(CONFIG["bootstrap_replicates"]):
        counts=np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys))
        weights=counts[positions]
        for a in scores:boots[a].append(mechanism(pb_eligible,values[a],weights))
    intervals={a:{k:np.nanquantile([r[k] for r in boots[a]],[.025,.975]).tolist() for k in point[a]} for a in scores}
    differences={}
    for left,right in [("C","B"),("D","B"),("D","C")]:
        differences[left+"-"+right]={k:{"estimate":point[left][k]-point[right][k],"paired_question_bootstrap_95CI":np.nanquantile([u[k]-w[k] for u,w in zip(boots[left],boots[right])],[.025,.975]).tolist()} for k in point[left]}
    strata={}
    for kind in ["mixed","all_correct","all_wrong"]:
        selected=[(i,g) for i,g in enumerate(group_results) if g["type"]==kind and not g["illustrative"]]
        ids=np.array([j for i,g in selected for j in range(i*8,i*8+8)],dtype=int)
        strata[kind]={"groups":len(selected),"arms":{}}
        for a,z in all_results.items():
            strata[kind]["arms"][a]={"mean_abs_advantage":float(np.abs(z["advantage"][ids]).mean()),
              "mean_process_score_including_invalid_zero":float(z["m"][ids].mean()),
              "positive_advantage_wrong_count":int(np.sum((y[ids]==0)&(z["advantage"][ids]>1e-6))),
              "negative_advantage_correct_count":int(np.sum((y[ids]==1)&(z["advantage"][ids]<-1e-6)))}
        strata[kind]["C_D_mean_sign_agreement"]=float(np.mean([g["C_D_sign_agreement"] for _,g in selected]))
        strata[kind]["C_D_max_group_RMS"]=max(g["C_D_rms"] for _,g in selected)
    summary={"scope":"Bounded post-B pilot; no policy training, no claim of improved actor accuracy, no formal resource-gate decision",
      "config_sha256":file_hash(HERE/"config.json"),"batch_manifest":json_read(OUT/"batch_manifest.json"),
      "wiring":json_read(OUT/"paired_wiring_check.json"),"known_error_cases":case_results,
      "fixed_group_strata":strata,"outcome_proxy":proxy,
      "processbench_pilot":{"requested":len(pb),"scored":len(pb_eligible),"excluded":[r["id"] for r in pb if r not in pb_eligible],
        "strata_counts":dict(collections.Counter(r["pilot_stratum"] for r in pb_eligible)),"point_estimates":point,"intervals_95":intervals,"paired_differences":differences,
        "interpretation":"Stratified 96-example pilot, not official full ProcessBench score. Intervals cover sampled questions, not training seeds. No tuning based on these results."},
      "limitations":CONFIG["pilot_deviations"]+["No anchor-only S trained in this bounded C/D request; C/D versus B cannot separate anchor adaptation from outcome supervision.","Known error cases are selected examples, not error-rate estimates.","Positive advantage on a wrong final answer is not automatically a faulty process signal; judge the checked mathematical examples separately."],
      "B_guard":guard(),"finished_unix":time.time()}
    atomic_json(OUT/"group_advantages.json",group_results)
    atomic_json(OUT/"summary.json",summary)
    emit("checks_complete",summary_path=str(OUT/"summary.json"),processbench_scored=len(pb_eligible),known_error_cases=len(case_results),actor_RL_started=False)


if __name__=="__main__":
    cpu_setup();guard()
    assert json_read(OUT/"paired_wiring_check.json")["passed"]
    for arm in ["B","C","D"]:score(arm)
    summarize()
