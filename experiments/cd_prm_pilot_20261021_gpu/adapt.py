"""Matched CPU LoRA adaptation: C=Lfail+Lsuccess+Lstep, D=Lfail+Lstep."""
import argparse
import gc
import random
import time

from pilot_common import *


def pooled_error(probabilities):
    import torch
    e=1-probabilities
    w=torch.softmax(e/CONFIG["pool_temperature"],dim=0).detach()
    return (w*e).sum().clamp(1e-7,1-1e-7)


def technical_math_check():
    import torch
    z=torch.tensor([-1.2,.7,3.1],requires_grad=True)
    p=z.sigmoid(); e=1-p
    w=torch.softmax(e/CONFIG["pool_temperature"],0).detach()
    E=(w*e).sum()
    observed=torch.autograd.grad(-E.log(),z)[0]
    expected=w*p.detach()*(1-p.detach())/E.detach()
    assert torch.allclose(observed,expected,atol=1e-7,rtol=1e-6)
    single=torch.tensor(3.8,requires_grad=True)
    grad=torch.autograd.grad(-torch.log(1-single.sigmoid()),single)[0]
    assert torch.allclose(grad,single.detach().sigmoid(),atol=1e-6)
    return {"detached_pool_gradient_verified":True,"single_error_gradient_verified":True}


def prepare_batches(tok):
    from prm_reward import encode_native
    raw=read_jsonl(OUT/"buffer_rollouts.jsonl")
    data={"fail":[],"success":[],"anchor_positive":[],"anchor_negative":[]}
    for r in raw:
        if r["status"] not in {"exact","symbolic"}: continue
        encoded=encode_native(tok,r["question"],r["completion"],CONFIG["prm_max_tokens"])
        data["success" if r["reward"] else "fail"].append({**r,**encoded})
    for r in read_jsonl(OUT/"anchors.jsonl"):
        encoded=encode_native(tok,r["question"],r["completion"],CONFIG["anchor_max_tokens"])
        data["anchor_positive" if r["label"] else "anchor_negative"].append({**r,**encoded})
    assert all(data.values())
    rng=random.Random(CONFIG["seed"])
    indices={}
    for key,rows in data.items():
        seq=[]
        while len(seq)<CONFIG["optimizer_steps"]:
            ix=list(range(len(rows)));rng.shuffle(ix);seq.extend(ix)
        indices[key]=seq[:CONFIG["optimizer_steps"]]
    schedule=[{k:indices[k][i] for k in data} for i in range(CONFIG["optimizer_steps"])]
    manifest={"counts":{k:len(v) for k,v in data.items()},"schedule":schedule,
      "buffer_sha256":file_hash(OUT/"buffer_rollouts.jsonl"),"anchors_sha256":file_hash(OUT/"anchors.jsonl"),
      "config_sha256":file_hash(HERE/"config.json"),"component_reduction":"Separate component means, summed; anchors mean of positive and negative; no reaveraging when c=0"}
    path=OUT/"batch_manifest.json"
    if path.exists(): assert json_read(path)==manifest
    else: atomic_json(path,manifest)
    return data,schedule


def flat_hash(params):
    h=hashlib.sha256()
    for n,p in params:
        h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(arm):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from peft import LoraConfig,get_peft_model
    from prm_reward import encode_native
    dest=OUT/arm
    assert not dest.exists(),f"Refusing to overwrite an existing arm: {dest}"
    dest.mkdir()
    random.seed(CONFIG["seed"]);np.random.seed(CONFIG["seed"]);torch.manual_seed(CONFIG["seed"])
    modelpath=ROOT/"models/H4-PRM-1.5B-0.2"/CONFIG["prm_revision"]
    tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
    data,schedule=prepare_batches(tok)
    base=AutoModelForTokenClassification.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float32,attn_implementation="sdpa").to(CONFIG["device"])
    base.config.use_cache=False
    # Generic PEFT wrapper: keep the existing classification head frozen.
    model=get_peft_model(base,LoraConfig(r=CONFIG["lora_r"],lora_alpha=CONFIG["lora_alpha"],
      lora_dropout=0.0,target_modules=CONFIG["target_modules"],bias="none"))
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.0
    model.train()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    frozen=[(n,p,p._version) for n,p in model.named_parameters() if not p.requires_grad]
    assert trainable and all("lora_" in n for n,p in trainable)
    params=[p for n,p in trainable]
    initial_hash=flat_hash(trainable)
    initial={"arm":arm,"success_coefficient":CONFIG["arms"][arm],"initial_adapter_hash":initial_hash,
       "parameter_names":[n for n,p in trainable],"trainable_parameters":sum(p.numel() for p in params),
       "batch_manifest_sha256":file_hash(OUT/"batch_manifest.json"),"config_sha256":file_hash(HERE/"config.json"),
       "prm_revision":CONFIG["prm_revision"],"classification_head_frozen":True}
    atomic_json(dest/"initialization.json",initial)
    opt=torch.optim.AdamW(params,lr=CONFIG["learning_rate"],betas=tuple(CONFIG["adam_betas"]),
                         eps=CONFIG["adam_eps"],weight_decay=CONFIG["weight_decay"])
    started=time.monotonic()
    c=CONFIG["arms"][arm]
    for step,batch in enumerate(schedule):
        opt.zero_grad(set_to_none=True)
        values={}; token_count=0
        for key in ["fail","success","anchor_positive","anchor_negative"]:
            r=data[key][batch[key]]
            ids=torch.tensor([r["input_ids"]],dtype=torch.long,device=CONFIG["device"])
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,r["positions"]].float()
            assert torch.isfinite(logits).all()
            if key in {"fail","success"}:
                E=pooled_error(logits.softmax(-1)[:,1])
                raw_loss=-torch.log(E) if key=="fail" else -torch.log1p(-E)
                coefficient=1.0 if key=="fail" else c
            else:
                raw_loss=torch.nn.functional.cross_entropy(logits[-1:].float(),torch.tensor([r["label"]],device=CONFIG["device"]))
                coefficient=0.5*CONFIG["anchor_beta"]
            assert torch.isfinite(raw_loss)
            values[key]=float(raw_loss.detach())
            if step==0 and arm=="C" and key=="success":
                grads=torch.autograd.grad(raw_loss,params,retain_graph=True)
                torch.save(torch.cat([g.detach().reshape(-1) for g in grads]),dest/"first_success_gradient.pt")
                del grads
            # D executes the same successful-sample forward/backward with coefficient zero.
            (coefficient*raw_loss).backward()
            token_count+=ids.numel()
        assert all(p.grad is None and p._version==version for n,p,version in frozen)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        if step==0:
            torch.save(torch.cat([p.grad.detach().reshape(-1) for p in params]),dest/"first_gradient.pt")
        grad_norm=float(torch.nn.utils.clip_grad_norm_(params,CONFIG["max_grad_norm"]))
        assert grad_norm>0 and np.isfinite(grad_norm)
        opt.step()
        metrics={"arm":arm,"step":step+1,"component_losses":values,"total_loss":values["fail"]+c*values["success"]+.5*(values["anchor_positive"]+values["anchor_negative"]),
                 "gradient_norm_before_clip":grad_norm,"input_tokens":token_count,"elapsed_seconds":time.monotonic()-started}
        with (dest/"metrics.jsonl").open("a") as f:f.write(json.dumps(metrics)+"\n")
        emit("adapting",**metrics)
    assert all(p.grad is None and p._version==version for n,p,version in frozen)
    final_hash=flat_hash(trainable)
    assert final_hash!=initial_hash
    model.save_pretrained(dest/"adapter",safe_serialization=True)
    manifest={**initial,"complete":True,"steps":len(schedule),"final_adapter_hash":final_hash,
      "adapter_file_sha256":file_hash(dest/"adapter/adapter_model.safetensors"),
      "elapsed_seconds":time.monotonic()-started,"frozen_parameters_unchanged":True,
      "cuda_initialized":torch.cuda.is_initialized(),"device":CONFIG["device"],"dtype":"float32"}
    assert manifest["cuda_initialized"]
    atomic_json(dest/"completion.json",manifest)
    emit("adaptation_complete",arm=arm,steps=len(schedule),seconds=manifest["elapsed_seconds"])
    del model,base,opt,params,trainable,frozen,data
    gc.collect()


def compare_wiring():
    import torch
    a=json_read(OUT/"C/initialization.json");b=json_read(OUT/"D/initialization.json")
    for k in ["initial_adapter_hash","parameter_names","trainable_parameters","batch_manifest_sha256","config_sha256","prm_revision"]: assert a[k]==b[k],k
    C=torch.load(OUT/"C/first_gradient.pt",map_location="cpu",weights_only=True)
    D=torch.load(OUT/"D/first_gradient.pt",map_location="cpu",weights_only=True)
    S=torch.load(OUT/"C/first_success_gradient.pt",map_location="cpu",weights_only=True)
    error=float((C-D-S).abs().max())
    assert torch.allclose(C-D,S,atol=3e-6,rtol=3e-4),error
    result={"passed":True,"identical_initialization":True,"identical_batch_manifest":True,
            "first_unclipped_gradient_C_minus_D_equals_success_gradient":True,"max_absolute_error":error,
            "note":"Gradient equality checked at common initialization, before parameter divergence or clipping."}
    atomic_json(OUT/"paired_wiring_check.json",result)
    emit("paired_wiring_passed",**result)


if __name__=="__main__":
    cpu_setup();guard()
    atomic_json(OUT/"loss_math_check.json",technical_math_check())
    for arm in ["C","D"]:
        if (OUT/arm/"completion.json").exists():
            receipt=json_read(OUT/arm/"completion.json")
            assert receipt["config_sha256"]==file_hash(HERE/"config.json")
            assert file_hash(OUT/arm/"adapter/adapter_model.safetensors")==receipt["adapter_file_sha256"]
            continue
        train(arm)
    compare_wiring()
    atomic_json(OUT/"B_guard_after_adaptation.json",guard())
