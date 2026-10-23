"""Actual 1.5B LoRA gradient/save/reload check, isolated from final C/D models."""
import gc
from pilot_common import *


def main():
    cpu_setup();guard()
    import torch
    from transformers import AutoModelForTokenClassification,AutoTokenizer
    from peft import LoraConfig,get_peft_model,PeftModel
    from prm_reward import encode_native
    from adapt import pooled_error,flat_hash
    torch.manual_seed(CONFIG["seed"])
    path=ROOT/"models/H4-PRM-1.5B-0.2"/CONFIG["prm_revision"]
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    anchor=min(read_jsonl(OUT/"anchors.jsonl"),key=lambda r:r["input_tokens"])
    e=encode_native(tokenizer,anchor["question"],anchor["completion"],4096)
    base=AutoModelForTokenClassification.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation="sdpa")
    model=get_peft_model(base,LoraConfig(r=CONFIG["lora_r"],lora_alpha=CONFIG["lora_alpha"],lora_dropout=0.0,target_modules=CONFIG["target_modules"],bias="none"))
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.0
    model.eval()
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    frozen=[(n,p,p._version) for n,p in model.named_parameters() if not p.requires_grad]
    assert all("lora_" in n for n,p in params)
    ids=torch.tensor([e["input_ids"]]);mask=torch.ones_like(ids)
    logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[0,e["positions"]]
    E=pooled_error(logits.float().softmax(-1)[:,1])
    fail=-E.log();success=-torch.log1p(-E)
    anchor_loss=torch.nn.functional.cross_entropy(logits[-1:],torch.tensor([anchor["label"]]))
    p=[x for n,x in params]
    C=torch.autograd.grad(fail+success+anchor_loss,p,retain_graph=True)
    D=torch.autograd.grad(fail+0*success+anchor_loss,p,retain_graph=True)
    S=torch.autograd.grad(success,p,retain_graph=True)
    max_diff=max(float((a-b-c).abs().max()) for a,b,c in zip(C,D,S))
    assert all(torch.allclose(a-b,c,atol=3e-6,rtol=3e-4) for a,b,c in zip(C,D,S)),max_diff
    init=flat_hash(params)
    (fail+success+anchor_loss).backward()
    norm=float(torch.nn.utils.clip_grad_norm_(p,1.0));assert norm>0
    optimizer=torch.optim.AdamW(p,lr=CONFIG["learning_rate"],weight_decay=0.0)
    optimizer.step()
    assert flat_hash(params)!=init
    assert all(v._version==version and v.grad is None for n,v,version in frozen)
    model.eval()
    with torch.inference_mode():pred=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[0,e["positions"]].clone()
    dest=OUT/"technical_smoke_discard_not_for_C_D"
    assert not dest.exists()
    model.save_pretrained(dest,safe_serialization=True)
    del model,base,optimizer,params,frozen,p,C,D,S,logits,fail,success,anchor_loss,E
    gc.collect()
    base=AutoModelForTokenClassification.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation="sdpa")
    loaded=PeftModel.from_pretrained(base,dest,is_trainable=False).eval()
    with torch.inference_mode():replay=loaded(input_ids=ids,attention_mask=mask,use_cache=False).logits[0,e["positions"]]
    difference=float((pred-replay).abs().max());assert difference<1e-5,difference
    result={"passed":True,"actual_model_gradient_switch_max_abs_error":max_diff,"gradient_norm":norm,
        "save_reload_max_abs_error":difference,"trainable_parameters":sum(x.numel() for n,x in loaded.named_parameters() if "lora_" in n),
        "only_lora_updated":True,"frozen_classifier":True,"cuda_initialized":torch.cuda.is_initialized(),
        "note":"Technical smoke uses one public anchor for synthetic loss wiring only. Its adapter is never loaded into C/D or used for scoring."}
    assert not result["cuda_initialized"]
    atomic_json(OUT/"model_preflight.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
