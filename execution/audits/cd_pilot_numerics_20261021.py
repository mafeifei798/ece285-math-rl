"""Independent post-pilot checks of GRPO normalization and held-out separation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

root=Path(sys.argv[1]).resolve()
pilot=root/"experiments/cd_prm_pilot_20261021_gpu"
out=pilot/"artifacts"
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="0"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
groups=read(out/"group_advantages.json")
assert read(pilot/"pipeline_status.json")["status"]=="completed"
results=[]
for arm in ["B","C","D"]:
    R=np.array([g["arms"][arm]["rewards"] for g in groups])
    saved=np.array([g["arms"][arm]["advantages"] for g in groups])
    c=torch.tensor(R,dtype=torch.float32)
    d=torch.tensor(R,dtype=torch.float32,device="cuda:0")
    # Same reduction order and sample standard deviation as TRL 0.23.1.
    ac=(c-c.mean(dim=1,keepdim=True))/(c.std(dim=1,keepdim=True)+1e-4)
    ag=(d-d.mean(dim=1,keepdim=True))/(d.std(dim=1,keepdim=True)+1e-4)
    ag=ag.cpu().numpy()
    ref=torch.tensor(R,dtype=torch.float64)
    ref=((ref-ref.mean(dim=1,keepdim=True))/(ref.std(dim=1,keepdim=True)+1e-4)).numpy()
    assert np.max(np.abs(ac.numpy()-saved))<1e-5
    changed=np.argwhere((np.abs(saved)>1e-4)&(np.sign(saved)!=np.sign(ag)))
    assert not len(changed),changed
    results.append({"arm":arm,"max_CPU_GPU_difference":float(np.max(np.abs(saved-ag))),
                    "max_CPU_float64_difference":float(np.max(np.abs(saved-ref))),
                    "max_GPU_float64_difference":float(np.max(np.abs(ag-ref))),
                    "nontrivial_sign_flips":len(changed),"GPU_advantages":ag.tolist()})
checks={}
C=read(out/"C/initialization.json");D=read(out/"D/initialization.json")
for k in ["initial_adapter_hash","parameter_names","batch_manifest_sha256","config_sha256"]:
    checks[k]=C[k]==D[k]
assert all(checks.values())
for arm in ["C","D"]:
    m=rows(out/arm/"metrics.jsonl")
    assert len(m)==32 and [x["step"] for x in m]==list(range(1,33))
    assert all(np.isfinite(x["total_loss"]) and x["gradient_norm_before_clip"]>0 for x in m)
    coeff=1 if arm=="C" else 0
    for x in m:
        v=x["component_losses"]
        assert abs(x["total_loss"]-(v["fail"]+coeff*v["success"]+.5*(v["anchor_positive"]+v["anchor_negative"])))<1e-12
    receipt=read(out/arm/"completion.json")
    assert receipt["frozen_parameters_unchanged"]
    assert hashlib.sha256((out/arm/"adapter/adapter_model.safetensors").read_bytes()).hexdigest()==receipt["adapter_file_sha256"]
forbidden=set(read(out/"forbidden_anchor_question_hashes.json"))
assert not {r["question_sha256"] for r in rows(out/"buffer_rollouts.jsonl")}&forbidden
assert not {r["question_sha256"] for r in rows(out/"anchors.jsonl")}&forbidden
report={"passed":True,"matched_initialization_and_batches":checks,"both_arms_32_finite_updates":True,
        "loss_components_verified":True,"heldout_question_separation_verified":True,"adapter_hashes_verified":True,
        "normalization":results,"note":"Only reward normalization is recomputed on GPU; no actor or PRM update. The small numeric tolerance issue that stopped B is recorded separately, not silently altered."}
(out/"independent_validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="normalization"}))
print(json.dumps([{k:v for k,v in r.items() if k!="GPU_advantages"} for r in results]))
