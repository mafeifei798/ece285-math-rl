"""Locked authorized pipeline: real pilots -> fixed A/B training -> evaluation."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from pair_utils import HERE, ROOT, atomic, read, sha, read_jsonl, verify_sources, preservation_snapshot, assert_preserved
sys.path.insert(0, str(ROOT / "operations"))
from run_continuation import environment, verify_evaluation


def gpu_free(index):
    devices = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    uuid = next(r.split(",")[1].strip() for r in devices.splitlines() if int(r.split(",")[0])==index)
    active = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    return uuid not in {r.split(",")[0].strip() for r in active.splitlines()}


def pilot_gate(plan, parent):
    checks = {}
    specs, first = {}, {}
    tokens = {}
    for arm in ["A","B"]:
        path = ROOT / plan["pilots"][arm]
        status, spec = read(path/"status.json"), read(path/"run_config.json")
        data, metrics = read_jsonl(path/"rollouts.jsonl"), read_jsonl(path/"metrics.jsonl")
        audit = read(path/"live_audits.json")
        checks[arm+"_two_updates"] = status["status"]=="pilot_completed" and status["global_step"]==2
        checks[arm+"_parameters_updated"] = status["actor_adapter_changed"] and status["lora_B_abs_sum"]>0
        checks[arm+"_finite_gradient"] = any(math.isfinite(m.get("grad_norm",0)) and m.get("grad_norm",0)>0 for m in metrics)
        checks[arm+"_finite_metrics"] = all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v,(int,float)))
        checks[arm+"_actual_loss_audit"] = audit["passed"] and audit["loss_rows"]==64 and audit["generation_batches"]==2
        checks[arm+"_accounting"] = len(data)==64 and sum(r["completion_tokens"] for r in data)==status["accounting"]["generated_completion_tokens"]
        checks[arm+"_original_reward"] = all(abs(r["reward"]-(r["outcome_reward"] if arm=="A" else
            r["outcome_reward"]+.2*min(r["step_scores"]) if r["valid_for_process_reward"] else 0))<1e-12 for r in data)
        checks[arm+"_no_verifier_exceptions"] = not any(r["status"].startswith("verifier_error") for r in data)
        specs[arm],first[arm] = spec,read(path/"initial_sampling.json")
        tokens[arm] = status["accounting"]["generated_completion_tokens"]
    checks["same_initial_actor_adapter"] = specs["A"]["actor_adapter_initial_sha256"]==specs["B"]["actor_adapter_initial_sha256"]
    checks["same_initial_samples"] = first["A"]==first["B"]
    checks["same_training_arguments"] = all(specs["A"][k]==specs["B"][k] for k in
        ["shared_training_arguments","packages","model_revision","config_sha256","data_manifest_sha256","unit_metadata_sha256"])
    checks["B_frozen_PRM"] = read(ROOT/plan["pilots"]["B"]/"status.json")["prm_frozen_verified"]
    checks["B_native_weights_verified"] = read(ROOT/plan["pilots"]["B"]/"prm_loading.json")["required_scoring_tensors_exactly_matched"]==342
    checks["B_process_reward_exercised"] = any(r["reward"]>r["outcome_reward"] for r in read_jsonl(ROOT/plan["pilots"]["B"]/"rollouts.jsonl"))
    checks["real_LoRA_inference"] = read(parent/"lora_preflight.json")["passed"]
    receipt=dict(passed=all(checks.values()),checks=checks,pilot_extra_completion_tokens=tokens,
        release_sha256=sha(HERE/"release.json"),finished_unix=time.time(),
        scope="Engineering gate only, no test accuracy or reward/loss selection")
    atomic(parent/"pilot_gate.json",receipt)
    assert receipt["passed"], json.dumps(checks)
    return receipt


def compare(plan,parent):
    result={"arms":{},"paired_B_minus_A":{},"seed":17,"verifier":"shared_v3",
            "limitation":"One training seed; paired bootstrap covers questions, not seed variability."}
    for arm in ["A","B"]:
        run=ROOT/plan["runs"][arm]
        result["arms"][arm]={"status":read(run/"status.json"),"evaluation":read(run/"evaluation/evaluation_summary.json")}
    for name,n in [("math500",500),("gsm8k",1319)]:
        a=read_jsonl(ROOT/plan["runs"]["A"]/"evaluation"/(name+".jsonl"))
        b=read_jsonl(ROOT/plan["runs"]["B"]/"evaluation"/(name+".jsonl"))
        ad={r["problem_id"]:r for r in a};bd={r["problem_id"]:r for r in b}
        assert len(ad)==len(bd)==n and ad.keys()==bd.keys()
        diff=np.array([bd[i]["reward"]-ad[i]["reward"] for i in ad])
        rng=np.random.default_rng(285);draws=[]
        for _ in range(20):
            draws.extend(diff[rng.integers(0,n,size=(500,n))].mean(1).tolist())
        result["paired_B_minus_A"][name]={"n":n,"difference_pp":float(diff.mean()*100),
            "paired_question_bootstrap_95ci_pp":[float(v*100) for v in np.quantile(draws,[.025,.975])],
            "B_only_correct":int((diff>0).sum()),"A_only_correct":int((diff<0).sum())}
    atomic(parent/"comparison.json",result)


def main():
    plan=read(HERE/"plan.json");verify_sources()
    parent=ROOT/plan["controller"]
    parent.mkdir(parents=True,exist_ok=True)
    locks=[]
    for path in [ROOT/"receipts/pipeline.lock",ROOT/"receipts/ab_corrected_gpu1.lock",parent/"controller.lock"]:
        f=path.open("a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert not (parent/"controller.json").exists(),"Controller exists; inspect before recovery"
    assert not any((ROOT/p).exists() for p in [*plan["runs"].values(),*plan["pilots"].values()]),"Prior attempt exists"
    preserved=preservation_snapshot()
    atomic(parent/"originals_before.json",preserved)
    atomic(parent/"execution_plan.json",plan)
    state=dict(status="preflight",pid=os.getpid(),started_unix=time.time(),scope=plan["scope"],
               gpu=plan["gpu"],release_sha256=sha(HERE/"release.json"),completed_stages=[])

    def record(**values):
        state.update(values,heartbeat_unix=time.time());atomic(parent/"controller.json",state)

    def execute(label,command):
        verify_sources();assert_preserved(preserved)
        deadline=time.time()+24*3600
        while not gpu_free(plan["gpu"]):
            record(status="waiting_for_gpu",next_stage=label,child_pid=None)
            assert time.time()<deadline,"GPU unavailable; preserve state"
            time.sleep(15)
        started=time.time();log=ROOT/"logs"/(plan["experiment"]+"_"+label+".log")
        with log.open("x") as stream:
            child=subprocess.Popen([str(ROOT/".venv/bin/python"),*command],cwd=ROOT,
                env=environment(plan["gpu"]),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            record(status=label,child_pid=child.pid,log=str(log),stage_started_unix=started)
            while child.poll() is None:
                time.sleep(15);record()
            code=child.returncode
        with (parent/"stages.jsonl").open("a") as f:
            f.write(json.dumps(dict(stage=label,started_unix=started,finished_unix=time.time(),exit_code=code))+"\n")
        record(child_pid=None,last_child_exit_code=code)
        assert code==0,label+" failed; evidence retained, no automatic restart"
        state["completed_stages"].append(label);record()

    record()
    try:
        # Verify frozen PRM assets before any model loading or optimizer step.
        prm=read(ROOT/"audits/qwen7b_offline_20261021/model_assets.json")
        assert prm["complete"] and prm["revision"]==plan["prm_revision"]
        for f in prm["files"]:
            assert sha(ROOT/plan["prm_path"]/f["name"])==f["sha256"]
        for arm in ["A","B"]:
            execute("pilot_"+arm,[str(HERE/"train_arm.py"),"--arm",arm,"--pilot"])
        execute("lora_preflight",[str(ROOT/"operations/lora_preflight.py"),"--adapter",
            str(ROOT/plan["pilots"]["B"]/"final_adapter"),"--receipt",str(parent/"lora_preflight.json")])
        pilot_gate(plan,parent);record(pilot_gate_passed=True)
        for arm in ["A","B"]:
            execute("train_"+arm,[str(HERE/"train_arm.py"),"--arm",arm])
            assert read(ROOT/plan["runs"][arm]/"initial_sampling.json")==read(ROOT/plan["pilots"][arm]/"initial_sampling.json")
        for arm in ["A","B"]:
            run=ROOT/plan["runs"][arm];status=read(run/"status.json")
            assert status["status"]=="completed" and status["accounting"]["generated_completion_tokens"]>=6000000
            checkpoint=Path(status["checkpoint"])
            for name,digest in status["checkpoint_sha256"].items():assert sha(checkpoint/name)==digest
            execute("eval_"+arm,[str(HERE/"evaluate.py"),"--output",str(run/"evaluation"),"--adapter",str(checkpoint)])
            verify_evaluation(run/"evaluation",checkpoint,read(run/"run_config.json"))
            contract=read(run/"evaluation/evaluation_config.json")
            assert contract["unit_metadata_sha256"]==read(HERE/"release.json")["unit_metadata_sha256"]
        compare(plan,parent)
        record(status="completed",finished_unix=time.time(),all_evaluations_completed=True,
               preservation=assert_preserved(preserved),no_new_reward_or_loss=True)
    except Exception as exc:
        record(status="failed",error=str(exc),traceback=traceback.format_exc(),finished_unix=time.time())
        raise


if __name__=="__main__":main()
