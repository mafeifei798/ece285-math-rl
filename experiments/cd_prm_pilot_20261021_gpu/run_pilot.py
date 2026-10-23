"""One-shot authorized PRM pilot on GPU0, released by B's independent failure."""
import fcntl
import os
import subprocess
import sys
import time
import traceback
from pilot_common import *


def main():
    lock=(HERE/"pipeline.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=HERE/"pipeline_status.json"
    assert not receipt.exists(),"Refusing duplicate pilot launch"
    expected={p.name:file_hash(p) for p in HERE.glob("*.py")}
    expected["config.json"]=file_hash(HERE/"config.json")
    state={"pid":os.getpid(),"started_unix":time.time(),"source_hashes":expected,"scope":CONFIG["scope"]}
    def status(name,**kw):
        state.update(status=name,updated_unix=time.time(),**kw);atomic_json(receipt,state)
    try:
        guard()
        b=json_read(ROOT/"runs/B_seed17_24M/controller.json")
        assert b["status"]=="failed", "GPU migration allowed only after observed B failure"
        mem=int(subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
        assert mem<100,f"GPU0 occupied ({mem} MiB); do not compete with a resumed B"
        atomic_json(OUT/"B_failure_at_pilot_start.json",json_read(ROOT/"runs/B_seed17_24M/failure.json"))
        status("starting")
        for script,stage in [("prepare.py","initial_actor_sampling"),("preflight_model.py","actual_model_preflight"),
                             ("adapt.py","matched_C_D_adaptation"),("check.py","paired_scoring_and_advantages")]:
            assert all(file_hash(HERE/name)==sha for name,sha in expected.items()),"Pilot source changed"
            guard();status(stage)
            with (HERE/(script+".log")).open("ab") as log:
                child=subprocess.Popen([sys.executable,"-u","-B",str(HERE/script)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                status(stage,child_pid=child.pid)
                try:code=child.wait(timeout=4*3600)
                except subprocess.TimeoutExpired:
                    child.terminate();child.wait(timeout=30);raise
                assert code==0,f"{script} failed with code {code}; inspect retained log"
            if script=="preflight_model.py":assert json_read(OUT/"model_preflight.json")["passed"]
        assert (OUT/"summary.json").exists()
        guard();status("completed",child_pid=None,summary=str(OUT/"summary.json"),C_D_RL_started=False)
    except Exception as exc:
        status("failed",error=repr(exc),traceback=traceback.format_exc());raise


if __name__=="__main__":main()
