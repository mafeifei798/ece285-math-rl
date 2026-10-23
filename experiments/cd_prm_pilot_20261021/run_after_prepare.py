"""One-shot bounded continuation: finish preparation, adapt C/D, score, then exit."""
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
    assert not receipt.exists(),"Pilot pipeline already launched; inspect its receipt instead of duplicating"
    expected={p.name:file_hash(p) for p in HERE.glob("*.py")}
    expected["config.json"]=file_hash(HERE/"config.json")
    started=time.time()
    state={"pid":os.getpid(),"started_unix":started,"source_hashes":expected,"scope":CONFIG["scope"]}
    def status(name,**kw):
        state.update(status=name,updated_unix=time.time(),**kw);atomic_json(receipt,state)
    try:
        guard()
        assert json_read(OUT/"model_preflight.json")["passed"]
        assert all(json_read(OUT/"preflight.json")[k] for k in ["detached_pool_gradient_verified","single_error_gradient_verified","reward_replay_verified","mechanism_metrics_verified","pilot_data_separation_verified"])
        status("waiting_for_preparation")
        prepare=json_read(HERE/"prepare_launch.json")
        # The already-running preparation is the only actor inference; it performs no training.
        for name,sha in prepare["files"].items():
            if name in expected:assert expected[name]==sha,name
        while not (OUT/"B_guard_after_prepare.json").exists():
            proc=Path(f"/proc/{prepare['pid']}/stat")
            assert proc.exists() and proc.read_text().split()[2]!="Z","Preparation exited without completion; inspect prepare.log"
            assert time.time()-started<12*3600,"Bounded preparation wait exceeded"
            time.sleep(10)
        for script,stage in [("adapt.py","adapting_C_D"),("check.py","scoring_and_replaying")]:
            assert all(file_hash(HERE/name)==sha for name,sha in expected.items()),"Pilot source changed after launch"
            guard();status(stage)
            with (HERE/(script+".log")).open("ab") as log:
                child=subprocess.Popen([sys.executable,"-u","-B",str(HERE/script)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                status(stage,child_pid=child.pid)
                code=child.wait(timeout=12*3600)
                assert code==0,f"{script} failed with code {code}; do not continue"
        assert (OUT/"summary.json").exists()
        guard();status("completed",summary=str(OUT/"summary.json"),C_D_RL_started=False)
    except Exception as exc:
        status("failed",error=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=="__main__":main()
