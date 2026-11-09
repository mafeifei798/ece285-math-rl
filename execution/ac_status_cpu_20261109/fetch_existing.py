"""Retrieve already-written files only. No model imports or GPU queries."""
from pathlib import Path
import base64
import gzip
import hashlib
import json
import runpy
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = "/path/to/ece285_math_rl_project"

CODE = r'''
from pathlib import Path
import base64,gzip,hashlib,json,time
root=Path('/path/to/ece285_math_rl_project')
run=root/'runs/Budget_extension_seed17_12M_20261108'
files=[]
for repeat in [0,1]:
    base=run/'evaluation/baseline'/f'repeat_{repeat}'
    files += [base/'summary.json',base/'contract.json']
    files += [base/arm/'predictions.jsonl' for arm in ['A','C_bounded']]
for arm,step in [('A',432),('C_bounded',365)]:
    base=run/arm
    files += [base/name for name in ['progress.json','last_checkpoint.json',
        'generation_checks.jsonl','actual_signals.jsonl','update_audits.jsonl',
        'resources.jsonl','format_monitoring.jsonl','first_update_acceptance_432.json']]
    files += [base/f'checkpoint-{step}'/name for name in [
        'pilot_checkpoint.json','completion_accounting.json','trainer_state.json','adapter_config.json']]
files += [run/'controller.json',run/'children.json',run/'A/stage_432_result.json']
checkpoint_checks={}
for arm,step in [('A',432),('C_bounded',365)]:
    cp=run/arm/f'checkpoint-{step}'
    contract=json.loads((cp/'pilot_checkpoint.json').read_text())
    assert contract['global_step']==step and contract['complete_generation_boundary']
    checked={}
    for name,expected in contract['files_sha256'].items():
        h=hashlib.sha256()
        with (cp/name).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==expected,(arm,step,name)
        checked[name]=expected
    accounting=json.loads((cp/'completion_accounting.json').read_text())
    assert accounting['loss_rows']==accounting['generated_completions']==step*32
    assert hashlib.sha256((cp/'completion_accounting.json').read_bytes()).hexdigest()==contract['accounting_sha256']
    checkpoint_checks[arm]={'step':step,'path':str(cp),'files':checked,'accounting_verified':True}
payload={}
for path in files:
    assert path.is_relative_to(run)
    raw=path.read_bytes()
    payload[str(path.relative_to(run))]={'sha256':hashlib.sha256(raw).hexdigest(),
        'bytes':len(raw),'data':base64.b64encode(raw).decode()}
out={'retrieved_unix':time.time(),'files':payload,'checkpoint_checks':checkpoint_checks,
    'scope':'Existing files and file hashes only; no model load, GPU query, generation, optimizer or remote write.'}
print(base64.b64encode(gzip.compress(json.dumps(out).encode())).decode())
'''


def main():
    HERE.mkdir(exist_ok=True)
    ssh = runpy.run_path(str(ROOT / "execution/ac_seed42_v1_20261106/inspect_remote.py"))["SSH"]
    command = f"CUDA_VISIBLE_DEVICES='' {REMOTE}/.venv/bin/python -B -"
    response = subprocess.run(ssh + [command], input=CODE, text=True,
                              capture_output=True, timeout=55)
    if response.returncode:
        raise RuntimeError(response.stderr[-3000:])
    result = json.loads(gzip.decompress(base64.b64decode(response.stdout.strip())))
    manifest = {}
    for name, entry in result.pop("files").items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        raw = base64.b64decode(entry.pop("data"))
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        target = HERE / "existing_extension_evidence" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == raw, "Refuse to replace evidence"
        else:
            target.write_bytes(raw)
        manifest[name] = entry
    result["files"] = manifest
    (HERE / "retrieval_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"files": len(manifest), "bytes": sum(x["bytes"] for x in manifest.values()),
                      "checkpoint_steps_verified": {a: x["step"] for a, x in result["checkpoint_checks"].items()},
                      "GPU_calls": 0, "new_remote_jobs": 0}))


if __name__ == "__main__":
    main()
