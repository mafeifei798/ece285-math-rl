"""Read-only remote CPU verification, gated on the existing completed controller."""
from pathlib import Path
import json,subprocess,runpy
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SSH=runpy.run_path(str(ROOT/'execution/ac_seed42_v1_20261106/inspect_remote.py'))['SSH']


def main():
    out=HERE/'independent_tensor_acceptance.json';assert not out.exists(),'Preserve completed verification evidence'
    old=json.loads((HERE/'initial_remote_snapshot.json').read_text())['checkpoint']
    expected={k:v['sha256'] for k,v in old.items()};assert len(expected)==8 and all(expected.values())
    code='EXPECTED_SOURCE='+repr(expected)+'\n'+(HERE/'independent_tensor_verify.py').read_text()
    r=subprocess.run(SSH+['/path/to/ece285_math_rl_project/.venv/bin/python -B -'],input=code,text=True,capture_output=True,timeout=180)
    if r.returncode:raise RuntimeError(r.stderr[-5000:])
    value=json.loads(r.stdout);assert value['passed'] and not value['CUDA_initialized']
    out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
