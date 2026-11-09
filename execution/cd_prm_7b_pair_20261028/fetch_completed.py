"""Retrieve only completed evidence; never execute or resume remote work."""
from pathlib import Path
import subprocess
import json
import hashlib

BASE=Path(__file__).resolve().parent
FILES=['controller.json','progress.json','cpu_checks.json','model_preflight.json','paired_wiring.json',
 'summary.json','RESULTS.txt','review.html','group_advantages.jsonl','probe.jsonl','illustrations.jsonl',
 'processbench.jsonl','selection.json','anchor_manifest.json','buffer_manifest.json','batch_manifest.json',
 'data_ready.json','independent_input_audit_full.json','first_checkpoint_audit.json','protected_originals.json',
 'wiring_recovery/postprocess.json','wiring_recovery/postprocess_dispatch.json','wiring_recovery/reassessment.json',
 'wiring_recovery/gradient_replay.json','wiring_recovery/fp32_reference_v2/gradient_replay.json',
 'admission_migration_v2/accepted.json']
for a in ['B','C','D']:FILES.extend([f'scoring_{a}.json',f'scores_{a}.jsonl'])
for a in ['C','D']:FILES.extend([f'{a}/initialization.json',f'{a}/completion.json',f'{a}/latest_checkpoint.json',f'{a}/metrics.jsonl'])
REMOTE='names='+repr(FILES)+r'''
from pathlib import Path
import json,hashlib
root=Path('/path/to/ece285_math_rl_project');out=root/'runs/CD_PRM7B_pair_seed17_20261028'
post=json.loads((out/'wiring_recovery/postprocess.json').read_text());assert post['status']=='complete',post
data={n:(out/n).read_text() for n in names}
data['release.json']=(root/'experiments/cd_prm_7b_pair_20261028/release.json').read_text()
print(json.dumps(data))
'''


def main():
    cmd=['ssh','-p','11032','-i','/home/USER/.ssh/fym_125_76_110_84_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
      'USER@GPU_HOST','/path/to/ece285_math_rl_project/.venv/bin/python -B -']
    result=subprocess.run(cmd,input=REMOTE,text=True,capture_output=True,timeout=120)
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    data=json.loads(result.stdout);assert set(data)==set(FILES+['release.json'])
    dest=BASE/'artifacts';dest.mkdir(exist_ok=True);hashes={}
    for name,value in data.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (BASE/'retrieved_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(files=len(hashes),total_bytes=sum((dest/n).stat().st_size for n in hashes),local=str(dest))))


if __name__=='__main__':main()
