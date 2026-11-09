"""Recovery bookkeeping; original experiment and validated loss are immutable."""
from pathlib import Path
import importlib.util
_path=Path(__file__).resolve().parents[1]/'prefix_triplet_seed17_48_20261107/common.py'
_spec=importlib.util.spec_from_file_location('immutable_triplet_common',_path)
_old=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_old)
for _name in dir(_old):
    if not _name.startswith('_'):globals()[_name]=getattr(_old,_name)
ORIGINAL_HERE=HERE
HERE=Path(__file__).resolve().parent

def verify():
    old_release=_old.verify()
    release=read(HERE/'release.json')
    assert old_release==release['original_release_sha256']
    for n,h in release['files'].items():assert sha(HERE/n)==h,('Recovery source changed',n)
    return sha(HERE/'release.json')
