"""Shared experiment stop guard, NOT an empty-answer reward or training filter.

The unmodified PURE reward recipe can reward an empty response. Until a reward
protocol is explicitly chosen, preserve the entire batch and stop before an
update in BOTH arms. Do not drop rows, resample, assign scores, or mask loss.
Nonempty batches execute the previously audited implementation unchanged.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pure_integration_20261022'))
from common import atomic
from trainer import PureIntegration


class EmptyResponseBatchError(RuntimeError):
    def __init__(self, incidents):
        self.incidents = incidents
        super().__init__(f'{len(incidents)} empty response(s): entire batch retained; update stopped')


def inspect_empty(rows):
    incidents = []
    for i, row in enumerate(rows):
        raw = row['completion']
        if not isinstance(raw, str):
            raise TypeError('Decoded completion must be a string; never coerce missing values to empty')
        if not raw.strip():
            incidents.append(dict(row=i, problem_id=row.get('problem_id'),
                raw=raw, completion_ids=row.get('completion_ids'),
                terminated=row.get('terminated'), nonempty_reasoning_steps=0,
                status='empty_decoded_response', mathematical_validity='not_assessed'))
    return incidents


def require_nonempty_batch(rows, *, incident_dir=None, stage='before_signal_preparation'):
    incidents = inspect_empty(rows)
    if incidents:
        if incident_dir is not None:
            directory = Path(incident_dir)
            directory.mkdir(parents=True, exist_ok=True)
            receipt = dict(guard='shared_empty_response_stop_v1', stage=stage,
                semantics='experiment stop; not a reward rule or filtered training batch',
                affected_rows=incidents, batch_size=len(rows), optimizer_steps_permitted=0,
                rows_dropped=0, resampled=False, reward_substitution=False)
            atomic(directory / 'empty_boundary_incident.json', receipt)
            # Full text and actual IDs survive; never silently retry this batch.
            atomic(directory / 'empty_boundary_batch.json', rows)
        raise EmptyResponseBatchError(incidents)


class GuardedPureIntegration(PureIntegration):
    """Independent no-update integration; original implementation is unchanged."""

    def generate(self, questions):
        rows, costs = super().generate(questions)
        require_nonempty_batch(rows, incident_dir=self.run, stage='after_generation_before_PRM')
        return rows, costs

    def prepare(self, arm, rows, scored, old, ref):
        require_nonempty_batch(rows, incident_dir=getattr(self, 'run', None))
        return super().prepare(arm, rows, scored, old, ref)

    def backward_no_update(self, signals, rows):
        require_nonempty_batch(rows, incident_dir=self.run, stage='before_backward')
        return super().backward_no_update(signals, rows)
