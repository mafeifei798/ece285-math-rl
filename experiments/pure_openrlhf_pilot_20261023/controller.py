"""Bounded stage controller for the authorized paired PURE pilot only."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from support import ROOT, HERE, atomic, read, sha, verify_release


ACCEPTANCE = ('finite_gradients', 'parameters_updated', 'frozen_base_unchanged',
              'PRM_frozen', 'normalization_once', 'loss_normalization_zero', 'safe_vllm_sync')


def snapshot_gpu(gpu):
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    selected = [line for line in output.splitlines() if int(line.split(',')[0]) == gpu]
    assert len(selected) == 1
    fields = [int(value.strip()) for value in selected[0].split(',')]
    return dict(index=fields[0], total_mib=fields[1], free_mib=fields[2],
                utilization_percent=fields[3], raw=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorized-pilot', action='store_true', required=True)
    parser.parse_args()
    verify_release()
    protocol = read(HERE / 'protocol.json')
    assert protocol['stage_targets'] == [2, 10, 20, 40]
    assert set(protocol['runs']) == {'P_VR', 'P_PRM'}
    directory = ROOT / protocol['controller']
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = directory / 'controller.json'
    assert not state_file.exists(), 'Controller already has a receipt; inspect, do not silently restart'
    state = dict(status='starting', pid=os.getpid(), started_unix=time.time(),
                 release_sha256=sha(HERE / 'release.json'), protocol=protocol,
                 formal_6M_started=False, children=[], completed_stages=[])

    def update(**values):
        state.update(values)
        atomic(state_file, state)

    update()

    def wait_memory(required_mib, label):
        while True:
            snap = snapshot_gpu(int(protocol['gpu']))
            if snap['free_mib'] >= required_mib:
                update(status='ready', next_action=label, gpu_snapshot=snap)
                return
            update(status='waiting_for_memory', next_action=label, required_free_mib=required_mib,
                   gpu_snapshot=snap, last_checked_unix=time.time())
            time.sleep(30)

    def run_child(command, label, memory_mib=None):
        verify_release()
        if memory_mib is not None:
            wait_memory(memory_mib, label)
        logfile = directory / f'{label}.log'
        assert not logfile.exists(), 'Existing child log: inspect instead of a duplicate launch'
        with logfile.open('x') as output:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        receipt = dict(label=label, pid=child.pid, command=command, log=str(logfile),
                       started_unix=time.time(), exit_code=None)
        state['children'].append(receipt)
        update(status='running_child', active_child=receipt)
        code = child.wait()
        receipt.update(exit_code=code, finished_unix=time.time())
        update(active_child=None)
        if code != 0:
            raise RuntimeError(f'{label} exited with {code}; no automatic retry or budget extension')

    def checked_stage(arm, target):
        run = ROOT / protocol['runs'][arm]
        stage = read(run / f'stage_{target}.json')
        status = read(run / 'status.json')
        assert stage['batch_step'] == target
        assert status['status'] in ('stage_complete', 'completed') and status['batch_step'] == target
        assert stage['optimizer_steps'] == target, 'No skipped or substituted update is permitted'
        assert all(stage['acceptance'].get(key) is True for key in ACCEPTANCE), stage['acceptance']
        assert stage['release_sha256'] == state['release_sha256']
        assert 0 < stage['actual_generated_tokens'] <= target * 32 * 2048
        checkpoint = Path(stage['checkpoint']).resolve()
        assert checkpoint.is_relative_to(run.resolve())
        assert sha(checkpoint / 'adapter/adapter_model.safetensors') == stage['adapter_file_sha256']
        assert stage['adapter_sha256'] != stage['initial_adapter_sha256']
        assert type(stage['collapsed']) is bool and isinstance(stage['collapse_reasons'], list)
        return stage

    try:
        final_target = None
        for target in protocol['stage_targets']:
            stages = {}
            for arm in ['P_VR', 'P_PRM']:
                run_child([sys.executable, '-B', '-u', str(HERE / 'train_arm.py'),
                           '--arm', arm, '--target-step', str(target)],
                          f'{arm}_stage_{target}', memory_mib=56000)
                stages[arm] = checked_stage(arm, target)
            assert stages['P_VR']['initial_adapter_sha256'] == stages['P_PRM']['initial_adapter_sha256']
            assert stages['P_VR']['initial_samples_sha256'] == stages['P_PRM']['initial_samples_sha256']
            gate = dict(passed=True, common_step=target,
                        acceptance_keys=list(ACCEPTANCE),
                        initial_adapters_equal=True, initial_samples_equal=True,
                        fixed_updates_and_questions=True, same_actual_token_budget=False,
                        actual_generated_tokens={arm: stage['actual_generated_tokens'] for arm, stage in stages.items()},
                        collapsed={arm: stage['collapsed'] for arm, stage in stages.items()},
                        collapse_reasons={arm: stage['collapse_reasons'] for arm, stage in stages.items()},
                        stage_receipt_sha256={arm: sha(ROOT / protocol['runs'][arm] / f'stage_{target}.json') for arm in stages})
            atomic(directory / f'gate_{target}.json', gate)
            state['completed_stages'].append(target)
            update(status='stage_pair_complete', common_step=target, last_gate=gate)
            if any(stage['collapsed'] for stage in stages.values()) or target == protocol['stage_targets'][-1]:
                final_target = target
                update(stop_reason='predeclared_collapse' if any(stage['collapsed'] for stage in stages.values()) else 'fixed_40_update_budget',
                       common_evaluation_step=target)
                break
        assert final_target is not None
        for arm in ['P_VR', 'P_PRM']:
            stage = checked_stage(arm, final_target)
            run = ROOT / protocol['runs'][arm]
            output = run / f'evaluation_step_{final_target}'
            assert not (output / 'evaluation_summary.json').exists(), 'Unexpected preexisting pilot evaluation'
            run_child([sys.executable, '-B', '-u', str(HERE / 'evaluate.py'),
                       '--adapter', str(Path(stage['checkpoint']) / 'adapter'),
                       '--step', str(final_target), '--output', str(output)],
                      f'{arm}_eval_{final_target}', memory_mib=54000)
            assert (output / 'evaluation_summary.json').is_file()
        comparison = directory / 'comparison.json'
        run_child([sys.executable, '-B', '-u', str(HERE / 'compare.py'),
                   '--step', str(final_target), '--output', str(comparison)], 'paired_comparison')
        assert read(comparison)['passed']
        update(status='completed_pilot', finished_unix=time.time(), comparison=str(comparison),
               comparison_sha256=sha(comparison), formal_6M_started=False,
               further_training_requires_result_review=True)
    except Exception as exc:
        update(status='failed', failure=str(exc), traceback=traceback.format_exc(),
               failed_unix=time.time(), formal_6M_started=False,
               instruction='Preserve evidence; inspect the failed child before any explicit resume')
        raise


if __name__ == '__main__':
    main()
