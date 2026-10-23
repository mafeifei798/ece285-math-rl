"""Fixed final-checkpoint greedy evaluation; no PRM, reranking or test selection."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from support import ROOT, HERE, atomic, read, sha, verify_release


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def behavior(rows):
    lengths = sorted(row['generated_tokens'] for row in rows)
    n = len(rows)
    assert n and len(lengths) == n
    return dict(count=n, correct=sum(row['reward'] for row in rows),
                accuracy=sum(row['reward'] for row in rows) / n,
                generated_tokens=sum(lengths), mean_tokens=sum(lengths) / n,
                median_tokens=(lengths[(n - 1) // 2] + lengths[n // 2]) / 2,
                empty=sum(row['empty'] for row in rows),
                short=sum(row['short'] for row in rows),
                short_nonempty=sum(row['short'] and not row['empty'] for row in rows),
                short_no_box=sum(row['short'] and row['no_box'] for row in rows),
                short_incorrect=sum(row['short'] and not row['reward'] for row in rows),
                missing_or_invalid_box=sum(row['status'] == 'missing_or_invalid_box' for row in rows),
                truncated=sum(row['status'] == 'truncated' for row in rows),
                no_box=sum(row['no_box'] for row in rows),
                no_literal_double_lf=sum(row['no_literal_double_lf'] for row in rows),
                statuses=dict(sorted(Counter(row['status'] for row in rows).items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    cli = parser.parse_args()
    verify_release()
    protocol = read(HERE / 'protocol.json')
    run_paths = {(ROOT / value).resolve() for value in protocol['runs'].values()}
    output, adapter = cli.output.resolve(), cli.adapter.resolve()
    assert output.parent in run_paths and output.name == f'evaluation_step_{cli.step}'
    assert cli.step in protocol['stage_targets']
    assert adapter.is_dir() and adapter.is_relative_to(output.parent)
    assert adapter.name == 'adapter'
    output.mkdir(exist_ok=True)
    lock = (output / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (output / 'evaluation_summary.json').exists(), 'Evaluation complete; inspect it instead of rerunning'
    cfg, assets = read(ROOT / 'training/baseline_config.json'), read(ROOT / 'receipts/assets.json')
    manifest = read(ROOT / 'data/prepared/manifest.json')
    assert manifest['config_sha256'] == sha(ROOT / 'training/baseline_config.json')
    assert cfg['max_new_tokens'] == cfg['max_prompt_tokens'] == 2048
    assert assets['model_revision'] == '4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2'
    units_path = ROOT / 'audits/reward_repair_v3_20261021/unit_metadata.json'
    corrected_release = read(ROOT / 'experiments/ab_corrected_6m_seed17_20261022/release.json')
    assert sha(units_path) == corrected_release['unit_metadata_sha256']
    adapter_weights = adapter / 'adapter_model.safetensors'
    adapter_config = adapter / 'adapter_config.json'
    contract = dict(schema_version=1, checkpoint_step=cli.step,
        model_revision=assets['model_revision'], config_sha256=sha(ROOT / 'training/baseline_config.json'),
        data_manifest_sha256=sha(ROOT / 'data/prepared/manifest.json'),
        reward_code_sha256=sha(ROOT / 'training/reward.py'),
        repair_source_sha256=sha(ROOT / 'experiments/reward_repair_v3_20261021/repairs.py'),
        repair_core_sha256=sha(ROOT / 'experiments/reward_repair_v2_20261021/v2core.py'),
        unit_metadata_sha256=sha(units_path), verifier_version='reward-repair-v3-20261021',
        adapter_sha256=sha(adapter_weights), adapter_config_sha256=sha(adapter_config),
        evaluator_sha256=sha(Path(__file__)), release_sha256=sha(HERE / 'release.json'),
        versions={name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'vllm', 'peft']},
        engine=dict(dtype='bfloat16', gpu_memory_utilization=.6, max_model_len=4096,
                    max_num_seqs=64, max_num_batched_tokens=4096, seed=285,
                    enable_lora=True, max_lora_rank=cfg['lora_rank'], enforce_eager=True),
        sampling=dict(n=1, temperature=0., max_tokens=2048, min_tokens=0, top_p=1., top_k=-1,
                      seed=None, skip_special_tokens=True, include_stop_str_in_output=False),
        prompt='unchanged prepared dataset prompt', reranking=False, process_reward_used=False,
        selection='precommitted common stage checkpoint; no best-test-checkpoint selection',
        behavior_definitions=dict(empty='decoded completion.strip() is empty',
            short='raw generated token IDs excluding one evidenced terminal EOS <=32',
            no_box='literal backslash-boxed opening brace absent; diagnostic only',
            no_literal_double_lf='literal double LF absent; not an official step count'))
    existing_contract = output / 'evaluation_config.json'
    if existing_contract.exists():
        assert read(existing_contract) == contract, 'Resume contract differs; do not mix predictions'
    else:
        atomic(existing_contract, contract)
    sys.path.insert(0, str(ROOT / 'experiments/reward_repair_v3_20261021'))
    from repairs import repaired_outcome
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    units = read(units_path)
    start = time.time()
    llm = LLM(model=assets['model_path'], **contract['engine'])
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(**contract['sampling'])
    request = LoRARequest(adapter.parent.name, 1, str(adapter))
    summary = dict(model_revision=assets['model_revision'], adapter=str(adapter),
                   checkpoint_step=cli.step, evaluation_contract_sha256=sha(existing_contract),
                   greedy=True, process_reward_used=False, verifier_version=contract['verifier_version'],
                   started_unix=start, datasets={}, limitation=manifest.get('limitation'),
                   prm800k_known_question_overlap=manifest.get('prm800k_known_question_overlap'))
    for name, expected_count in [('math500', 500), ('gsm8k', 1319)]:
        data_path = ROOT / 'data/prepared' / f'{name}.jsonl'
        assert sha(data_path) == manifest['files'][data_path.name]['sha256']
        data = read_rows(data_path)
        assert len(data) == expected_count
        original = {row['problem_id']: row for row in data}
        assert len(original) == expected_count
        prediction_path = output / f'{name}.jsonl'
        saved = read_rows(prediction_path)
        by_id = {}
        for row in saved:
            key = row['problem_id']
            assert key in original and key not in by_id, 'Unknown or duplicate saved prediction'
            assert row['question_sha256'] == original[key]['question_sha256']
            assert row['gold'] == original[key]['answer']
            assert row['prompt_sha256'] == hashlib.sha256(original[key]['prompt'].encode()).hexdigest()
            by_id[key] = row
        todo = [row for row in data if row['problem_id'] not in by_id]
        with prediction_path.open('a') as stream:
            for offset in range(0, len(todo), 64):
                batch = todo[offset:offset + 64]
                answers = llm.generate([row['prompt'] for row in batch], params,
                                       lora_request=request, use_tqdm=False)
                assert len(answers) == len(batch)
                for row, result in zip(batch, answers):
                    answer = result.outputs[0]
                    ids = list(answer.token_ids)
                    judged = repaired_outcome(answer.text, row['answer'], answer.finish_reason == 'stop',
                        question=row['problem'], problem_id=row['problem_id'], unit_metadata=units)
                    content = len(ids) - int(bool(ids) and ids[-1] == tokenizer.eos_token_id)
                    record = dict(problem_id=row['problem_id'], question_sha256=row['question_sha256'],
                        prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),
                        gold=row['answer'], completion=answer.text, generated_token_ids=ids,
                        generated_tokens=len(ids), raw_content_tokens=content,
                        finish_reason=answer.finish_reason, stop_reason=answer.stop_reason,
                        empty=not answer.text.strip(), short=content <= 32,
                        no_box=re.search(r'\\boxed\s*\{', answer.text) is None,
                        no_literal_double_lf='\n\n' not in answer.text, **judged)
                    assert record['reward'] in (0, 1, 0., 1.)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    by_id[row['problem_id']] = record
                stream.flush()
                os.fsync(stream.fileno())
                atomic(output / 'progress.json', dict(dataset=name, done=len(by_id), total=expected_count,
                    checkpoint_step=cli.step, elapsed_seconds=time.time() - start))
                print(name, len(by_id), '/', expected_count, flush=True)
        assert set(by_id) == set(original)
        summary['datasets'][name] = dict(**behavior(list(by_id.values())),
            predictions_sha256=sha(prediction_path), dataset_sha256=sha(data_path))
    summary.update(finished_unix=time.time(), elapsed_seconds_this_invocation=time.time() - start,
                   adapter_hash_after=sha(adapter_weights))
    assert summary['adapter_hash_after'] == contract['adapter_sha256']
    atomic(output / 'evaluation_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
