"""Frozen Qwen scoring of saved responses only; no policy, sampling or optimizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pure_integration_20261022'))
from common import atomic, read, read_jsonl, sha, verify_release
from alignment import SYSTEM, align_actor, canonical_view


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # Never overwrite/resume an ambiguous GPU job or its evidence.
    a.output.mkdir(parents=True, exist_ok=False)
    atomic(a.output / 'status.json', dict(status='starting', pid=os.getpid(),
        mode='frozen_prm_only_saved_responses', optimizer_steps=0, new_generations=0))
    verify_release()
    before = sha(ROOT / 'experiments/pure_integration_20261022/release.json')
    rows = read_jsonl(a.input)
    assert len(rows) == 288 and sum(not r['completion'].strip() for r in rows) == 9

    import torch
    from transformers import AutoTokenizer
    from prm import PureQwenPRM
    torch.set_num_threads(2)
    t = time.perf_counter()
    scorer = PureQwenPRM(ROOT / 'models/qwen25_math_prm_7b', receipt_dir=a.output)
    actor = AutoTokenizer.from_pretrained(read(ROOT / 'receipts/assets.json')['model_path'],
                                         local_files_only=True)
    costs, written = [], []
    with (a.output / 'scored.jsonl').open('x') as f, torch.inference_mode():
        for off in range(0, len(rows), 32):
            batch = rows[off:off + 32]
            scores, cost = scorer.score([r['question'] for r in batch],
                                        [r['completion'] for r in batch])
            costs.append(cost)
            for i, (r, native) in enumerate(zip(batch, scores)):
                # Literal Qwen model-card serialization: even [] joins to a
                # bare marker. This check does not assume an empty-step contract.
                steps = canonical_view(r['completion'], True)['steps']
                messages = [dict(role='system', content=SYSTEM),
                            dict(role='user', content=r['question']),
                            dict(role='assistant', content='<extra_0>'.join(steps) + '<extra_0>')]
                text = scorer.tokenizer.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=False)
                direct_ids = scorer.tokenizer.encode(text)
                assert direct_ids == native['input_ids']
                ids = actor.encode(r['completion'], add_special_tokens=False)
                if r['terminated']:
                    ids.append(actor.eos_token_id)
                result = dict(index=off + i, input_sha256=hashlib.sha256(r['completion'].encode()).hexdigest(),
                    native=native, reconstructed_ids=ids,
                    logged_length_matches=(len(ids) == r['completion_tokens_logged']),
                    token_ids_provenance='reconstructed; historical original emitted IDs were not saved',
                    direct_official_serialization_equal=True, empty=(not r['completion'].strip()))
                try:
                    result['alignment'] = align_actor(actor, ids, r['completion'], native['spans'],
                                                       empty_terminal_step=native['empty_terminal_step'])
                    result['alignment']['exact_original_token_mapping'] = False
                    result['alignment']['reconstructed_token_mapping'] = True
                except (ValueError, AssertionError, IndexError) as exc:
                    result['alignment_error'] = repr(exc)
                if result['empty']:
                    # Independent model-card forward, not reusing .score's
                    # extraction or padded batch. This is one CPU-saved answer.
                    inp = torch.tensor([direct_ids], dtype=torch.long, device=scorer.device)
                    out = scorer.model(input_ids=inp, use_cache=False)
                    logits = out.logits if hasattr(out, 'logits') else out[0]
                    marker = scorer.tokenizer.encode('<extra_0>')[0]
                    pos = [j for j, v in enumerate(direct_ids) if v == marker]
                    direct = logits[0, pos].float().softmax(-1)[:, 1].cpu().tolist()
                    result['direct_official_positive_probabilities'] = direct
                    result['direct_vs_adapter_max_abs_error'] = max(abs(x-y) for x,y in zip(direct,native['step_scores']))
                    assert result['direct_vs_adapter_max_abs_error'] < .001
                    result['model_card_bfloat16_softmax'] = logits[0, pos].softmax(-1)[:, 1].float().cpu().tolist()
                written.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
            print(f'Frozen scoring {len(written)}/{len(rows)}; zero sampling/updates', flush=True)
    scorer.assert_frozen()
    assert before == sha(ROOT / 'experiments/pure_integration_20261022/release.json')
    atomic(a.output / 'result.json', dict(passed=True, rows=len(written),
        empty_rows=sum(r['empty'] for r in written), all_official_input_ids_equal=True,
        max_direct_adapter_abs_error=max(r.get('direct_vs_adapter_max_abs_error', 0) for r in written),
        reconstruction_length_mismatches=sum(not r['logged_length_matches'] for r in written),
        alignment_failures=sum('alignment_error' in r for r in written),
        source_release_sha256=before, input_sha256=sha(a.input), costs=costs,
        elapsed_seconds=time.perf_counter()-t, model_frozen=True, optimizer_steps=0,
        policy_model_loaded=False, new_generations=0,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3))
    atomic(a.output / 'status.json', dict(status='completed', pid=os.getpid(),
        optimizer_steps=0, new_generations=0))


if __name__ == '__main__':
    main()
