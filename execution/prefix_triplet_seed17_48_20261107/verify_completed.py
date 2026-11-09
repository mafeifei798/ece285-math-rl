"""Local CPU receipt verification and descriptive case analysis; no model calls."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
from itertools import product
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
E = HERE / 'completed_evidence'
ARMS = ('source', 'bounded', 'uniform', 'prefix')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    manifest = read(HERE / 'completed_manifest.json')
    for name, item in manifest['files'].items():
        raw = (E / name).read_bytes()
        assert len(raw) == item['bytes']
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], name
    data = {arm: rows(E / 'evaluation' / arm / 'predictions.jsonl') for arm in ARMS}
    questions = rows(E / 'admission/validation.jsonl')
    by_id = {q['problem_id']: q for q in questions}
    assert len(by_id) == len(questions) == 500
    expected_ids = [q['problem_id'] for q in questions]
    comparison = read(E / 'comparison.json')
    for arm, records in data.items():
        assert [r['problem_id'] for r in records] == expected_ids
        for r, q in zip(records, questions):
            assert r['gold'] == q['answer']
            assert r['question_sha256'] == q['question_sha256']
            assert r['reward'] in (0, 1)
            assert r['generated_tokens'] == len(r['completion_token_ids'])
        assert sum(r['reward'] for r in records) == comparison['validation_summary'][arm]['correct']

    failures = {arm: dict(Counter(r['status'] for r in rr if not r['reward'])) for arm, rr in data.items()}
    discordant = []
    for u, p in zip(data['uniform'], data['prefix']):
        if u['reward'] == p['reward']:
            continue
        winner = 'prefix' if p['reward'] else 'uniform'
        loser = u if p['reward'] else p
        discordant.append({'problem_id': p['problem_id'], 'winner': winner,
                           'loser_status': loser['status'], 'loser_finish_reason': loser['finish_reason']})
    assert Counter(r['winner'] for r in discordant) == {'prefix': 18, 'uniform': 17}

    controls = {}
    orders = []
    for arm in ARMS[1:]:
        batches = rows(E / arm / 'generation_checks.jsonl')
        updates = rows(E / arm / 'update_audits.jsonl')
        result = read(E / arm / 'result.json')
        assert len(batches) == len(updates) == 48
        assert [r['global_step'] for r in updates] == list(range(289, 337))
        assert result['passed'] and result['source_checkpoint_unchanged']
        assert result['actor_base_unchanged'] and result['PRM_frozen']
        assert len(rows(E / arm / 'rollouts.jsonl')) == 1536
        orders.append([b['question_ids'] for b in batches])
        assert all(b['wrong_positive_tokens'] == b['post_scalar_normalizations'] == 0 for b in batches)
        counts = Counter()
        for b in batches:
            counts.update(b['statuses'])
        controls[arm] = {'updates': 48, 'rows': 1536, 'wrong_positive_tokens': 0,
                         'extra_normalizations': 0, 'alignment_fallback_rows': sum(v for k, v in counts.items() if k.startswith('alignment_fallback:'))}
    assert orders[0] == orders[1] == orders[2] == read(E / 'admission/preflight.json')['question_groups']

    # Independently written small calculations, NOT execution of model-generated code.
    triangle = 0
    unweighted = 0
    for bits in product((0, 1), repeat=11):
        triangle += sum(math.comb(10, j) * bit for j, bit in enumerate(bits)) % 3 == 0
        unweighted += sum(bits) % 3 == 0
    assert triangle == 640 and unweighted == 683
    assert Fraction(120, math.prod(range(10, 15))) == Fraction(1, 2002)
    x, y = Fraction(3, 5), Fraction(7, 9)
    assert (5*x + 9*y) / (45*x*y) == Fraction(10, 21)
    def valuation2(n):
        return (n & -n).bit_length() - 1
    assert valuation2(10**6 + 64) == 7 and valuation2(10**7 + 64) == 6
    notes = [
        ('math/train/counting_and_probability/393', 'prefix',
         '前缀保护用逐层求和的算法得到640；均匀减罚把顶层错误地当成底层普通求和，得到683。独立枚举确认640。此例有实质建模差异，不能据此证明收益由训练中的首错定位引起。'),
        ('math/train/intermediate_algebra/152', 'prefix',
         '两者正文均推到b-c=-1，均匀减罚随后擅自改成绝对值并框出1。前缀保护最终框出-1，但开头仍有一个错误的1框，不能称整个回答始终一致。'),
        ('math/train/algebra/164', 'uniform',
         '均匀减罚正确得到10/21；前缀保护把45xy写成(45x)(45y)，多乘一次45，得到2/189。这是数学错误，不是验答器漏识别。'),
        ('math/train/number_theory/658', 'uniform',
         '均匀减罚终答7正确，但其“k=5时有7个因子2”的解释错误；正确最大值在k=4取得，k=5时只有6。前缀保护答6且正文也有错误。终答得分不等于过程有效性。'),
    ]
    cases = []
    maps = {a: {r['problem_id']: r for r in records} for a, records in data.items()}
    for pid, winner, note in notes:
        assert maps[winner][pid]['reward'] == 1
        cases.append({'problem_id': pid, 'winner': winner, 'question': by_id[pid]['problem'], 'gold': by_id[pid]['answer'],
                      'note': note, 'responses': {a: {k: maps[a][pid][k] for k in ('completion', 'answer', 'reward', 'status', 'generated_tokens')} for a in ('uniform', 'prefix')}})
    summary = {'passed': True, 'manifest_files_verified': len(manifest['files']), 'gpu_processes_after_completion': manifest['gpu_after_completion'],
               'same_question_order': True, 'training_controls': controls, 'validation_wrong_statuses': failures,
               'discordant_pairs': discordant,
               'discordant_loser_types': dict(Counter(r['winner'] + '/' + r['loser_status'] for r in discordant)),
               'case_selection': 'Four qualitative examples from hash-ordered three wins and three losses; not a blind or representative process-validity audit.',
               'cases': cases,
               'interpretation': 'Prefix exceeds uniform by one answer; no established location-specific advantage. Both observed deltas over bounded have confidence intervals crossing zero. Do not claim the methods equivalent or process validity improved.'}
    (HERE / 'completion_verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('cases', 'discordant_pairs')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
