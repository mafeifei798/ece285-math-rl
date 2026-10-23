"""Exact-token, nonpacked OpenRLHF input construction, fixed to 41f9363b.

The caller supplies actual generated token IDs, not decoded/re-tokenized text.
The default applies the official Actor.process_sequences method first, including
its replacement of a truncated last token with EOS. Both ID sequences are kept.
No CRLF normalization, textual step splitting, or empty-answer reward is added.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE_PIN = '41f9363b8cc4af044b7fc5f7f9bbbf62139e9553'
ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / 'execution/pure_official_fix_audit_20261023/source_openrlhf'
ACTOR_SOURCE = OFFICIAL_ROOT / 'openrlhf/models/actor.py'
EXPERIENCE_SOURCE = OFFICIAL_ROOT / 'openrlhf/trainer/ppo_utils/experience_maker.py'
EXPERIENCE_SHA = '05ae7184e2223ffeb3a329b05a551e884967264be76b5d7d756acad5409e30cb'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ids(values, name):
    if isinstance(values, torch.Tensor):
        if values.device.type != 'cpu':
            raise ValueError(f'{name} must be CPU IDs; no implicit GPU transfers')
        values = values.tolist()
    values = list(values)
    if not values or any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f'{name} must contain nonempty nonnegative integer IDs')
    return values


def _official_process_sequences(source=ACTOR_SOURCE):
    # Match against the independently saved public-source manifest before AST use.
    import json
    manifest_path = OFFICIAL_ROOT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['commit'] == SOURCE_PIN
    expected = [x['sha256'] for x in manifest['files'] if x['path'] == 'openrlhf/models/actor.py']
    if len(expected) != 1 or _sha(source) != expected[0]:
        raise ValueError('Official Actor source hash mismatch')
    tree = ast.parse(Path(source).read_text(), filename=str(source))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Actor')
    fn = copy.deepcopy(next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'process_sequences'))
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    return namespace['process_sequences'], expected[0]


def _independent_process_sequences(prompt, emitted, eos_id, pad_id):
    """List oracle for the pinned tensor method, including EOS overwrite."""
    sequence = list(prompt) + list(emitted)
    original_mask = [x not in (eos_id, pad_id) for x in sequence]
    active = [i for i, value in enumerate(original_mask) if value]
    first = active[0] if active else 0
    reverse_argmax = len(sequence) - 1 - active[-1] if active else 0
    eos_position = len(sequence) - max(reverse_argmax, 1)
    sequence[eos_position] = eos_id
    attention = [first <= i <= eos_position for i in range(len(sequence))]
    states = sequence[len(prompt) - 1:-1]
    action = [x not in (eos_id, pad_id) for x in states]
    action[0] = True
    return sequence, attention, action


def _literal_tensor_oracle(prompt_ids, actor_ids, separator_ids, newline_id):
    """Literal single-row tensor operations from _generate_vllm/make_experience.

    The builder uses list slicing below; this oracle separately executes the
    pinned tensor-index construction. It deliberately retains the official
    behavior of dropping an entire token whose decoded suffix is a separator.
    """
    solution = torch.tensor([actor_ids], dtype=torch.long)
    row, columns = torch.where(torch.isin(solution, torch.tensor(separator_ids, dtype=torch.long)))
    action_mask = torch.ones_like(solution, dtype=torch.bool)
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(1)
    num_steps = columns[row == 0].numel()
    score_ids = torch.full((1, num_steps + 1), -1, dtype=torch.long)
    reward_mask = torch.zeros_like(solution, dtype=torch.bool)
    score_ids[0, :num_steps] = columns[row == 0] + 1
    reward_mask[0, columns[row == 0]] = True
    score_ids[0, num_steps] = eos_indices[0] + 1
    reward_mask[0, eos_indices[0]] = True
    score_mask = score_ids != -1
    prm_input = torch.tensor(prompt_ids, dtype=torch.long)
    sizes = []
    for idx, j in enumerate(score_ids[0][score_mask[0]]):
        if idx == 0:
            step = solution[0, :j - 1]
            sizes.append(int(j - 1))
        else:
            step = solution[0, score_ids[0, idx - 1]:j - 1]
            sizes.append(int(j - score_ids[0, idx - 1] - 1))
        prm_input = torch.cat([prm_input, step, torch.tensor([newline_id])])
    pred_ids = score_ids - 1 + len(prompt_ids)
    return dict(actor_boundary_positions=torch.where(reward_mask[0])[0].tolist(),
                score_ids=score_ids[0].tolist(), num_tokens_per_step=sizes,
                prm_input_ids=prm_input.tolist(), prm_score_positions=pred_ids[0].tolist())


class OfficialBoundaryBuilder:
    """Cache the official vocabulary scan and construct exact per-row inputs."""

    def __init__(self, tokenizer):
        if _sha(EXPERIENCE_SOURCE) != EXPERIENCE_SHA:
            raise ValueError('Official experience-maker source hash mismatch')
        self.tokenizer = tokenizer
        # Intentionally identical to the official init: do not replace with a
        # text regex, special separator ID list, or decode-cleanup override.
        self.separator_ids = [i for i in range(len(tokenizer))
                              if tokenizer.decode(i).endswith('\n\n')]
        self.separator_set = set(self.separator_ids)
        newline_ids = tokenizer.encode('\n', add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError('The official position formula requires a single newline token')
        self.newline_id = int(newline_ids[0])
        self.eos_id, self.pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
        if type(self.eos_id) is not int or type(self.pad_id) is not int:
            raise ValueError('A single integer EOS and PAD ID are required by this official path')
        self.process_sequences, self.actor_source_sha = _official_process_sequences()

    def build(self, prompt_ids, completion_ids, *, apply_process_sequences=True):
        prompt = _ids(prompt_ids, 'prompt_ids')
        emitted = _ids(completion_ids, 'completion_ids')
        if max(prompt + emitted) >= len(self.tokenizer):
            raise ValueError('Token ID is outside the actor tokenizer vocabulary')
        if apply_process_sequences:
            sequence = torch.tensor([prompt + emitted], dtype=torch.long)
            processed, attention, action = self.process_sequences(
                SimpleNamespace(), sequence, len(prompt), self.eos_id, self.pad_id)
            independent = _independent_process_sequences(prompt, emitted, self.eos_id, self.pad_id)
            literal = (processed[0].tolist(), [bool(v) for v in attention[0].tolist()],
                       [bool(v) for v in action[0].tolist()])
            if literal != independent:
                raise AssertionError('Independent list/official tensor process_sequences disagreement')
            mask = action[0].tolist()
            valid_length = int(action[0].sum())
            if valid_length <= 0 or mask != [True] * valid_length + [False] * (len(mask) - valid_length):
                raise ValueError('Official nonpacked action mask is not a contiguous valid prefix')
            actor = processed[0, len(prompt):len(prompt) + valid_length].tolist()
            prompt_processed = processed[0, :len(prompt)].tolist()
            if prompt_processed != prompt:
                raise ValueError('Official sequence processing unexpectedly modified the prompt')
            if not all(attention[0, :len(prompt) + valid_length].tolist()):
                raise ValueError('Single-example prompt/action contains masked positions; supply unpadded IDs')
        else:
            actor = list(emitted)
            valid_length = len(actor)
        if actor[-1] != self.eos_id or self.eos_id in actor[:-1] or self.pad_id in actor[:-1]:
            raise ValueError('Expected the official processed action sequence with one terminal EOS and no padding')
        positions = [i for i, token in enumerate(actor) if token in self.separator_set]
        positions.append(len(actor) - 1)
        if len(set(positions)) != len(positions):
            raise ValueError('Official separator and EOS boundaries overlap')
        score_ids = [i + 1 for i in positions]
        previous = -1
        prm_ids = list(prompt)
        sizes, drop_ids, content_slices = [], [], []
        for end in positions:
            content = actor[previous + 1:end]
            sizes.append(len(content)); content_slices.append(content)
            drop_ids.append(actor[end])
            prm_ids.extend(content); prm_ids.append(self.newline_id)
            previous = end
        prm_positions = [len(prompt) + i for i in positions]
        values = dict(actor_boundary_positions=positions, score_ids=score_ids,
                      num_tokens_per_step=sizes, prm_input_ids=prm_ids,
                      prm_score_positions=prm_positions)
        oracle = _literal_tensor_oracle(prompt, actor, self.separator_ids, self.newline_id)
        if values != oracle:
            raise AssertionError('Independent list/tensor official serialization disagreement')
        assert len(prm_ids) == len(prompt) + len(actor)
        assert all(prm_ids[i] == self.newline_id for i in prm_positions)
        emitted_text = self.tokenizer.decode(emitted, skip_special_tokens=True)
        actor_text = self.tokenizer.decode(actor, skip_special_tokens=True)
        return dict(**values, prompt_ids=prompt, emitted_completion_ids=emitted,
                    actor_completion_ids=actor, response_tokens=valid_length,
                    actor_attention_mask=[True] * (len(prompt) + valid_length),
                    actor_action_mask=[True] * valid_length,
                    step_count=len(positions), step_content_ids=content_slices,
                    dropped_boundary_token_ids=drop_ids,
                    dropped_boundary_token_text=[self.tokenizer.decode(i) for i in drop_ids],
                    separator_token_ids=self.separator_ids, prm_newline_token_id=self.newline_id,
                    source_pin=SOURCE_PIN, source_hashes=dict(actor=self.actor_source_sha,
                                                            experience_maker=EXPERIENCE_SHA),
                    process_sequences_applied=apply_process_sequences,
                    process_sequences_changed=actor != emitted,
                    process_sequences_oracle_passed=bool(apply_process_sequences),
                    emitted_ended_in_eos=emitted[-1] == self.eos_id,
                    terminal_replaced_with_eos=bool(apply_process_sequences and emitted[-1] != self.eos_id),
                    emitted_text=emitted_text, actor_text=actor_text,
                    decoded_text_unchanged=emitted_text == actor_text,
                    empty_decoded=not actor_text.strip(),
                    original_token_ids=True, text_retokenized=False, crlf_normalized=False,
                    serialization='official actor prompt IDs plus token-sliced steps with one newline each',
                    oracle_passed=True)


def build_official_alignment(tokenizer, prompt_ids, completion_ids, *, apply_process_sequences=True):
    return OfficialBoundaryBuilder(tokenizer).build(prompt_ids, completion_ids,
                                                  apply_process_sequences=apply_process_sequences)
