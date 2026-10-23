"""CPU-only exact-token boundary checks; imports no transformers or model."""
from __future__ import annotations

import json

from alignment import OfficialBoundaryBuilder


def run_tokenizer_checks(tokenizer, builder=None):
    builder = builder or OfficialBoundaryBuilder(tokenizer)
    prompt = tokenizer.encode('Problem: Compute 2 + 2.\nAnswer:', add_special_tokens=False)
    eos = tokenizer.eos_token_id
    def completion(text, terminal=True):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids + ([eos] if terminal else [])
    cases = [
        ('no_separator', completion('The result is 4.')),
        ('crlf', completion('Add two.\r\n\r\nThe result is 4.')),
        ('multi_newline', completion('Add two.\n\n\n\nThe result is 4.')),
        ('empty_eos', [eos]),
        ('whitespace', completion('  \n \t')),
        ('truncated_no_eos', completion('The result is 4.', terminal=False)),
    ]
    if builder.separator_ids:
        cases.append(('consecutive_separator_tokens', [builder.separator_ids[0],
                                                     builder.separator_ids[0], eos]))
    text_delimiters = [i for i in builder.separator_ids if tokenizer.decode(i).strip()]
    if text_delimiters:
        cases.append(('delimiter_token_has_non_whitespace_text', [text_delimiters[0], eos]))
    details = []
    for name, ids in cases:
        row = builder.build(prompt, ids)
        assert row['oracle_passed'] and row['process_sequences_oracle_passed']
        assert row['actor_attention_mask'] == [True] * (len(prompt) + len(row['actor_completion_ids']))
        assert row['actor_action_mask'] == [True] * len(row['actor_completion_ids'])
        assert len(set(row['actor_boundary_positions'])) == row['step_count']
        assert row['prm_score_positions'][-1] == len(row['prm_input_ids']) - 1
        assert sum(row['num_tokens_per_step']) + row['step_count'] == row['response_tokens']
        if ids[-1] == eos:
            assert row['actor_completion_ids'] == ids
            assert row['decoded_text_unchanged']
        else:
            assert row['actor_completion_ids'] == ids[:-1] + [eos]
            assert row['process_sequences_changed']
        if name == 'empty_eos':
            assert row['num_tokens_per_step'] == [0] and row['step_count'] == 1
        if name == 'consecutive_separator_tokens':
            assert row['num_tokens_per_step'] == [0, 0, 0]
        if name == 'delimiter_token_has_non_whitespace_text':
            assert row['num_tokens_per_step'] == [0, 0]
        details.append(dict(case=name, passed=True, **row))
    # An EOS in the prompt is valid under official attention reconstruction.
    interior_prompt = prompt[:1] + [eos] + prompt[1:]
    row = builder.build(interior_prompt, completion('The result is 4.'))
    assert row['prompt_ids'] == interior_prompt and all(row['actor_attention_mask'])
    details.append(dict(case='prompt_interior_eos', passed=True, **row))
    return dict(passed=True, cases=len(details), all_literal_oracles_equal=True,
                duplicate_endpoints=0, new_generations=0, optimizer_steps=0,
                crlf_normalized=False, source_pin=builder.build(prompt, [eos])['source_pin'],
                separator_token_count=len(builder.separator_ids), details=details)


class _SyntheticTokenizer:
    """Small local test fixture; not evidence about a real model tokenizer."""
    def __init__(self):
        pieces = ['<PAD>', '<EOS>', '\n', '\n\n', '\r\n\r\n', '.\n\n', '\n\n\n\n']
        pieces += [chr(i) for i in range(9, 127) if chr(i) not in pieces]
        self.pieces = pieces
        self.vocab = {s: i for i, s in enumerate(pieces)}
        self.pad_token_id, self.eos_token_id = 0, 1
    def __len__(self): return len(self.pieces)
    def get_vocab(self): return dict(self.vocab)
    def decode(self, ids, skip_special_tokens=False):
        if type(ids) is int: ids = [ids]
        return ''.join(self.pieces[i] for i in ids if not (skip_special_tokens and i in (0, 1)))
    def encode(self, text, add_special_tokens=False):
        result = []
        ordered = sorted(self.vocab, key=len, reverse=True)
        while text:
            value = next(s for s in ordered if text.startswith(s))
            result.append(self.vocab[value]); text = text[len(value):]
        return result


if __name__ == '__main__':
    result = run_tokenizer_checks(_SyntheticTokenizer())
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2))
