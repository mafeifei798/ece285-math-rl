"""Native PRM character spans -> original actor byte-BPE token positions.

No retokenization of a generated completion is used to infer token indices.
Historical logs without token IDs must explicitly use the reconstructed mode.
"""
from bisect import bisect_left
import hashlib
import re
import sys
from common import ROOT

sys.path.insert(0, str(ROOT / 'experiments/reward_repair_v2_20261021'))
from v2core import canonical_view

SYSTEM = 'Please reason step by step, and put your final answer within \\boxed{}.'

def byte_decoder():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

BYTE_DECODER = byte_decoder()

def native_input(tokenizer, question, raw, max_length=4096):
    if '<extra_0>' in question or '<extra_0>' in raw:
        raise ValueError('Literal scoring marker in source text')
    view = canonical_view(raw, True)
    empty = not view['steps']
    steps = view['steps'] or ['']
    spans = view['spans'] or [dict(raw_start=0, raw_end=len(raw), canonical_start=0, canonical_end=len(view['text']))]
    marker = tokenizer.encode('<extra_0>', add_special_tokens=False)
    assert len(marker) == 1
    serialized = tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': '<extra_0>'.join(steps) + '<extra_0>'}],
        tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(serialized, add_special_tokens=False)
    positions = [i for i, token in enumerate(ids) if token == marker[0]]
    assert len(positions) == len(steps)
    if len(ids) > max_length:
        raise ValueError(f'Native PRM context overflow: {len(ids)} > {max_length}')
    return dict(input_ids=ids, positions=positions, steps=steps, spans=spans,
                raw_sha256=view['raw_sha256'], crlf_replacements=view['crlf_replacements'],
                empty_terminal_step=empty,
                input_sha256=hashlib.sha256(serialized.encode()).hexdigest())

def align_actor(tokenizer, ids, raw, spans, *, empty_terminal_step=False):
    if not ids:
        raise ValueError('No generated tokens: cannot assign a terminal reward')
    if len(ids) > 2048:
        raise ValueError('Response exceeds fixed completion width')
    eos = tokenizer.eos_token_id
    if eos in ids and ids.index(eos) != len(ids) - 1:
        raise ValueError('Tokens after EOS are not valid generated response tokens')
    decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if decoded != raw:
        raise ValueError('Raw response differs from decoding original token IDs')
    special = set(tokenizer.all_special_ids)
    pieces, ends = [], []
    total = 0
    for token_id in ids:
        if token_id in special:
            piece = b''
        else:
            token = tokenizer.convert_ids_to_tokens(token_id)
            try:
                piece = bytes(BYTE_DECODER[c] for c in token)
            except KeyError as exc:
                raise ValueError('Unsupported non-byte-BPE token') from exc
        total += len(piece); pieces.append(piece); ends.append(total)
    if b''.join(pieces) != raw.encode('utf-8'):
        raise ValueError('Lossless UTF-8 byte alignment failed; no guessed mapping')
    if empty_terminal_step:
        assert len(spans) == 1 and not raw.strip()
        positions = [len(ids) - 1]
    else:
        positions = []
        for i, span in enumerate(spans):
            assert 0 <= span['raw_start'] < span['raw_end'] <= len(raw)
            end_byte = len(raw[:span['raw_end']].encode('utf-8'))
            if i == len(spans) - 1:
                pos = len(ids) - 1  # EOS if complete, actual last token otherwise.
            else:
                # Official PURE assigns the intermediate reward to the step
                # separator token, not the preceding content token. Include
                # the raw CRLF/LF separator, while native PRM content is stripped.
                gap = raw[span['raw_end']:spans[i + 1]['raw_start']]
                separator = re.search(r'\r?\n\r?\n', gap)
                if separator is None:
                    raise ValueError('Missing raw step separator')
                boundary = span['raw_end'] + separator.end()
                end_byte = len(raw[:boundary].encode('utf-8'))
                pos = bisect_left(ends, end_byte)
                if ends[pos] > len(raw[:spans[i + 1]['raw_start']].encode('utf-8')):
                    raise ValueError('One actor token crosses two mathematical steps')
            positions.append(pos)
    if sorted(set(positions)) != positions or len(positions) != len(spans):
        raise ValueError('Step scores collide or are reordered')
    assert all(0 <= p < len(ids) for p in positions)
    return dict(actor_positions=positions, native_steps=len(spans),
                actor_tokens=len(ids), terminated=bool(ids[-1] == eos),
                exact_original_token_mapping=True, byte_roundtrip=True,
                placement='nonfinal step-separator token; final EOS or actual last token')
