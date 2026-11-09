"""Independent prefix-protection signals; no model load or training on import.

All arms share the original C-bounded scalar. Only the policy advantage changes;
KL, completion masks, original lengths, clipping and reward verification do not.
"""
from pathlib import Path
import importlib.util
import math
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/reward_repair_v2_20261021'))
from v2core import canonical_view
spec = importlib.util.spec_from_file_location('frozen_c_bounded_advantages', ROOT/'experiments/abc_seed17_resumed_v1_20261104/advantages.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

def byte_decoder():
    bs = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    chars = list(bs)
    for b in range(256):
        if b not in bs:
            bs.append(b); chars.append(256 + len(chars) - 188)
    # Use the canonical construction explicitly; the number of initial symbols
    # is 188 and every remaining byte maps monotonically from code point 256.
    return {chr(c):b for b,c in zip(bs,chars)}

class BoundaryError(ValueError):
    pass

def prefix_token_count(tokenizer, ids, raw, raw_cut):
    """Locate a raw character boundary in ORIGINAL byte-BPE token IDs.

No encode(decode(ids)) round trip is used to invent offsets. Tokens crossing the
boundary remain penalized. Exactly one trailing EOS is required and preserved.
"""
    ids = list(ids)
    eos = tokenizer.eos_token_id
    if not ids or ids[-1] != eos or any(x in tokenizer.all_special_ids for x in ids[:-1]):
        raise BoundaryError('special_token_or_EOS_layout')
    if tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != raw:
        raise BoundaryError('decoded_text_mismatch')
    mapping = byte_decoder()
    chunks=[]
    for token in tokenizer.convert_ids_to_tokens(ids[:-1]):
        try: chunks.append(bytes(mapping[c] for c in token))
        except (KeyError,TypeError): raise BoundaryError('not_byte_BPE')
    if b''.join(chunks) != raw.encode('utf-8'):
        raise BoundaryError('byte_reconstruction_mismatch')
    cut = len(raw[:raw_cut].encode('utf-8'))
    consumed=0; keep=0
    for chunk in chunks:
        consumed += len(chunk)
        if consumed <= cut: keep += 1
        else: break
    if keep >= len(ids): raise BoundaryError('EOS_would_be_masked')
    return keep

def signals(records, tokenizer, *, device='cpu', threshold=.8):
    if threshold != .8: raise ValueError('Fixed externally motivated threshold; no silent tuning')
    if not records or len(records)%8: raise ValueError('Complete groups of eight required')
    lengths=[len(r['completion_token_ids']) for r in records]
    if any(n<1 for n in lengths): raise ValueError('Empty raw token stream')
    y=torch.tensor([r['outcome_reward'] for r in records],dtype=torch.float32,device=device)
    base=old.compute_advantages(y,[r['min_score'] for r in records],valid=[r['valid_for_process_reward'] for r in records])['advantages']
    width=max(lengths)
    mask=torch.arange(width,device=device)[None,:]<torch.tensor(lengths,device=device)[:,None]
    original=base[:,None].expand(-1,width).clone()*mask
    prefix=original.clone(); uniform=original.clone(); annotations=[]
    for group_start in range(0,len(records),8):
        group=records[group_start:group_start+8]
        if len({(r['problem_id'],r['optimizer_step_before_update']) for r in group}) != 1:
            raise ValueError('Changed group membership')
        count=sum(r['outcome_reward'] for r in group)
        for i in range(group_start,group_start+8):
            r=records[i]; note={'protected_tokens':0,'first_error_step':None,'status':'unchanged'}
            if r['outcome_reward']: note['status']='correct_unchanged'
            elif count==0: note['status']='all_wrong_unchanged'
            elif not r['terminated'] or not r['valid_for_process_reward']: note['status']='invalid_fallback'
            else:
                scores=r['step_scores']; view=canonical_view(r['completion'],True)
                if not scores or len(scores)!=len(view['steps']) or any(not math.isfinite(s) or not 0<=s<=1 for s in scores):
                    note['status']='step_scores_fallback'
                else:
                    k=next((j for j,s in enumerate(scores) if s<threshold),None)
                    note['first_error_step']=k
                    if k is None: note['status']='no_low_score_fallback'
                    elif k==0: note['status']='first_step_unchanged'
                    else:
                        try:
                            n=prefix_token_count(tokenizer,r['completion_token_ids'],r['completion'],view['spans'][k]['raw_start'])
                            if not n: raise BoundaryError('no_complete_prefix_token')
                            prefix[i,:n]=0
                            uniform[i,:lengths[i]] *= (lengths[i]-n)/lengths[i]
                            note.update(status='prefix_protected',protected_tokens=n,raw_error_start=view['spans'][k]['raw_start'])
                        except BoundaryError as exc: note['status']='alignment_fallback:'+str(exc)
            annotations.append(note)
    for v in (original,prefix,uniform):
        assert v.dtype==torch.float32 and not v.requires_grad and torch.isfinite(v).all()
        assert not (v[y==0]>0).any() and not v[~mask].any()
        assert torch.equal(v[y==1],original[y==1])
    torch.testing.assert_close(prefix.abs().sum(1),uniform.abs().sum(1),atol=2e-3,rtol=2e-6)
    return {'bounded':original,'uniform':uniform,'prefix':prefix,'completion_mask':mask,
            'scalar':base,'annotations':annotations,'normalizations_after_scalar':0}
