"""Frozen PURE PRM with literal OpenRLHF nonpacked input/head semantics.

This module scores already aligned token IDs. It neither generates responses nor
adds rewards, penalties, masking, softmin aggregation, or advantage normalization.
Every forward has batch size one and no padding. We preserve native bf16 logits
so the separately pinned signal path can reproduce official arithmetic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

REVISION = 'b0cce6763f2317762c9cb46720c8fe7b07f44a88'
SOURCE_PIN = '41f9363b8cc4af044b7fc5f7f9bbbf62139e9553'
MODEL_SOURCE_SHA = 'c6c7c67bb5684de89ae21637b3f005164a3dbd2f5dd5e6884f840bd9041589c9'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


class FrozenOfficialPRM:
    def __init__(self, model_path, actor_tokenizer, *, device='cuda:0', assets_path,
                 receipt_dir=None):
        from safetensors import safe_open
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        start = time.perf_counter()
        self.path, self.device = Path(model_path), torch.device(device)
        self.receipt_dir = Path(receipt_dir) if receipt_dir else None
        self.assets_path = Path(assets_path)
        assets = json.loads(self.assets_path.read_text())
        if not assets.get('complete') or assets.get('revision') != REVISION:
            raise ValueError('The existing complete, pinned PURE PRM asset receipt is required')
        if assets.get('repo') not in ('jinachris/PURE-PRM-7B', 'jinachris/Qwen2.5-Math-7B-PRM800K'):
            raise ValueError('Unexpected model repository in PURE PRM receipt')
        verified_assets = []
        for item in assets['files']:
            path = self.path / item['file']
            if not path.is_file() or path.stat().st_size != item['bytes'] or _sha(path) != item['sha256']:
                raise ValueError(f'Model asset mismatch: {item["file"]}')
            verified_assets.append(dict(file=item['file'], sha256=item['sha256'], bytes=item['bytes']))
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True,
                                                      padding_side='right')
        # PRM EOS/PAD metadata can differ: these are never inserted into its
        # input. What must match is the meaning of every exact actor token ID.
        if self.tokenizer.get_vocab() != actor_tokenizer.get_vocab():
            raise ValueError('Actor and PRM token-ID vocabularies differ; do not silently re-tokenize')
        self.newline_ids = self.tokenizer.encode('\n', add_special_tokens=False)
        if len(self.newline_ids) != 1 or self.newline_ids != actor_tokenizer.encode('\n', add_special_tokens=False):
            raise ValueError('Actor and PRM newline token IDs differ')

        self.model, loading = AutoModelForTokenClassification.from_pretrained(
            self.path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', output_loading_info=True, num_labels=2)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError(f'Incomplete or mismatched model loading: {loading}')
        unexpected = set(loading.get('unexpected_keys', []))
        if not unexpected <= {'lm_head.weight'}:
            raise ValueError(f'Unexpected PRM checkpoint keys: {sorted(unexpected)}')
        index = json.loads((self.path / 'model.safetensors.index.json').read_text())['weight_map']
        state = self.model.state_dict()
        if set(state) != set(index) - unexpected:
            raise ValueError('Loaded model state keys do not match the frozen checkpoint')
        checked = 0
        for shard in sorted(set(index.values())):
            with safe_open(self.path / shard, framework='pt', device='cpu') as weights:
                for key in weights.keys():
                    if key in state:
                        if not torch.equal(state[key], weights.get_tensor(key)):
                            raise ValueError(f'Loaded tensor differs from checkpoint: {key}')
                        checked += 1
        del state
        if not hasattr(self.model, 'model') or not hasattr(self.model, 'score') or not hasattr(self.model, 'dropout'):
            raise ValueError('Expected native Qwen2 TokenClassification model, dropout and score head')
        if self.model.score.out_features != 2 or self.model.score.bias is None:
            raise ValueError('Official PURE PRM uses a two-class linear head with bias')
        self.model = self.model.to(self.device).requires_grad_(False).eval()
        self.model.config.use_cache = False
        self.versions = {name: p._version for name, p in self.model.named_parameters()}
        self.verify_frozen()
        self.max_context = int(self.model.config.max_position_embeddings)
        self.loading_receipt = dict(
            passed=True, repository=assets['repo'], revision=REVISION,
            assets_receipt_sha256=_sha(self.assets_path), verified_assets=verified_assets,
            matched_checkpoint_tensors=checked, loading=loading,
            model_class=type(self.model).__name__, dtype=str(next(self.model.parameters()).dtype),
            device=str(self.device), attention='sdpa', batch_size=1, padding=False,
            tokenizer_vocab_identical=True, newline_token_ids=self.newline_ids,
            actor_eos_token_id=actor_tokenizer.eos_token_id,
            prm_eos_token_id=self.tokenizer.eos_token_id,
            actor_pad_token_id=actor_tokenizer.pad_token_id,
            prm_pad_token_id=self.tokenizer.pad_token_id,
            explicit_position_ids='attention_mask.cumsum(-1)-1; masked positions set to 1',
            normalize_reward=False, normalize_reward_reason='Pinned official CLI default, no flag in train_pure.sh',
            source_pin=SOURCE_PIN, official_model_source_sha256=MODEL_SOURCE_SHA,
            torch_version=torch.__version__, max_position_embeddings=self.max_context,
            frozen=True, optimizer_steps=0, elapsed_seconds=time.perf_counter() - start)
        if self.receipt_dir:
            _atomic(self.receipt_dir / 'prm_loading.json', self.loading_receipt)

    def verify_frozen(self):
        if self.model.training:
            raise AssertionError('PRM must remain in eval mode')
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad or parameter.grad is not None or parameter._version != self.versions[name]:
                raise AssertionError(f'PRM parameter updated or unfrozen: {name}')
        return True

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def score(self, alignments, *, verify_literal_head=True):
        """Return one record per alignment, using no padding or batching.

        Native bf16 step logits are the primary signal input. Float32 softmax is
        descriptive only; native bf16 softmax/signed scores are saved separately.
        verify_literal_head repeats the first forward through the exact official
        backbone -> eval-dropout -> score head path and requires bitwise equality.
        """
        self.verify_frozen()
        results = []
        for row_index, aligned in enumerate(alignments):
            if aligned.get('source_pin') != SOURCE_PIN or not aligned.get('oracle_passed'):
                raise ValueError('Only pinned, oracle-verified exact-ID alignment is accepted')
            values, positions = aligned['prm_input_ids'], aligned['prm_score_positions']
            if not 0 < len(values) <= self.max_context:
                raise ValueError('PRM context overflow/empty input; no truncation or substitution is allowed')
            if not positions or len(set(positions)) != len(positions) or positions != sorted(positions):
                raise ValueError('PRM score positions must be unique and increasing')
            if any(i < 0 or i >= len(values) or values[i] != self.newline_ids[0] for i in positions):
                raise ValueError('PRM score position does not point at the official single-newline token')
            ids = torch.tensor([values], dtype=torch.long, device=self.device)
            attention = torch.ones_like(ids)
            position_ids = attention.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention == 0, 1)
            self._sync(); start = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=attention,
                                    position_ids=position_ids, use_cache=False).logits
                if logits.shape != (1, len(values), 2) or not torch.isfinite(logits).all():
                    raise AssertionError('Invalid two-class PRM output')
                selected = logits[0, positions]
                native_probabilities = selected.softmax(-1)
                float_probabilities = selected.float().softmax(-1)
                native_signed = native_probabilities[:, 1] - native_probabilities[:, 0]
                self._sync(); seconds = time.perf_counter() - start
                literal_equal, literal_seconds = None, 0.0
                if verify_literal_head and row_index == 0:
                    self._sync(); literal_start = time.perf_counter()
                    # Same operations as official ProcessRewardModel.forward,
                    # with normalize_reward=False and model.eval().
                    outputs = self.model.model(input_ids=ids, attention_mask=attention,
                                               position_ids=position_ids, use_cache=False)
                    literal = self.model.score(self.model.dropout(outputs.last_hidden_state))
                    literal_equal = torch.equal(logits, literal)
                    if not literal_equal:
                        raise AssertionError('Native token-classification and literal official head differ')
                    self._sync(); literal_seconds = time.perf_counter() - literal_start
                    del outputs, literal
                results.append(dict(
                    step_logits=selected.float().cpu().tolist(), step_logits_dtype=str(selected.dtype),
                    step_scores=float_probabilities[:, 1].cpu().tolist(),
                    signed_scores=(float_probabilities[:, 1] - float_probabilities[:, 0]).cpu().tolist(),
                    native_step_scores=native_probabilities[:, 1].float().cpu().tolist(),
                    native_signed_scores=native_signed.float().cpu().tolist(),
                    score_positions=list(positions), prm_input_tokens=len(values),
                    forward_seconds=seconds, literal_head_equal=literal_equal,
                    literal_head_seconds=literal_seconds, batch_size=1, padding=False,
                    normalize_reward=False, model_frozen=True, optimizer_steps=0,
                    source_pin=SOURCE_PIN, model_revision=REVISION))
            del ids, attention, position_ids, logits, selected, native_probabilities, float_probabilities, native_signed
        self.verify_frozen()
        return results
