"""Official pinned frozen Qwen PRM scoring, with the already-audited raw splitter."""
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

from support import ROOT, atomic
sys.path.insert(0, str(ROOT / "experiments/qwen7b_offline_20261021"))
from core import encode_qwen


class FrozenQwenPRM:
    def __init__(self, path, *, receipt_dir, microbatch_size=4, device="cuda:0"):
        self.device = torch.device(device)
        self.microbatch_size = microbatch_size
        self.receipt_dir = Path(receipt_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="right")
        model, loading = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
        assert not loading["missing_keys"] and not loading["mismatched_keys"]
        assert set(loading["unexpected_keys"]) == {"lm_head.weight"}
        index = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
        state = model.state_dict()
        assert set(state) == set(index) - {"lm_head.weight"}
        checked = 0
        for shard in sorted(set(index.values())):
            with safe_open(Path(path) / shard, framework="pt", device="cpu") as weights:
                for key in weights.keys():
                    if key != "lm_head.weight":
                        assert torch.equal(state[key], weights.get_tensor(key)), key
                        checked += 1
        del state
        self.model = model.to(self.device).requires_grad_(False).eval()
        self.model.config.use_cache = False
        self.max_length = min(4096, self.model.config.max_position_embeddings)
        self.versions = {n: p._version for n, p in self.model.named_parameters()}
        self.assert_frozen()
        atomic(self.receipt_dir / "prm_loading.json", {**loading,
            "required_scoring_tensors_exactly_matched": checked, "expected_unused_head": "lm_head.weight",
            "frozen": True, "context": self.max_length, "microbatch": microbatch_size,
            "native_marker_id": self.tokenizer.encode("<extra_0>", add_special_tokens=False),
            "normalize_crlf": False})

    def assert_frozen(self):
        assert not self.model.training
        for name, p in self.model.named_parameters():
            assert not p.requires_grad and p.grad is None and p._version == self.versions[name], name

    @torch.inference_mode()
    def score(self, questions, completions):
        assert len(questions) == len(completions)
        self.assert_frozen()
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        encoded = [encode_qwen(self.tokenizer, q, c, normalize=False) for q, c in zip(questions, completions)]
        if any(len(e["input_ids"]) > self.max_length for e in encoded):
            raise ValueError("Native PRM context overflow; no silent truncation or reward substitution")
        results, padded = [], 0
        for offset in range(0, len(encoded), self.microbatch_size):
            batch = encoded[offset:offset + self.microbatch_size]
            inputs = self.tokenizer.pad([{"input_ids": e["input_ids"]} for e in batch],
                padding=True, return_tensors="pt").to(self.device)
            output = self.model(**inputs, use_cache=False)
            logits = output.logits if hasattr(output, "logits") else output[0]
            assert logits.shape[:2] == inputs.input_ids.shape and logits.shape[-1] == 2
            assert torch.isfinite(logits).all()
            padded += inputs.input_ids.numel()
            for i, e in enumerate(batch):
                values = logits[i, e["positions"]].float().softmax(-1)[:, 1].cpu().tolist()
                assert len(values) == len(e["steps"]) and all(0 <= p <= 1 for p in values)
                text = completions[offset + i]
                results.append({"step_scores": values, "min_score": min(values), "step_count": len(values),
                    "prm_input_tokens": len(e["input_ids"]), "no_internal_separator": len(values) == 1,
                    "raw_no_double_newline": "\n\n" not in text.strip()})
        torch.cuda.synchronize(self.device)
        self.assert_frozen()
        return results, {"prm_forward_seconds": time.perf_counter() - start,
            "prm_input_tokens": sum(len(e["input_ids"]) for e in encoded),
            "prm_padded_tokens": padded, "prm_scored_completions": len(encoded)}
