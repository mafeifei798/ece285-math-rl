"""All-response native Qwen scoring; old B scorer and gating stay unchanged."""
import sys
import time
import torch
from common import ROOT, atomic
from alignment import native_input

sys.path.insert(0, str(ROOT / 'experiments/b_qwen7b_6m_20261021'))
from frozen_prm import FrozenQwenPRM

class PureQwenPRM(FrozenQwenPRM):
    def __init__(self, path, *, receipt_dir):
        super().__init__(path, receipt_dir=receipt_dir, microbatch_size=2)
        atomic(self.receipt_dir / 'pure_prm_interface.json', dict(
            native_checkpoint=str(path), scoring_all_responses=True, normalize_crlf=True,
            empty_terminal_step=True, max_length=self.max_length, microbatch=2,
            no_outcome_gate=True, overflow='raise before update'))

    @torch.inference_mode()
    def score(self, questions, completions):
        assert len(questions) == len(completions)
        self.assert_frozen()
        torch.cuda.synchronize(); start = time.perf_counter()
        encoded = [native_input(self.tokenizer, q, c, self.max_length) for q, c in zip(questions, completions)]
        results, padded = [], 0
        for offset in range(0, len(encoded), self.microbatch_size):
            sub = encoded[offset:offset + self.microbatch_size]
            inp = self.tokenizer.pad([dict(input_ids=e['input_ids']) for e in sub], padding=True, return_tensors='pt').to(self.device)
            output = self.model(**inp, use_cache=False)
            logits = output.logits if hasattr(output, 'logits') else output[0]
            assert logits.shape[:2] == inp.input_ids.shape and logits.shape[-1] == 2
            assert torch.isfinite(logits).all()
            padded += inp.input_ids.numel()
            for i, e in enumerate(sub):
                ps = logits[i, e['positions']].float().softmax(-1)[:, 1].cpu().tolist()
                results.append({**e, 'step_scores': ps, 'prm_input_tokens': len(e['input_ids'])})
            del output, logits, inp
        torch.cuda.synchronize(); self.assert_frozen()
        return results, dict(seconds=time.perf_counter() - start,
            scored_responses=len(results), prm_input_tokens=sum(len(e['input_ids']) for e in encoded),
            prm_padded_tokens=padded)
