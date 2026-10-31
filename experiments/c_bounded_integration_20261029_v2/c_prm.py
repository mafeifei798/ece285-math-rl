"""Frozen original Qwen7B plus the already trained C adapter; no optimization."""
from pathlib import Path
import torch
import sys
from audit_common import ROOT,CD,C_ADAPTER,read,sha,atomic
sys.path.insert(0,str(ROOT/'experiments/b_qwen7b_6m_20261021'))
from frozen_prm import FrozenQwenPRM

class FrozenCScorer(FrozenQwenPRM):
    def __init__(self,*,receipt_dir):
        from peft import PeftModel
        receipt=read(CD/'C/completion.json')
        assert receipt['complete'] and receipt['steps_completed']==512
        assert receipt['adapter_sha256']==sha(C_ADAPTER/'adapter_model.safetensors')
        assert receipt['adapter_sha256']=='4917be38966c8438f58e283eb06c7fa818fed2bb13b726152078c19af013c356'
        super().__init__(ROOT/'models/qwen25_math_prm_7b',receipt_dir=receipt_dir,microbatch_size=1)
        self.model=PeftModel.from_pretrained(self.model,C_ADAPTER,is_trainable=False).requires_grad_(False).eval()
        self.versions={n:p._version for n,p in self.model.named_parameters()}
        self.assert_frozen()
        atomic(Path(receipt_dir)/'C_loading.json',dict(adapter_sha256=receipt['adapter_sha256'],C_steps=512,
            frozen=True,no_new_PRM_adaptation=True,canonical_CRLF_adapter_used=True,native_marker_id=self.tokenizer.encode('<extra_0>',add_special_tokens=False)))

    @torch.inference_mode()
    def score(self,questions,completions):
        # vLLM may change global FP32 matmul precision. Match the frozen C audit
        # without changing actor/vLLM precision outside this scoring context.
        cuda_flag=torch.backends.cuda.matmul.allow_tf32
        cudnn_flag=torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            result=super().score(questions,completions)
        finally:
            torch.backends.cuda.matmul.allow_tf32=cuda_flag
            torch.backends.cudnn.allow_tf32=cudnn_flag
        assert torch.backends.cuda.matmul.allow_tf32==cuda_flag
        assert torch.backends.cudnn.allow_tf32==cudnn_flag
        return result
