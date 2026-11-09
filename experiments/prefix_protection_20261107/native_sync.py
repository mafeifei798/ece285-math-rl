"""Single-GPU native LoRA synchronization without merging the actor base."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import torch
from peft import PeftModel
from vllm.lora.request import LoRARequest

class NativeLoRASyncMixin:
    def _move_model_to_vllm(self):
        assert isinstance(self.model,PeftModel)
        assert self.vllm_mode=='colocate' and self.accelerator.num_processes==1
        assert not self.is_fsdp_enabled and not self.is_deepspeed_enabled
        before={n:p._version for n,p in self.model.named_parameters() if 'lora_' not in n}
        if hasattr(self,'_frozen_base_versions'):
            assert before==self._frozen_base_versions,'Frozen actor base changed between synchronizations'
        else:self._frozen_base_versions=before
        step=int(self.state.global_step)
        path=Path(self.args.output_dir)/'sampling_adapters'/f'step_{step:06d}'
        assert not path.exists(),'Sampling snapshot already exists; inspect before any retry'
        self.model.save_pretrained(path,safe_serialization=True)
        assert before=={n:p._version for n,p in self.model.named_parameters() if 'lora_' not in n}
        self._active_lora_request=LoRARequest(f'policy_step_{step}',step+1,str(path))
        self.llm.reset_prefix_cache()
        receipt={'step':step,'adapter_id':step+1,'path':str(path),'actor_base_unchanged':True,
                 'sha256':hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()}
        with (Path(self.args.output_dir)/'native_sync.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')

def trainer_class():
    # Load the isolated derivative under the TRL package namespace so its
    # original relative imports retain their pinned semantics. No global patch.
    name='trl.trainer.prefix_native_grpo_trainer'
    path=Path(__file__).with_name('native_grpo_trainer.py')
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod)
    from runtime_mixin import PrefixSignalMixin
    from token_loss import TokenAdvantageLossMixin
    class PrefixTrainer(PrefixSignalMixin,TokenAdvantageLossMixin,NativeLoRASyncMixin,mod.GRPOTrainer):
        pass
    return PrefixTrainer
