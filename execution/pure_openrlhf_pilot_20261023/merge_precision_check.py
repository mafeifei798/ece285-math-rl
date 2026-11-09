"""Read-only CPU diagnostic for two tensors in an immutable saved checkpoint."""
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
root = Path('/path/to/ece285_math_rl_project')
checkpoint = root / 'runs/PURE_OpenRLHF_PVR_pilot_seed17_20261023/checkpoints/step_0001'
meta = json.loads((checkpoint / 'checkpoint.json').read_text())
assert meta['batch_step'] == meta['optimizer_steps'] == 1
adapter = checkpoint / 'adapter/adapter_model.safetensors'
assert sha(adapter) == meta['files']['adapter/adapter_model.safetensors']
config = json.loads((checkpoint / 'adapter/adapter_config.json').read_text())
assert not config['use_dora'] and not config['use_rslora'] and not config['fan_in_fan_out']
assert not config['alpha_pattern'] and not config['rank_pattern'] and config['bias'] == 'none'
scale = config['lora_alpha'] / config['r']
base = Path(config['base_model_name_or_path'])
index_path = base / 'model.safetensors.index.json'
index = json.loads(index_path.read_text()) if index_path.exists() else None
result = dict(mode='CPU-only, two saved first-layer tensors; no model instantiation or GPU access',
    checkpoint=str(checkpoint), checkpoint_receipt_sha256=sha(checkpoint / 'checkpoint.json'),
    adapter_file_sha256=sha(adapter), batch_step=1, optimizer_steps=1,
    arm='P_VR', torch_threads=torch.get_num_threads(), scale=scale,
    formula='BF16(FP32(base) + FP32(B) @ FP32(A) * scale)', layers=[],
    base_index_sha256=sha(index_path) if index_path.exists() else None,
    base_config_sha256=sha(base / 'config.json'),
    parameters_modified=False, training_interrupted=False, started_unix=time.time())
with safe_open(adapter, framework='pt', device='cpu') as weights:
    for stem in ['model.layers.0.self_attn.q_proj', 'model.layers.0.mlp.gate_proj']:
        key = stem + '.weight'
        filename = index['weight_map'][key] if index else 'model.safetensors'
        with safe_open(base / filename, framework='pt', device='cpu') as original:
            w = original.get_tensor(key)
        a = weights.get_tensor('base_model.model.' + stem + '.lora_A.weight').float()
        b = weights.get_tensor('base_model.model.' + stem + '.lora_B.weight').float()
        assert w.dtype == torch.bfloat16
        delta = (b @ a) * scale
        assert delta.shape == w.shape and torch.isfinite(delta).all()
        merged = (w.float() + delta).to(w.dtype)
        actual = merged.float() - w.float()
        changed = merged != w
        nonzero = delta != 0
        n = w.numel()
        detail = dict(name=stem, shape=list(w.shape), base_dtype=str(w.dtype),
            base_tensor_sha256=tensor_sha(w), lora_A_shape=list(a.shape), lora_B_shape=list(b.shape),
            base_mean_abs=float(w.float().abs().mean()), delta_mean_abs=float(delta.abs().mean()),
            delta_max_abs=float(delta.abs().max()), delta_rms=float(delta.square().mean().sqrt()),
            entries=n, nonzero_delta_entries=int(nonzero.sum()), changed_bf16_entries=int(changed.sum()),
            changed_bf16_fraction=float(changed.sum()) / n,
            changed_fraction_among_nonzero_delta=float((changed & nonzero).sum()) / int(nonzero.sum()),
            merged_change_mean_abs=float(actual.abs().mean()), merged_change_rms=float(actual.square().mean().sqrt()),
            lora_A_mean_abs=float(a.abs().mean()), lora_B_mean_abs=float(b.abs().mean()))
        result['layers'].append(detail)
        del w, a, b, delta, merged, actual, changed, nonzero
assert not torch.cuda.is_initialized()
result.update(cuda_initialized=False, finished_unix=time.time(),
    limitations=['Only step1 P-VR first-layer q_proj and gate_proj; not all layers, later steps or P-PRM.',
                 'A zero BF16 weight difference does not prove unchanged model outputs or no learning in the unmerged actor.',
                 'Changed-weight fraction is a numerical diagnostic, not a reward-quality or accuracy metric.',
                 'No learning-rate, reward, loss or runtime change is made based on this diagnostic.'])
print(json.dumps(result, indent=2, allow_nan=False))
