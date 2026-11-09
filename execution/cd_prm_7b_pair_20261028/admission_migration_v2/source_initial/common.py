"""Isolated, bounded 7B PRM adaptation. No policy optimizer exists here."""
from pathlib import Path
import hashlib
import json
import os
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'runs/CD_PRM7B_pair_seed17_20261028'
CFG = json.loads((HERE / 'config.json').read_text())
PRM = ROOT / 'models/qwen25_math_prm_7b'
sys.path.insert(0, str(ROOT / 'experiments/qwen7b_offline_20261021'))
from core import encode_qwen, cpu_advantages, auc
sys.path.insert(0, str(ROOT / 'experiments/reward_repair_v3_20261021'))
from repairs import repaired_outcome, valid_for_prm
from v2core import legacy_module
question_key = legacy_module('common').question_key


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(s) for s in Path(p).open() if s.strip()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def atomic(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def write_rows(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w') as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def rank(tag, ident):
    return hashlib.sha256(f'285:CD7B:{tag}:{ident}'.encode()).hexdigest()


def event(stage, **kw):
    d = dict(stage=stage, pid=os.getpid(), unix=time.time(), **kw)
    atomic(OUT / 'progress.json', d)
    print(json.dumps(d, ensure_ascii=False), flush=True)


def assert_root():
    assert ROOT.resolve().is_relative_to(Path('/path/to/project_parent')), ROOT
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    assert CFG['actor_optimizer_updates'] == 0 and not CFG['automatic_RL_launch']


def verify_release():
    d = read(HERE / 'release.json')
    for n, checksum in d['files'].items():
        assert sha(HERE / n) == checksum, n
    return sha(HERE / 'release.json')


def protect(create=False):
    dest = OUT / 'protected_originals.json'
    if create and not dest.exists():
        paths = []
        for folder in ['training', 'training_b', 'experiments/ab_corrected_6m_seed17_20261022',
                       'experiments/reward_repair_v2_20261021', 'experiments/reward_repair_v3_20261021',
                       'experiments/b_qwen7b_6m_20261021', 'experiments/qwen7b_offline_20261021']:
            paths += [p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.json'}]
        for arm in ['A', 'B']:
            run = ROOT / f'runs/{arm}_corrected_seed17_6M_20261022'
            paths += [p for p in run.rglob('*') if p.is_file() and (p.name in {'evaluation_summary.json', 'adapter_model.safetensors', 'run_config.json', 'rollouts.jsonl'} or (p.parent.name == 'evaluation' and p.suffix == '.jsonl'))]
        paths += [ROOT / 'data/prepared/manifest.json', ROOT / 'receipts/assets.json', ROOT / 'audits/reward_repair_v3_20261021/unit_metadata.json']
        paths = sorted(set(paths))
        atomic(dest, {str(p.relative_to(ROOT)): sha(p) for p in paths})
    d = read(dest)
    changed = [n for n, digest in d.items() if not (ROOT / n).exists() or sha(ROOT / n) != digest]
    assert not changed, changed
    return {'files_checked': len(d), 'unchanged': True, 'manifest_sha256': sha(dest)}


def encode_steps(tok, question, steps):
    assert steps and all(s.strip() for s in steps)
    assert '<extra_0>' not in question and all('<extra_0>' not in s for s in steps)
    marker = tok.encode('<extra_0>', add_special_tokens=False)
    assert len(marker) == 1
    text = tok.apply_chat_template([
        {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': '<extra_0>'.join(steps) + '<extra_0>'}], tokenize=False, add_generation_prompt=False)
    ids = tok.encode(text, add_special_tokens=False)
    pos = [i for i, x in enumerate(ids) if x == marker[0]]
    assert len(pos) == len(steps)
    return dict(input_ids=ids, positions=pos)


def setup_torch():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def tensor_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode()); h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_prm(adapter=None, training=False):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    random.seed(CFG['seed']); torch.manual_seed(CFG['seed']); torch.cuda.manual_seed_all(CFG['seed'])
    tok = AutoTokenizer.from_pretrained(PRM, local_files_only=True, padding_side='right')
    model, info = AutoModel.from_pretrained(PRM, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
    assert not info['missing_keys'] and not info['mismatched_keys']
    assert set(info['unexpected_keys']) == {'lm_head.weight'}
    model.config.use_cache = False
    if training:
        model = get_peft_model(model, LoraConfig(r=CFG['lora_rank'], lora_alpha=CFG['lora_alpha'],
            lora_dropout=0., target_modules=CFG['targets'], bias='none'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout): m.p = 0.
        assert all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
        assert all(not p.requires_grad for n, p in model.named_parameters() if '.score.' in n)
    else:
        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model.requires_grad_(False).eval()
    return tok, model.to('cuda:0'), info


def load_data():
    assert sha(OUT / 'batch_manifest.json') == read(OUT / 'data_ready.json')['batch_manifest_sha256']
    return read(OUT / 'batch_manifest.json'), read(OUT / 'encoded_training.json')


def pooled_error(p):
    import torch
    error = 1 - p.float()
    weights = torch.softmax(error / CFG['pool_temperature'], 0).detach()
    return (weights * error).sum().clamp(1e-7, 1 - 1e-7)
