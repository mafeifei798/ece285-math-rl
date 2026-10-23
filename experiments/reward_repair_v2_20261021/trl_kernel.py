"""Extract only reward tensor algebra from the installed, pinned TRL source."""
import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from v2core import atomic_json, sha256


@lru_cache(maxsize=4)
def extract(path):
    source = Path(path).read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GRPOTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions")
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "rewards" for t in n.targets)
                 and "nansum" in ast.unparse(n))
    end = next(i for i in range(start, len(method.body)) if isinstance(method.body[i], ast.If)
               and ast.unparse(method.body[i].test) in {"self.scale_rewards != 'none'", 'self.scale_rewards != "none"'})
    nodes = method.body[start:end + 1]
    text = "\n".join(ast.unparse(n) for n in nodes)
    assert "self.model" not in text and "_calculate_rewards" not in text and "backward" not in text
    assert "1e-04" in text or "0.0001" in text
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(module, str(path) + "::reward_tensor_kernel", "exec"), text


def actual_trl_advantages(values, source_path, device="cpu"):
    import torch
    code, _ = extract(source_path)
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1, 1)
    environment = {"torch": torch, "device": tensor.device, "rewards_per_func": tensor,
                   "self": SimpleNamespace(reward_weights=torch.ones(1, device=tensor.device),
                                           num_generations=8, scale_rewards="group")}
    exec(code, environment)
    return environment["rewards"], environment["advantages"]


def save_provenance(source_path, output):
    _, extracted = extract(source_path)
    atomic_json(output, {"source_path": str(source_path), "source_sha256": sha256(source_path),
                         "extracted_tensor_algebra": extracted,
                         "executes_generation_or_training": False})
