"""Opt-in diagnostic-only B trainer subclass; never constructs or starts a trainer.

The original method is checked against the exact archived B source. Only its
cross-device assert_close expression is replaced. Reward generation, actual
advantages, accounting and all remaining assertions remain the original AST.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import time

from v2core import HERE, audit_advantages


def patched_b_trainer_class(legacy_class, *, audit_directory):
    original_source = (HERE / "legacy_snapshot/training_b/train_b.py").read_text()
    tree = ast.parse(original_source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BTrainer")
    original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions")
    actual_source = textwrap.dedent(inspect.getsource(legacy_class._generate_and_score_completions))
    actual_ast = ast.parse(actual_source).body[0]
    if ast.dump(original, include_attributes=False) != ast.dump(actual_ast, include_attributes=False):
        raise ValueError("B trainer method differs from archived version; review before applying patch")
    patched = ast.parse(ast.unparse(original)).body[0]
    target = [i for i, n in enumerate(patched.body) if isinstance(n, ast.Expr)
              and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "torch.testing.assert_close"]
    if len(target) != 1:
        raise ValueError("Expected exactly one original numeric comparison")
    replacement = ast.parse('''_v2_audit_advantages(
        torch.as_tensor(self.last_reward_values, dtype=output["advantages"].dtype,
                        device=output["advantages"].device), output["advantages"],
        receipt_path=_v2_audit_directory / (str(self.state.global_step) + "_" + str(_v2_time_ns()) + ".json"),
        group_size=self.num_generations, problem_ids=[r["problem_id"] for r in rows])''').body[0]
    replaced_index = target[0]
    patched.body[replaced_index] = replacement
    # Prove no other method statement was rewritten.
    for i, (a, b) in enumerate(zip(original.body, patched.body)):
        if i != replaced_index and ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False):
            raise AssertionError("Unexpected change outside numerical audit")
    namespace = dict(legacy_class._generate_and_score_completions.__globals__)
    namespace.update(_v2_audit_advantages=audit_advantages, _v2_audit_directory=Path(audit_directory), _v2_time_ns=time.time_ns)
    module = ast.fix_missing_locations(ast.Module(body=[patched], type_ignores=[]))
    exec(compile(module, str(HERE / "b_audit_patch.py") + "::verified_B_method", "exec"), namespace)
    method = namespace[patched.name]
    result = type("NumericallyAudited" + legacy_class.__name__, (legacy_class,), {patched.name: method})
    result.numeric_patch_receipt = {"original_method_ast_sha256": hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
                                    "replaced_statement_index": replaced_index,
                                    "changes_reward_or_loss": False,
                                    "remaining_method_statements_identical": True,
                                    "starts_training": False}
    return result
