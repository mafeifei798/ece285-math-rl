from common import *
import ast,inspect
import trl
import torch

def loss_ast(path):
    tree=ast.parse(Path(path).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='_compute_loss')
    return ast.dump(node,include_attributes=False)

def main():
    verify();source_unchanged();protected_unchanged()
    schedule=read(ROOT/'audits/ABC_seed17_resumed_V1_20261104/preflight.json')['first_epoch_question_groups']
    assert len(schedule)>=576 and all(len(x)==4 for x in schedule[:576])
    ids=[q for g in schedule[:576] for q in g];assert len(set(ids))==2304
    original=inspect.getsourcefile(trl.GRPOTrainer)
    derivative=ROOT/'experiments/prefix_protection_20261107/native_grpo_trainer.py'
    assert loss_ast(original)==loss_ast(derivative),'Native sampler derivative changed scalar loss'
    settings=None
    for arm in ARMS:
        cp=SOURCE/arm/'checkpoint-288';contract=read(cp/'pilot_checkpoint.json')
        assert contract['global_step']==288 and contract['complete_generation_boundary']
        a=read(cp/'completion_accounting.json');assert a['loss_rows']==a['generated_completions']==9216
        args=torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False).to_dict()
        comparable={k:v for k,v in args.items() if k not in ('output_dir','logging_dir','run_name')}
        if settings is None:settings=comparable
        else:assert settings==comparable,'Original A/C training settings differ'
    atomic(AUDIT/'preflight.json',{'passed':True,'first_epoch_question_groups':schedule,
        'scalar_loss_AST_equal':True,'same_first576_schedule':True,'unique_questions_through576':2304,
        'reward_loss_unchanged':True,'native_sync_runtime_disclosed':True})
    print('CPU preflight passed; scalar loss AST and source checkpoints verified.')

if __name__=='__main__':main()
