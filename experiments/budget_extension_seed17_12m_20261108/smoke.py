from common import *
import argparse,math,traceback
from build import build,VALIDATED
sys.path.insert(0,str(VALIDATED))
from no_update import tensor_hash
import torch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True);arm=parser.parse_args().arm
    guard();verify();source_unchanged();assert read(AUDIT/'preflight.json')['passed']
    out=AUDIT/'no_update'/arm;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'intent.json').exists();atomic(out/'intent.json',{'pid':os.getpid(),'optimizer_updates':0})
    cp=SOURCE/arm/'checkpoint-288';started=time.time();t,tok=build(arm,out,cp,'smoke')
    t.state.global_step=288;t.current_gradient_accumulation_steps=32;t._load_rng_state(str(cp))
    t.model.train();t.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});t.model.enable_input_require_grads()
    base=tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n);adapter=adapter_digest(t.model)
    data={r['problem_id']:r for r in rows(ROOT/'data/prepared/train.jsonl')}
    chosen=read(AUDIT/'preflight.json')['first_epoch_question_groups'][288]
    batch=[dict(data[q]) for q in chosen for _ in range(8)]
    output=t._generate_and_score_completions(batch)
    assert tensor_hash((n,p) for n,p in t.model.named_parameters() if 'lora_' not in n)==base
    candidates=[i for i,a in enumerate(output['advantages']) if abs(float(a))>0]
    assert candidates
    picked=sorted(set([min(candidates,key=lambda i:t.ledger.records[i]['completion_tokens']),max(candidates,key=lambda i:t.ledger.records[i]['completion_tokens'])]))
    probes=[]
    for i in picked:
        one={k:v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0]==32 else v for k,v in output.items()}
        t.used_tags=set();t.model.zero_grad(set_to_none=True)
        with t.compute_loss_context_manager():loss=t._compute_loss(t.model,one)
        loss.backward();grad=math.sqrt(sum(float(p.grad.float().square().sum()) for p in t.model.parameters() if p.requires_grad and p.grad is not None))
        assert math.isfinite(grad) and grad>0 and torch.isfinite(loss)
        probes.append({'row':i,'loss':float(loss.detach()),'gradient_L2':grad})
    t.model.zero_grad(set_to_none=True)
    assert t.optimizer is None and adapter_digest(t.model)==adapter
    if t.prm:t.prm.assert_frozen()
    verify();source_unchanged()
    atomic(out/'result.json',{'passed':True,'arm':arm,'optimizer_constructed':False,'optimizer_updates':0,
        'actor_adapter_unchanged':True,'actor_base_unchanged':True,'source_checkpoint_unchanged':True,
        'rows':32,'backward_probes':probes,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'seconds':time.time()-started})

if __name__=='__main__':main()
