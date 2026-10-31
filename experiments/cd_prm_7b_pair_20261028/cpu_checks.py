"""Meaningful mathematical checks before any model update."""
from common import *


def main():
    import torch
    from transformers import AutoTokenizer
    z=torch.tensor([-1.7,.2,2.3],dtype=torch.float64,requires_grad=True)
    p=z.sigmoid();E=pooled_error(p)
    c=-E.log()-torch.log1p(-E);d=-E.log();s=-torch.log1p(-E)
    gc=torch.autograd.grad(c,z,retain_graph=True)[0]
    gd=torch.autograd.grad(d,z,retain_graph=True)[0]
    gs=torch.autograd.grad(s,z)[0]
    assert torch.allclose(gc-gd,gs,atol=2e-7,rtol=1e-6)
    # A common score offset cannot be interpreted as a repaired relative signal.
    a=cpu_advantages([0.2*x for x in [.9,.8,.7,.6,.5,.4,.3,.2]])
    b=cpu_advantages([0.2*x for x in [.8,.7,.6,.5,.4,.3,.2,.1]])
    assert max(abs(x-y) for x,y in zip(a,b))<1e-5
    assert max(a)>0
    tok=AutoTokenizer.from_pretrained(PRM,local_files_only=True)
    q='Solve 2x+3=7.';text='Subtract 3.\r\n\r\nThen x=2.\r\n\r\n\\boxed{2}'
    e=encode_qwen(tok,q,text,normalize=True)
    assert len(e['positions'])==3 and e['crlf_replacements']==4
    assert len(encode_steps(tok,q,['One annotated step.\n\nStill the same step.'])['positions'])==1
    checks=dict(passed=True,success_term_gradient_identity=True,score_translation_does_not_change_relative_advantage=True,
        qwen_native_markers=True,CRLF_normalization=True,anchor_original_boundaries_preserved=True,policy_updates=0)
    atomic(OUT/'cpu_checks.json',checks);print(json.dumps(checks))


if __name__=='__main__':assert_root();verify_release();OUT.mkdir(parents=True,exist_ok=True);main()
