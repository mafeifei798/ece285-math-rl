"""Opt-in trainer integration. Instantiate explicitly; never monkey-patch TRL."""
import hashlib
import json
from pathlib import Path
import torch
from signals import signals

def fingerprint(inputs, i):
    h=hashlib.sha256()
    for name in ['completion_ids','completion_mask','advantages']:
        v=inputs[name][i].detach().cpu().contiguous()
        h.update(name.encode());h.update(str((v.dtype,tuple(v.shape))).encode())
        h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

class PrefixSignalMixin:
    def initialize_signals(self, variant, ledger, tokenizer, audit_path):
        if variant not in ['bounded','uniform','prefix']: raise ValueError(variant)
        self.signal_variant=variant;self.signal_ledger=ledger;self.signal_tokenizer=tokenizer
        self.signal_audit_path=Path(audit_path);self.signal_expected={};self.signal_serial=0

    def _generate_and_score_completions(self, inputs):
        output=super()._generate_and_score_completions(inputs)
        records=self.signal_ledger.records
        for i,r in enumerate(records):
            mask=output['completion_mask'][i].bool()
            ids=output['completion_ids'][i][mask].detach().cpu().tolist()
            assert ids==r['completion_token_ids'], 'Raw token IDs changed before signal construction'
        made=signals(records,self.signal_tokenizer,device=output['advantages'].device)
        assert torch.equal(made['completion_mask'],output['completion_mask'].bool())
        actual=made[self.signal_variant]
        output['advantages']=actual
        tags=torch.arange(self.signal_serial,self.signal_serial+len(records),device=actual.device)
        self.signal_serial+=len(records);output['signal_trace_id']=tags
        self.signal_expected={int(t):{'fingerprint':fingerprint(output,i),'outcome':int(records[i]['outcome_reward'])} for i,t in enumerate(tags.tolist())}
        with self.signal_audit_path.open('a') as f:
            f.write(json.dumps({'variant':self.signal_variant,'trace_ids':tags.tolist(),
                'annotations':made['annotations'],'normalizations_after_scalar':0,
                'shape':list(actual.shape),'wrong_positive_tokens':int((actual[torch.tensor([not r['outcome_reward'] for r in records],device=actual.device)]>0).sum())})+'\n')
        return output

    def _compute_loss(self, model, inputs):
        tags=inputs['signal_trace_id'].tolist()
        for i,tag in enumerate(tags):
            expected=self.signal_expected[tag]
            assert fingerprint(inputs,i)==expected['fingerprint'], 'Token/advantage changed or normalized after construction'
            if expected['outcome']==0: assert not (inputs['advantages'][i]>0).any()
        return super()._compute_loss(model,inputs)
