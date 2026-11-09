"""Independent per-instance tree sampling/token-credit integration."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import torch
from tree_core import make_rows
from sampler import TreeSamplerFacade

VALIDATED=Path(__file__).resolve().parents[2]/'experiments/prefix_protection_20261107'


def fingerprint(inputs,i):
    h=hashlib.sha256()
    for name in ('completion_ids','completion_mask','advantages','outcome_component','process_component'):
        v=inputs[name][i].detach().cpu().contiguous()
        h.update(name.encode());h.update(str((v.dtype,tuple(v.shape))).encode())
        h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class TreeSignalMixin:
    def initialize_tree_signals(self,arm,ledger,tokenizer,path):
        self.experiment_arm=arm;self.signal_ledger=ledger;self.signal_tokenizer=tokenizer
        self.signal_audit_path=Path(path);self.signal_serial=0;self.signal_expected={}
        self.tree_sampler=TreeSamplerFacade(self.llm,self,tokenizer)
        self.llm=self.tree_sampler

    def install_signal(self,output,records,trace,arm):
        made=make_rows(records,trace,arm)
        shape=output['completion_mask'].shape
        tensors={name:torch.zeros(shape,dtype=torch.float32,device=output['advantages'].device)
                 for name in ('advantages','outcome_component','process_component')}
        notes=[]
        for i,(r,m,metadata) in enumerate(zip(records,made,trace['rows'])):
            mask=output['completion_mask'][i].bool()
            assert output['completion_ids'][i][mask].detach().cpu().tolist()==r['completion_token_ids']
            n=len(r['completion_token_ids']);assert int(mask.sum())==n
            for name,t in tensors.items():t[i,:n]=torch.tensor(m[name],dtype=torch.float32,device=t.device)
            notes.append(m['coefficients'])
            if not r['outcome_reward']:
                assert not tensors['process_component'][i].any()
                cut=m['coefficients']['prefix_length']
                assert not (tensors['advantages'][i,cut:n]>0).any()
                if arm=='chain_cb':assert not (tensors['advantages'][i]>0).any()
        for name,t in tensors.items():
            assert not t.requires_grad and torch.isfinite(t).all() and not t[output['completion_mask']==0].any()
            output[name]=t
        if arm=='chain_cb':
            # Preserve the historical FP32 torch implementation exactly, rather
            # than changing the control through Python/FP64 intermediate sums.
            path=VALIDATED.parents[1]/'experiments/abc_seed17_resumed_v1_20261104/advantages.py'
            spec=importlib.util.spec_from_file_location('frozen_chain_bounded_advantages',path)
            old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
            result=old.compute_advantages([r['outcome_reward'] for r in records],
                    [r['min_score'] for r in records],valid=[r['valid_for_process_reward'] for r in records])
            mask=output['completion_mask']
            for dest,src in [('advantages','advantages'),('outcome_component','outcome_component'),('process_component','process_component')]:
                output[dest]=result[src].to(mask.device)[:,None]*mask
        torch.testing.assert_close(output['advantages'],output['outcome_component']+output['process_component'],atol=2e-7,rtol=1e-6)
        assert output['process_component'].abs().max()<=.2+1e-6
        tags=torch.arange(self.signal_serial,self.signal_serial+len(records),device=output['advantages'].device)
        self.signal_serial+=len(records);output['signal_trace_id']=tags
        self.signal_expected={int(tag):{'fingerprint':fingerprint(output,i),'outcome':int(records[i]['outcome_reward']),
                 'prefix_length':notes[i]['prefix_length'],'arm':arm} for i,tag in enumerate(tags.tolist())}
        wrong=torch.tensor([not r['outcome_reward'] for r in records],device=output['advantages'].device)
        self.last_signal_checks={
            'wrong_positive_total_tokens':int((output['advantages'][wrong]>0).sum()),
            'wrong_positive_prm_tokens':int((output['process_component'][wrong]>0).sum()),
            'wrong_positive_exclusive_suffix_tokens':0,'post_construction_normalizations':0,
            'advantage_RMS':float(output['advantages'][output['completion_mask'].bool()].square().mean().sqrt()),
            'fallback_roots':sum(bool(p['fallback']) for p in trace['roots']),
            'coefficients':notes,'max_path_tokens':max(len(r['completion_token_ids']) for r in records)}
        return output

    def _generate_and_score_completions(self,inputs):
        output=super()._generate_and_score_completions(inputs)
        trace=self.tree_sampler.last_trace
        assert trace and trace['step']==self.state.global_step
        output=self.install_signal(output,self.signal_ledger.records,trace,self.experiment_arm)
        with self.signal_audit_path.open('a') as f:
            f.write(json.dumps({'arm':self.experiment_arm,'step':self.state.global_step,
                               'checks':self.last_signal_checks,'shape':list(output['advantages'].shape)})+'\n')
        with (self.run/'tree_traces.jsonl').open('a') as f:f.write(json.dumps(trace)+'\n')
        return output

    def _compute_loss(self,model,inputs):
        for i,tag in enumerate(inputs['signal_trace_id'].tolist()):
            expected=self.signal_expected[tag]
            assert fingerprint(inputs,i)==expected['fingerprint'],'Tokens/components/advantage changed before loss'
            if not expected['outcome']:
                assert not inputs['process_component'][i].any()
                cut=expected['prefix_length']
                assert not (inputs['advantages'][i,cut:]>0).any()
        return super()._compute_loss(model,inputs)


def trainer_class():
    from native_sync import NativeLoRASyncMixin
    from token_loss import TokenAdvantageLossMixin
    name='trl.trainer.tree_credit_native_grpo_trainer'
    spec=importlib.util.spec_from_file_location(name,VALIDATED/'native_grpo_trainer.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    class TreeTrainer(TreeSignalMixin,TokenAdvantageLossMixin,NativeLoRASyncMixin,module.GRPOTrainer):
        pass
    return TreeTrainer
