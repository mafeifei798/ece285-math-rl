"""Audit-only TRL0.23.1 mixin; no generation, model load or optimizer on import.

No training launcher is provided. Official PAPO parity and CPU loss replay do
not constitute real-model, multi-update or checkpoint-resume acceptance.
"""
from collections import deque
import hashlib,inspect,importlib.metadata
from pathlib import Path
import torch
from signals import construct_signal
from diagnostics import summarize_signal

TRL_SHA='f217183c84be9b38b23454920eb1b914726aa3c4b9dd4c076d455bdaa67bf1b8'
UTILS_SHA='07cccc9023d60528544390d70efc0a505d2dc1e92473133c045d33f7ed976360'

def fingerprint(pack,i):
    h=hashlib.sha256()
    for name in ['prompt_ids','prompt_mask','completion_ids','completion_mask']:
        x=pack[name][i].detach().cpu().contiguous()
        h.update(name.encode());h.update(str(tuple(x.shape)).encode());h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

class SignalAuditMixin:
    def initialize_signal_audit(self,config,ledger):
        self.signal_config=config;self.signal_ledger=ledger;self.signal_serial=0
        self.signal_expected={};self.signal_metrics=[];self.signal_loss_rows=0;self.signal_loss_calls=0;self.signal_last_output=None
    def _generate_and_score_completions(self,inputs):
        if self.accelerator.num_processes!=1:raise RuntimeError('Only single-process buffering is admitted')
        pack=super()._generate_and_score_completions(inputs)
        records=self.signal_ledger.records;config=self.signal_config
        if pack['advantages'].numel()!=len(records):raise RuntimeError('Reward ledger/actor rows differ')
        kwargs=dict(outcomes=[r['outcome_reward'] for r in records],group_ids=[r['problem_id'] for r in records],
            callback_rewards=[r['reward'] for r in records],device=pack['advantages'].device)
        if config.mode!='outcome':kwargs.update(process_scores=[r['min_score'] for r in records],
            process_scored=[r['valid_for_process_reward'] for r in records])
        result=construct_signal(config,**kwargs)
        rw=result['callback_rewards'].view(-1,8)
        expected_native=((rw-rw.mean(1,keepdim=True))/(rw.std(1,keepdim=True)+1e-4)).flatten()
        torch.testing.assert_close(pack['advantages'],expected_native,atol=2e-5,rtol=2e-5)
        if config.mode in ['outcome','mixed']:
            # Preserve the actual TRL tensor for true baseline arms.
            torch.testing.assert_close(pack['advantages'],result['advantages'],atol=2e-5,rtol=2e-5)
            result['advantages']=pack['advantages']
            result['deviation_from_outcome']=pack['advantages']-result['outcome_component']
        else:
            pack['advantages']=result['advantages']
            log=self._logs['advantages'];old=list(log);log.clear();log.extend(old[:-len(records)]+pack['advantages'].cpu().tolist())
        tags=torch.arange(self.signal_serial,self.signal_serial+len(records),device=pack['advantages'].device,dtype=torch.int64)
        self.signal_serial+=len(records);pack['signal_trace_id']=tags
        self.signal_expected={int(tag):dict(advantage=float(pack['advantages'][i]),outcome=r['outcome_reward'],fingerprint=fingerprint(pack,i))
            for i,(tag,r) in enumerate(zip(tags.cpu().tolist(),records))}
        self.signal_metrics.append(summarize_signal(result,records));self.signal_last_output=pack
        return pack
    def assert_signal_input(self,inputs):
        tags=inputs.get('signal_trace_id')
        if tags is None or tags.dtype!=torch.int64:raise RuntimeError('Signal provenance missing')
        try:records=[self.signal_expected[int(t)] for t in tags.cpu().tolist()]
        except KeyError as e:raise RuntimeError('Signal tags do not match buffered responses') from e
        actual=inputs['advantages'];want=torch.tensor([r['advantage'] for r in records],device=actual.device,dtype=torch.float32)
        if actual.dtype!=torch.float32 or actual.requires_grad or not torch.equal(actual,want):
            raise RuntimeError('Advantage changed after construction, possibly normalized twice')
        for i,r in enumerate(records):
            if fingerprint(inputs,i)!=r['fingerprint']:raise RuntimeError('Token/mask and signal tags misaligned')
            if self.signal_config.mode!='mixed' and r['outcome']==0 and actual[i]>0:raise RuntimeError('Wrong positive advantage')
        return actual.clone()
    def _compute_loss(self,model,inputs):
        before=self.assert_signal_input(inputs);loss=super()._compute_loss(model,inputs)
        if not torch.equal(before,inputs['advantages']):raise RuntimeError('Loss mutated advantages')
        self.signal_loss_rows+=inputs['advantages'].numel();self.signal_loss_calls+=1
        return loss

def audit_trainer_class():
    from trl import GRPOTrainer
    from trl.trainer.utils import shuffle_sequence_dict
    if importlib.metadata.version('trl')!='0.23.1':raise RuntimeError('Pinned TRL required')
    if hashlib.sha256(Path(inspect.getsourcefile(GRPOTrainer)).read_bytes()).hexdigest()!=TRL_SHA:raise RuntimeError('TRL source differs')
    if hashlib.sha256(Path(inspect.getsourcefile(shuffle_sequence_dict)).read_bytes()).hexdigest()!=UTILS_SHA:raise RuntimeError('TRL buffer helpers differ')
    class AuditedSignalTrainer(SignalAuditMixin,GRPOTrainer):
        def __init__(self,*args,signal_config,signal_ledger,**kwargs):
            super().__init__(*args,**kwargs);self.initialize_signal_audit(signal_config,signal_ledger)
            if self.args.loss_type!='grpo' or self.num_generations!=8 or self.scale_rewards!='group':raise RuntimeError('Unaudited GRPO settings')
            if self.num_iterations!=1 or self.use_liger_loss or self.accelerator.num_processes!=1:raise RuntimeError('Unaudited execution path')
        def train(self,*args,**kwargs):raise RuntimeError('V3 is CPU-validated only; no new training authorized or GPU acceptance yet')
        def create_optimizer(self,*args,**kwargs):raise RuntimeError('Audit-only version forbids optimizer construction')
        def create_optimizer_and_scheduler(self,*args,**kwargs):raise RuntimeError('Audit-only version forbids optimizer construction')
    return AuditedSignalTrainer
