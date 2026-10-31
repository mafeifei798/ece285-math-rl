"""Opt-in isolated GRPO subclass. Formal training is blocked in this release."""
import inspect,hashlib,importlib.metadata
from pathlib import Path
from collections import deque
import torch
from advantages import compute_advantages
from audit_common import row_fingerprint

PINNED_TRL_SHA256='f217183c84be9b38b23454920eb1b914726aa3c4b9dd4c076d455bdaa67bf1b8'
PINNED_UTILS_SHA256='07cccc9023d60528544390d70efc0a505d2dc1e92473133c045d33f7ed976360'

class CBoundedMixin:
    def initialize_c_bounded(self,ledger):
        self.c_ledger=ledger;self.c_serial=0;self.c_expected={}
        self.c_generation_checks=[];self.c_loss_trace=[];self.c_loss_calls=0
        self.c_last_output=None

    def attach_c_signal(self,output,rows,*,phase):
        old=output['advantages'];n=old.numel()
        if n!=len(rows) or n%8:raise RuntimeError('Ledger/generation rows mismatch')
        for start in range(0,n,8):
            if len({r['problem_id'] for r in rows[start:start+8]})!=1:raise RuntimeError('Question grouping changed')
        y=torch.tensor([r['outcome_reward'] for r in rows],device=old.device,dtype=torch.float32)
        result=compute_advantages(y,[r['min_score'] for r in rows],valid=[r['valid_for_process_reward'] for r in rows])
        actual=result['advantages'];output['advantages']=actual
        tags=torch.arange(self.c_serial,self.c_serial+n,device=old.device,dtype=torch.int64);self.c_serial+=n
        output['c_bounded_trace_id']=tags
        for i,tag in enumerate(tags.cpu().tolist()):
            self.c_expected[tag]=dict(advantage=float(actual[i]),outcome=int(y[i]),process_min=float(rows[i]['min_score']),
                outcome_component=float(result['outcome_component'][i]),process_component=float(result['process_component'][i]),
                problem_id=rows[i]['problem_id'],phase=phase,row_in_generation=i,fingerprint=row_fingerprint(output,i))
        log=self._logs['advantages'];previous=list(log);log.clear();log.extend(previous[:-n]+actual.cpu().tolist())
        self.c_generation_checks.append(dict(phase=phase,rows=n,replaced_before_shuffle=True,
            outcome_normalizations=1,post_combination_normalizations=0,
            wrong_positive=int(((actual>0)&(y==0)).sum()),max_abs_process=float(result['process_component'].abs().max())))
        self.c_last_output=output
        return output

    def _generate_and_score_completions(self,inputs):
        if self.accelerator.num_processes!=1 or self.num_generations!=8:raise RuntimeError('Single-process group8 only')
        output=super()._generate_and_score_completions(inputs)
        rows=self.c_ledger.records
        mixed=torch.tensor([r['reward'] for r in rows],device=output['advantages'].device,dtype=torch.float32).reshape(-1,8)
        old=((mixed-mixed.mean(1,keepdim=True))/(mixed.std(1,keepdim=True)+1e-4)).flatten()
        torch.testing.assert_close(output['advantages'],old,atol=2e-5,rtol=2e-5)
        return self.attach_c_signal(output,rows,phase=getattr(self,'_c_phase','live_generation'))

    def assert_c_loss_input(self,inputs):
        tags=inputs.get('c_bounded_trace_id')
        if tags is None or tags.dtype!=torch.int64:raise RuntimeError('C provenance tag missing')
        actual=inputs['advantages']
        try:expected=[self.c_expected[i] for i in tags.detach().cpu().tolist()]
        except KeyError as e:raise RuntimeError('Stale/misaligned C buffer') from e
        values=torch.tensor([x['advantage'] for x in expected],device=actual.device,dtype=torch.float32)
        if actual.dtype!=torch.float32 or actual.requires_grad or not torch.equal(actual,values):
            raise RuntimeError('C advantage changed: possible second normalization')
        for i,record in enumerate(expected):
            if row_fingerprint(inputs,i)!=record['fingerprint']:raise RuntimeError('C response/token tag alignment changed')
            if record['outcome']==0 and actual[i]>0:raise RuntimeError('Wrong answer received positive advantage')
        return actual.detach().clone(),expected

    def _compute_loss(self,model,inputs):
        before,records=self.assert_c_loss_input(inputs)
        loss=super()._compute_loss(model,inputs)
        if not torch.equal(before,inputs['advantages']):raise RuntimeError('Loss modified C advantages')
        self.c_loss_calls+=1
        for i,r in enumerate(records):self.c_loss_trace.append(dict(r,tag=int(inputs['c_bounded_trace_id'][i]),
            advantage_at_loss=float(inputs['advantages'][i]),loss=float(loss.detach()),bitwise_preserved=True))
        return loss

def trainer_class():
    from trl import GRPOTrainer
    from trl.trainer.utils import shuffle_sequence_dict
    assert importlib.metadata.version('trl')=='0.23.1'
    assert hashlib.sha256(Path(inspect.getsourcefile(GRPOTrainer)).read_bytes()).hexdigest()==PINNED_TRL_SHA256
    assert hashlib.sha256(Path(inspect.getsourcefile(shuffle_sequence_dict)).read_bytes()).hexdigest()==PINNED_UTILS_SHA256
    class CBoundedTrainer(CBoundedMixin,GRPOTrainer):
        def __init__(self,*args,c_ledger,**kwargs):
            super().__init__(*args,**kwargs);self.initialize_c_bounded(c_ledger)
            assert self.args.loss_type=='grpo' and self.num_generations==8 and self.scale_rewards=='group'
            assert not self.use_liger_loss and self.num_iterations==1 and self.accelerator.num_processes==1
        def train(self,*args,**kwargs):raise RuntimeError('Formal RL is not authorized in this release')
        def create_optimizer(self,*args,**kwargs):raise RuntimeError('No-update audit forbids optimizer construction')
        def create_optimizer_and_scheduler(self,*args,**kwargs):raise RuntimeError('No-update audit forbids optimizer construction')
        def save_model(self,*args,**kwargs):raise RuntimeError('No-update audit does not save model checkpoints')
    return CBoundedTrainer
