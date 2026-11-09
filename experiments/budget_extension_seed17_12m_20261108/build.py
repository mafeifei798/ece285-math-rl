"""Original scalar signals/loss, with previously validated native LoRA sync."""
from common import *
import dataclasses,random,importlib.util
import torch,numpy as np
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('budget_frozen_runtime',OLD/'runtime.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
# Rebind only this isolated imported module's receipts/schedule, never old files.
legacy.AUDIT=AUDIT;legacy.verify=verify
VALIDATED=ROOT/'experiments/prefix_protection_20261107'
sys.path.insert(0,str(VALIDATED))
from native_sync import NativeLoRASyncMixin
spec=importlib.util.spec_from_file_location('trl.trainer.budget_native_grpo_trainer',VALIDATED/'native_grpo_trainer.py')
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
from c_prm import FrozenCScorer
from c_rewards import RewardLedger
from answer_format import describe

class RecordMixin:
    def _generate_and_score_completions(self,inputs):
        result=super()._generate_and_score_completions(inputs)
        step=self.state.global_step
        path=self.run/f'raw_signals_step-{step}.pt';assert not path.exists()
        torch.save({k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in result.items()},path)
        records=self.ledger.records
        for i,r in enumerate(records):
            mask=result['completion_mask'][i].bool()
            assert result['completion_ids'][i][mask].cpu().tolist()==r['completion_token_ids']
        fmt=[describe({'completion':r['completion'],'reward':r['outcome_reward'],
            'finish_reason':'stop' if r['terminated'] else 'length'}) for r in records]
        counts={k:sum(x[k] for x in fmt) for k,v in fmt[0].items() if isinstance(v,bool)}
        append(self.run/'format_monitoring.jsonl',{'step':step,'n':len(fmt),'counts':counts})
        append(self.run/'actual_signals.jsonl',{'step':step,'outcomes':[r['outcome_reward'] for r in records],
            'minima':[r['min_score'] for r in records],'valid':[r['valid_for_process_reward'] for r in records],
            'advantages':result['advantages'].detach().cpu().tolist(),'post_construction_normalizations':0})
        return result

class Trainer(RecordMixin,legacy.PilotMixin,NativeLoRASyncMixin,native.GRPOTrainer):
    def __init__(self,*a,pilot_arm,pilot_ledger,pilot_run,pilot_accounting,**kw):
        super().__init__(*a,**kw)
        self.initialize_pilot(pilot_arm,pilot_ledger,pilot_run,pilot_accounting)
        assert self.num_iterations==1 and self.loss_type=='grpo' and self.scale_rewards=='group'

def build(arm,run,checkpoint,target):
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from datasets import Dataset
    from peft import PeftModel
    from trl import GRPOConfig
    verify();set_seed(17);torch.set_num_threads(2)
    saved=torch.load(checkpoint/'training_args.bin',map_location='cpu',weights_only=False)
    values={f.name:getattr(saved,f.name) for f in dataclasses.fields(saved) if f.init}
    values.update(output_dir=str(run),logging_dir=str(run/'logs'),report_to=[],generation_batch_size=None,
        vllm_gpu_memory_utilization=.15,save_steps=5,save_total_limit=3)
    args=GRPOConfig(**values)
    assert args.seed==args.data_seed==17 and args.max_steps==20000
    assert args.learning_rate==5e-6 and args.beta==.04 and args.loss_type=='grpo' and args.num_iterations==1
    assert args.num_generations==8 and args.generation_batch_size==32 and args.steps_per_generation==32
    assert args.gradient_accumulation_steps==32 and args.per_device_train_batch_size==1 and args.max_completion_length==2048
    assets=read(ROOT/'receipts/assets.json');cfg=read(ROOT/'training/baseline_config.json')
    tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(assets['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation=cfg['attention_implementation'])
    model.config.use_cache=False;model=PeftModel.from_pretrained(model,checkpoint,is_trainable=True)
    assert adapter_digest(model)==read(checkpoint/'pilot_checkpoint.json')['actor_adapter_digest']
    units=read(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')
    ledger=legacy.OutcomeLedger(units) if arm=='A' else RewardLedger(None,units);owner={}
    t=Trainer(model=model,processing_class=tok,args=args,train_dataset=Dataset.from_list(rows(ROOT/'data/prepared/train.jsonl')),
        reward_funcs=ledger.callback(tok.eos_token_id,owner),pilot_arm=arm,pilot_ledger=ledger,pilot_run=run,
        pilot_accounting=read(checkpoint/'completion_accounting.json'))
    owner['trainer']=t
    if arm!='A':
        p,n=random.getstate(),np.random.get_state()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):scorer=FrozenCScorer(receipt_dir=run)
        random.setstate(p);np.random.set_state(n)
        t.prm=scorer;ledger.scorer.frozen_scorer=scorer
    atomic(run/f'build_{target}.json',{'arm':arm,'settings':args.to_dict(),'source_checkpoint':str(checkpoint),
        'source_release_sha256':verify(),'native_LoRA':True,'scalar_loss_unchanged':True,'PRM_loaded':arm!='A'})
    return t,tok
