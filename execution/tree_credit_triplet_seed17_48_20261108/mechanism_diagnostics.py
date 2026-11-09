"""CPU-only attribution/cost diagnostics from frozen rollout records."""
from pathlib import Path
import json,sys
from collections import Counter
import numpy as np

ROOT=Path('/path/to/ece285_math_rl_project')
RUN=ROOT/'runs/Tree_credit_triplet_seed17_48_20261108';AUDIT=ROOT/'audits/Tree_credit_triplet_seed17_48_20261108'
sys.path.insert(0,str(ROOT/'experiments/prefix_protection_20261107'))
from signals import byte_decoder
from transformers import AutoTokenizer


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.open() if s.strip()]


def main():
    import torch
    assert not torch.cuda.is_initialized()
    assert read(RUN/'controller.json')['stage']=='complete'
    tok=AutoTokenizer.from_pretrained(read(ROOT/'receipts/assets.json')['model_path'],local_files_only=True)
    decoder=byte_decoder();result={}
    for arm in ('chain_cb','tree_vr','tree_cb'):
        path=RUN/arm;rr=rows(path/'rollouts.jsonl');traces=rows(path/'tree_traces.jsonl')
        generation_checks=rows(path/'generation_checks.jsonl')
        sampler_seconds=sum(t['seconds'] for t in traces)
        generation_seconds=sum(g['seconds'] for g in generation_checks)
        prm_seconds=sum(g['PRM_cost'].get('prm_forward_seconds',0.) for g in generation_checks)
        groups=Counter();root_causes=Counter();locations=Counter();lengths=[];positive_wrong=Counter()
        by_group_type={name:dict(groups=0,correct_leaves=0,leaves=0,wrong_positive_shared_prefix_tokens=0,
                                  correct_tokens_reversed_from_positive_outcome_to_negative_total=0,
                                  total_policy_advantage_abs_mass=0.,process_advantage_abs_mass=0.,
                                  tree_vs_flat_2center_dot=0.,tree_square_mass=0.,flat_2center_square_mass=0.) for name in ('mixed','all_correct','all_wrong')}
        for batch,trace in enumerate(traces):
            raw=torch.load(path/f'raw_step_{288+batch}.pt',map_location='cpu',weights_only=False)
            records=rr[batch*32:(batch+1)*32]
            for root in trace['roots']:
                cut=root['cut'];lengths.append(cut)
                if root['fallback']:
                    ids=root['initial_ids'];pieces=[tok.decode([t],skip_special_tokens=False) for t in ids[:4]]
                    root_causes['no_eligible_prefix']+=1
                    if len(ids)<=2:root_causes['very_short_root']+=1
                    if any(any(w in s.lower() for w in ('answer','conclusion','boxed')) for s in pieces[:2]):root_causes['answer_marker_within_first_two_tokens']+=1
                else:
                    root_causes['nonempty_fork']+=1
                    if cut<=16:root_causes['nonempty_fork_at_most16_tokens']+=1
                    if cut>=32:root_causes['nonempty_fork_at_least32_tokens']+=1
            for g in range(4):
                sub=records[g*8:g*8+8];n=sum(x['outcome_reward'] for x in sub)
                kind='all_wrong' if n==0 else 'all_correct' if n==8 else 'mixed'
                groups[kind]+=1;stats=by_group_type[kind]
                stats['groups']+=1;stats['correct_leaves']+=n;stats['leaves']+=8
                stats['total_policy_advantage_abs_mass']+=float(raw['advantages'][g*8:g*8+8].abs().sum())
                stats['process_advantage_abs_mass']+=float(raw['process_component'][g*8:g*8+8].abs().sum())
                for j,r in enumerate(sub):
                    i=g*8+j;cut=trace['rows'][i]['prefix_length'];mask=raw['completion_mask'][i].bool()
                    o=raw['outcome_component'][i][mask].double()
                    flat=2*(r['outcome_reward']-n/8)
                    stats['tree_vs_flat_2center_dot']+=float(o.sum())*flat
                    stats['tree_square_mass']+=float(o.square().sum())
                    stats['flat_2center_square_mass']+=float(mask.sum())*flat**2
                    if r['outcome_reward']:
                        stats['correct_tokens_reversed_from_positive_outcome_to_negative_total']+=int(
                            ((raw['outcome_component'][i]>0)&(raw['advantages'][i]<0)&mask).sum())
                    if not r['outcome_reward']:
                        cnt=int((raw['advantages'][i,:cut]>0).sum())
                        positive_wrong['shared_prefix_tokens']+=cnt;stats['wrong_positive_shared_prefix_tokens']+=cnt
                        assert not (raw['advantages'][i,cut:][mask[cut:]]>0).any()
                        assert not raw['process_component'][i].any()
                    if arm!='tree_cb' or not r['outcome_reward'] or not r['step_scores']:
                        continue
                    # The score is for the whole answer; locate its minimum relative to the fork.
                    scores=r['step_scores'];minimum=min(scores)
                    minimum_indices=[k for k,s in enumerate(scores) if s==minimum]
                    spans=r.get('step_raw_spans',[])
                    if len(spans)!=len(scores):locations['span_unavailable']+=1;continue
                    try:
                        ids=r['completion_token_ids']
                        chunks=[bytes(decoder[c] for c in t) for t in tok.convert_ids_to_tokens(ids) if t not in tok.all_special_tokens]
                        if b''.join(chunks)!=r['completion'].encode('utf-8'):
                            locations['byte_alignment_unavailable']+=1;continue
                        prefix_bytes=sum(len(bytes(decoder[c] for c in t)) for t in tok.convert_ids_to_tokens(ids[:cut]))
                        places=set()
                        for k in minimum_indices:
                            start=len(r['completion'][:spans[k]['raw_start']].encode('utf-8'))
                            end=len(r['completion'][:spans[k]['raw_end']].encode('utf-8'))
                            places.add('shared_prefix' if end<=prefix_bytes else 'exclusive_suffix' if start>=prefix_bytes else 'crosses_fork')
                        place=next(iter(places)) if len(places)==1 else 'minimum_tied_across_regions'
                        locations[place]+=1
                    except (KeyError,TypeError,UnicodeError):locations['byte_alignment_unavailable']+=1
        for value in by_group_type.values():
            denom=(value['tree_square_mass']*value['flat_2center_square_mass'])**.5
            value['cosine_with_flat_2center_outcome_credit']=value['tree_vs_flat_2center_dot']/denom if denom else None
        result[arm]={'group_counts':dict(groups),'by_group_type':by_group_type,
                     'wall_time_breakdown':{'sampler_and_branch_assembly_seconds':sampler_seconds,
                        'all_generation_and_scoring_seconds':generation_seconds,'PRM_forward_seconds':prm_seconds,
                        'other_generation_work_seconds':generation_seconds-sampler_seconds-prm_seconds,
                        'updates_loading_saving_and_other_seconds':read(path/'result.json')['seconds']-generation_seconds,
                        'note':'Wall-clock accounting, not FLOPs or isolated GPU kernel time. Other generation work includes HF old/reference probabilities and answer verification.'},
                     'fork_root_counts':dict(root_causes),'fork_prefix_token_quantiles':np.quantile(lengths,[0,.25,.5,.75,1]).tolist() if lengths else None,
                     'wrong_positive_total_credit_location':dict(positive_wrong),
                     'correct_leaf_PRM_minimum_location':dict(locations),
                     'note':'PRM minimum location is structural, not a human correctness label; exact minima tied across regions are ambiguous. Root-cause categories overlap. Raw tree-credit scale differs from standardized GRPO; fixed0.2 may outweigh a small positive tree signal on a correct suffix. Count these reversals rather than claiming unchanged outcome dominance.'}
    assert not torch.cuda.is_initialized()
    out=AUDIT/'mechanism_diagnostics.json';assert not out.exists()
    out.write_text(json.dumps({'passed':True,'CUDA_initialized':False,'arms':result},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
