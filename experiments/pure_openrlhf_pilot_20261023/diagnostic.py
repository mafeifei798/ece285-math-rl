"""CPU-only, read-only descriptive diagnostics for the fixed PURE pilot.

No reward, filtering, stopping, generation, training, tokenizer or torch import.
summarize_batch(rows, scored=None, batch=None) returns JSON-compatible values.
Masks/advantages, when supplied, must already be on CPU. Raw generated text and
IDs remain distinct from the official processed action sequence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re


BOX = re.compile(r'\\boxed\s*\{')
SHORT_LIMIT = 32


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _distribution(values):
    values = sorted(float(x) for x in values if x is not None)
    if not values:
        return dict(count=0, mean=None, min=None, p25=None, median=None, p75=None,
                    p90=None, p95=None, max=None)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Diagnostic input contains nonfinite numbers')
    def quantile(q):
        pos=(len(values)-1)*q
        low=math.floor(pos); high=math.ceil(pos)
        return values[low]+(values[high]-values[low])*(pos-low)
    return dict(count=len(values),mean=math.fsum(values)/len(values),min=values[0],
                p25=quantile(.25),median=quantile(.5),p75=quantile(.75),
                p90=quantile(.9),p95=quantile(.95),max=values[-1])


def _plain_cpu(value, name):
    device=getattr(value,'device',None)
    if device is not None and getattr(device,'type',str(device))!='cpu':
        raise ValueError(f'{name} must already be on CPU; diagnostics never transfer GPU tensors')
    if hasattr(value,'detach'):
        value=value.detach()
    if hasattr(value,'tolist'):
        value=value.tolist()
    return value


def _field(batch, name, default=None):
    return batch.get(name,default) if isinstance(batch,dict) else getattr(batch,name,default)


def _matrix(batch, name, count, *, required=False):
    value=_field(batch,name)
    if value is None:
        if required:
            raise ValueError(f'Signal batch is missing {name}')
        return None
    result=_plain_cpu(value,name)
    if not isinstance(result,(list,tuple)) or len(result)!=count:
        raise ValueError(f'{name} batch dimension must match rows')
    if any(not isinstance(row,(list,tuple)) for row in result):
        raise ValueError(f'{name} must be a two-dimensional array')
    return result


def _row_info(row,index):
    text=row['emitted_completion']
    if not isinstance(text,str):
        raise ValueError('emitted_completion must be the actual decoded raw string')
    ids=_plain_cpu(row['emitted_completion_ids'],'emitted_completion_ids')
    if not isinstance(ids,(list,tuple)):
        raise ValueError('emitted_completion_ids must be a sequence')
    aligned=row.get('alignment') or {}
    terminated=bool(row['original_terminated'])
    # The explicit emitted flag is best evidence. Fallback is safe when the
    # official processed sequence exposes its terminal EOS ID; never infer EOS
    # merely because process_sequences appended/replaced the last action.
    actor=aligned.get('actor_completion_ids',row.get('completion_ids'))
    actor=_plain_cpu(actor,'actor_completion_ids') if actor is not None else None
    raw_eos_flag=aligned.get('emitted_ended_in_eos')
    if raw_eos_flag is not None:
        ended_in_eos=bool(raw_eos_flag)
        eos_evidence='alignment.emitted_ended_in_eos'
    elif terminated and len(ids)==1:
        ended_in_eos=True
        eos_evidence='original_terminated and one emitted token'
    elif terminated and actor and ids and aligned.get('process_sequences_applied'):
        ended_in_eos=ids[-1]==actor[-1]
        eos_evidence='original_terminated and last ID equals official processed terminal EOS'
    elif not terminated:
        ended_in_eos=False
        eos_evidence='original_terminated=False'
    else:
        ended_in_eos=None
        eos_evidence='unknown: no terminal-ID evidence'
    content_tokens=len(ids)-1 if ended_in_eos is True and ids else (len(ids) if ended_in_eos is False else None)
    boundaries=aligned.get('actor_boundary_positions')
    declared_steps=aligned.get('step_count')
    steps=len(boundaries) if boundaries is not None else declared_steps
    processed_tokens=len(actor) if actor is not None else aligned.get('response_tokens')
    processed_has_terminal=bool(actor) and bool(aligned.get('process_sequences_applied'))
    processed_content=processed_tokens-1 if processed_has_terminal else None
    outcome=row.get('outcome')
    if outcome is not None and outcome not in (0,1,0.,1.):
        raise ValueError('Outcome must be unchanged binary verifier result or None')
    empty=not text.strip()
    short=content_tokens<=SHORT_LIMIT if content_tokens is not None else None
    return dict(index=index,problem_id=row.get('problem_id'),outcome=outcome,
        empty=empty,literal_empty_text=text=='',whitespace_only=text!='' and empty,
        natural_eos_only=terminated and len(ids)==1 and ended_in_eos is True,
        short=short,short_nonempty=short and not empty if short is not None else None,
        truncated=not terminated,no_box=BOX.search(text) is None,
        no_sep=steps<=1 if steps is not None else None,
        raw_no_literal_double_lf='\n\n' not in text,contains_crlf='\r\n' in text,
        emitted_tokens=len(ids),raw_content_tokens=content_tokens,
        processed_action_tokens=processed_tokens,processed_content_tokens=processed_content,
        step_count=steps,separator_count=max(steps-1,0) if steps is not None else None,
        raw_ended_in_eos=ended_in_eos,raw_eos_evidence=eos_evidence,
        terminal_eos_flag_contradiction=ended_in_eos is True and not terminated,
        step_count_disagreement=boundaries is not None and declared_steps is not None and len(boundaries)!=declared_steps,
        processed_action_changed=aligned.get('process_sequences_changed'),
        terminal_replaced_with_eos=aligned.get('terminal_replaced_with_eos'))


def _group_outcomes(info):
    groups=defaultdict(list)
    missing=0
    for row in info:
        if row['problem_id'] is None:
            missing+=1
        else:
            groups[row['problem_id']].append(row['outcome'])
    counts=Counter()
    for values in groups.values():
        if any(value is None for value in values): counts['unknown']+=1
        elif all(value==1 for value in values): counts['all_correct']+=1
        elif all(value==0 for value in values): counts['all_wrong']+=1
        else: counts['mixed']+=1
    return dict(group_count=len(groups),rows_without_problem_id=missing,
                counts={key:counts[key] for key in ('all_correct','all_wrong','mixed','unknown')},
                group_sizes=_distribution([len(values) for values in groups.values()]))


def _process_scores(scored,count):
    if scored is None:
        return None
    if len(scored)!=count:
        raise ValueError('scored must align one-to-one with rows')
    result={}
    # Keep native BF16 values separate from FP32 descriptive probabilities.
    for field in ('native_step_scores','native_signed_scores','step_scores','signed_scores'):
        all_steps=[];mins=[];means=[];present=0
        for record in scored:
            values=record.get(field)
            if values is None:
                continue
            values=list(_plain_cpu(values,field))
            if not values:
                continue
            values=[float(value) for value in values]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'Nonfinite {field}')
            present+=1;all_steps.extend(values);mins.append(min(values));means.append(math.fsum(values)/len(values))
        result[field]=dict(responses_with_scores=present,steps=_distribution(all_steps),
                           response_min=_distribution(mins),response_mean=_distribution(means))
    result['note']='Describes saved scores only; no softmin/reward is recomputed or replaced.'
    return result


def _advantage_summary(batch,info):
    if batch is None:
        return None,None
    count=len(info)
    mask=_matrix(batch,'mask',count,required=True)
    advantage=_matrix(batch,'advantages',count,required=True)
    boundary=_matrix(batch,'boundary',count)
    task=_matrix(batch,'task',count)
    task_sums=[];valid_count=positive_count=boundary_positive=0
    positive_mass=0.
    group_stats={key:dict(responses=0,valid_tokens=0,positive_tokens=0,positive_advantage_mass=0.,responses_with_positive_advantage=0)
                 for key in ('empty','whitespace_only','natural_eos_only','short','short_nonempty','truncated','no_box','no_sep')}
    for i,row in enumerate(info):
        widths=[len(mask[i]),len(advantage[i])]
        if boundary is not None: widths.append(len(boundary[i]))
        if task is not None: widths.append(len(task[i]))
        if len(set(widths))!=1:
            raise ValueError('Signal row mask/advantage/boundary/task widths must match')
        if any(value not in (True,False,0,1) for value in mask[i]):
            raise ValueError('Signal mask must be boolean')
        valid=[j for j,value in enumerate(mask[i]) if value]
        values=[float(advantage[i][j]) for j in valid]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Valid advantages must be finite')
        pos=[max(value,0.) for value in values]
        mass=math.fsum(pos);number=sum(value>0 for value in values)
        positive_mass+=mass;positive_count+=number;valid_count+=len(valid)
        row['advantage_valid_tokens']=len(valid)
        row['positive_advantage_tokens']=number
        row['positive_advantage_mass']=mass
        row['has_positive_advantage']=number>0
        if boundary is not None:
            boundary_positive+=sum(bool(boundary[i][j]) and advantage[i][j]>0 for j in valid)
        if task is not None:
            task_sum=math.fsum(float(task[i][j]) for j in valid)
            if not math.isfinite(task_sum): raise ValueError('Nonfinite valid task reward')
            task_sums.append(task_sum);row['task_reward_sum']=task_sum
        for key,stats in group_stats.items():
            if row[key] is True:
                stats['responses']+=1;stats['valid_tokens']+=len(valid)
                stats['positive_tokens']+=number;stats['positive_advantage_mass']+=mass
                stats['responses_with_positive_advantage']+=number>0
    for stats in group_stats.values():
        stats['share_of_all_positive_tokens']=_ratio(stats['positive_tokens'],positive_count)
        stats['share_of_all_positive_advantage_mass']=_ratio(stats['positive_advantage_mass'],positive_mass)
        stats['within_category_positive_token_rate']=_ratio(stats['positive_tokens'],stats['valid_tokens'])
    result=dict(arm=_field(batch,'arm'),valid_tokens=valid_count,positive_tokens=positive_count,
        positive_advantage_mass=positive_mass,positive_token_rate=_ratio(positive_count,valid_count),
        positive_boundary_tokens=boundary_positive if boundary is not None else None,
        categories=group_stats,
        meaning='Normalized advantages supplied to loss; positive mass is sum(max(A_t,0)) over valid tokens. Categories overlap.',
        no_gradient_magnitude_claim=True)
    rewards=dict(arm=_field(batch,'arm'),response_task_reward_sum=_distribution(task_sums),
        meaning='Sum of the supplied task matrix before baseline/KL; P-VR task matrix is structurally zero, not its verifier reward.') if task is not None else None
    return result,rewards


def summarize_batch(rows,scored=None,batch=None):
    """Summarize one fixed batch without modifying rows, scores or signals."""
    info=[_row_info(row,index) for index,row in enumerate(rows)]
    counts={};rates={};known={}
    flags=('empty','literal_empty_text','whitespace_only','natural_eos_only','short',
           'short_nonempty','truncated','no_box','no_sep','raw_no_literal_double_lf',
           'contains_crlf','terminal_eos_flag_contradiction','step_count_disagreement')
    for key in flags:
        known[key]=sum(row[key] is not None for row in info)
        counts[key]=sum(row[key] is True for row in info)
        rates[key]=_ratio(counts[key],known[key])
    outcomes=[row['outcome'] for row in info if row['outcome'] is not None]
    advantage,task=_advantage_summary(batch,info)
    return dict(schema_version=1,rows=len(info),descriptive_only=True,
        reward_changed=False,filtered_rows=0,stop_decision=None,
        definitions=dict(empty='raw emitted_completion.strip() == empty string',
            whitespace_only='raw string is nonempty but strip() is empty',
            natural_eos_only='one raw emitted token, original_terminated=True and evidence that token is terminal EOS',
            short='raw content token count <=32, includes empty; subtract one only for evidenced original terminal EOS',
            no_sep='official alignment has <=1 step; differs from raw double-newline text marker',
            no_box='raw text lacks a literal backslash-boxed followed by opening brace; not a verifier reclassification',
            truncated='original_terminated=False, including cases where official processing inserts/replaces EOS',
            processed_content_tokens='official processed action length minus its terminal EOS; may differ from raw generated content',
            quantiles='linear interpolation at (n-1)*q',missing='unknowns excluded from each rate denominator; zero denominator -> null'),
        counts=counts,rates=rates,known_denominators=known,
        lengths={name:_distribution([row[name] for row in info]) for name in
                 ('emitted_tokens','raw_content_tokens','processed_action_tokens','processed_content_tokens','step_count','separator_count')},
        outcomes=dict(known=len(outcomes),correct=sum(value==1 for value in outcomes),
                      incorrect=sum(value==0 for value in outcomes),missing=len(info)-len(outcomes),
                      accuracy=_ratio(sum(value==1 for value in outcomes),len(outcomes))),
        groups=_group_outcomes(info),process_scores=_process_scores(scored,len(info)),
        task_rewards=task,advantages=advantage,per_response=info)
