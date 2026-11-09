"""Post-hoc descriptive interpretation only; reads collected immutable evidence."""
from pathlib import Path
import json
from collections import Counter

HERE=Path(__file__).resolve().parent
E=HERE/'completed_evidence'

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.open() if x.strip()]

def main():
    assert read(HERE/'completion_verification.json')['passed']
    comp=read(E/'comparison.json')
    result={}
    for arm in ('tree_vr','tree_cb'):
        trace=rows(E/arm/'tree_traces.jsonl');rr=rows(E/arm/'rollouts.jsonl')
        roots=[r for t in trace for r in t['roots']]
        assert len(roots)==384
        counts=Counter();examples=[]
        for batch,t in enumerate(trace):
            sub=rr[batch*32:batch*32+32]
            for i,r in enumerate(t['roots']):
                if r['fallback'] and len(examples)<3:
                    leaf=sub[i*4]
                    examples.append({'problem_id':leaf['problem_id'],'prefix_length':r['cut'],
                        'first_answer_marker_token_position':r['excluded_tail_start'],
                        'root_total_tokens':len(r['initial_ids']),
                        'completion_start':leaf['completion'][:450]})
            for g in range(4):
                group=sub[g*8:g*8+8];p=sum(r['outcome_reward'] for r in group)/8
                correct=[r for r in group if r['outcome_reward']]
                mu=sum(r['min_score'] for r in correct)/len(correct) if correct else 0.
                for j,r in enumerate(group):
                    cut=t['rows'][g*8+j]['prefix_length']
                    if r['outcome_reward']:
                        counts['answer_correct_leaves']+=1
                        counts['answer_correct_leaves_empty_prefix' if not cut else 'answer_correct_leaves_nonempty_prefix']+=1
                        if cut>=32:counts['answer_correct_leaves_prefix_at_least32_tokens']+=1
                    if arm=='tree_cb' and r['outcome_reward'] and 0<p<1:
                        parent=sum(x['outcome_reward'] for x in group[(j//4)*4:(j//4)*4+4])/4 if cut else p
                        outcome=(1-p)+(1-parent);bonus=.2*(r['min_score']-mu)
                        if outcome>0 and outcome+bonus<0:
                            counts['answer_correct_leaves_positive_outcome_reversed_by_process']+=1
                            counts['reversed_suffix_tokens']+=len(r['completion_token_ids'])-cut
        n=len(roots);nonempty=sum(bool(r['cut']) for r in roots)
        long=sum(r['cut']>=32 for r in roots)
        cost=comp['training_costs'][arm];base=comp['training_costs']['chain_cb']
        result[arm]={'roots':n,'fallback_roots':n-nonempty,'fallback_rate':(n-nonempty)/n,
            'nonempty_forks':nonempty,'forks_at_least32_tokens':long,
            'forks_at_least32_tokens_fraction_of_all_roots':long/n,
            'unique_vs_expanded_token_saving_fraction':1-cost['unique_generation_tokens']/cost['expanded_leaf_training_tokens'],
            'walltime_ratio_to_chain':cost['seconds']/base['seconds'],
            'counts':dict(counts),'fallback_examples':examples}
    mechanism=read(E/'admission/mechanism_diagnostics.json')['arms']['tree_cb']
    assert result['tree_cb']['counts'].get('reversed_suffix_tokens',0)==mechanism['by_group_type']['mixed']['correct_tokens_reversed_from_positive_outcome_to_negative_total']
    out={'post_hoc_descriptive_only':True,'no_training_or_regeneration':True,'arms':result,
        'interpretation_limits':[
            'A32-token prefix is a coverage descriptor, not a mathematical-validity label or tuned branch threshold.',
            'First boxed output can precede substantial reasoning; masking at its first token prevents meaningful later forks in this fixed protocol.',
            'Tree changes sampling, dependency structure and credit scale; coverage/cost statistics do not establish the cause of accuracy differences.',
            'Original NumPy20k secondary tree_vr interval ends at-0.2pp; independent stdlib20k ends at0.0pp. Exact McNemar p0.065; do not claim a robust significant loss from that boundary.'
        ]}
    (HERE/'interpretation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
