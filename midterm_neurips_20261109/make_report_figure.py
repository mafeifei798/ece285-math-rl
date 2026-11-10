"""CPU-only aggregation of fixed report evidence and observed advantages.

Does not load any policy/PRM, alter saved labels, or launch training.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = {}

def read(path):
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def main():
    p = ROOT / "execution/ab_corrected_6m_seed17_20261022/final_evidence/raw/runs/B_corrected_seed17_6M_20261022/rollouts.jsonl"
    INPUTS[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[(r['optimizer_step_before_update'],r['problem_id'])].append(r)
    kinds, distributions, errors = Counter(), defaultdict(list), []
    wrong_positive = Counter()
    for g in groups.values():
        assert len(g) == 8
        ys = [r['outcome_reward'] for r in g]
        kind = 'all_wrong' if sum(ys) == 0 else 'all_correct' if sum(ys) == 8 else 'mixed'
        kinds[kind] += 1
        rs = [r['reward'] for r in g]
        mu, sd = statistics.mean(rs), statistics.stdev(rs)
        for r, reward in zip(g, rs):
            a = r['advantage']
            errors.append(abs(a-(reward-mu)/(sd+1e-4)))
            distributions[kind].append(a)
            if r['outcome_reward'] == 0 and a > 0:
                wrong_positive[kind] += 1
    assert kinds == {'mixed':878,'all_wrong':212,'all_correct':30}
    assert wrong_positive == {'all_wrong':571,'mixed':6}
    assert max(errors) < 1e-5
    figdir = HERE / 'figures'
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(6.6,2.15),layout='constrained')
    ax.axvspan(0,2.8,color='#fcf0e6',zorder=0)
    for key,color,label in [('all_wrong','#bd572b','All wrong (1,696)'),
                            ('mixed','#246a8d','Mixed (7,024)'),
                            ('all_correct','#51854b','All correct (240)')]:
        a = np.sort(distributions[key])
        ax.step(np.r_[-2.8,a,2.8],np.r_[0,np.arange(1,len(a)+1)/len(a),1],
            where='post',color=color,lw=1.9,label=label)
    ax.axvline(0,color='#555555',lw=.7,ls=':',zorder=0)
    ax.text(.98,.15,'All-wrong responses with positive advantage:\n571 / 1,696 (33.7%)',
            ha='right',transform=ax.transAxes,fontsize=9)
    ax.set(xlim=(-2.8,2.8),ylim=(-.025,1.035),xlabel='Reward advantage',ylabel='Empirical CDF')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    ax.legend(loc='upper left',frameon=False,fontsize=8.6)
    fig.savefig(figdir/'b_advantage_by_group.pdf')
    fig.savefig(figdir/'b_advantage_by_group.png',dpi=180)
    plt.close(fig)
    with (figdir/'b_advantage_by_group.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['group_type','sorted_advantage','empirical_cdf'])
        for key,aa in distributions.items():
            a=np.sort(aa)
            w.writerows((key,float(v),(i+1)/len(a)) for i,v in enumerate(a))
    followups = {}
    for seed, directory in [(17,'abc_seed17_resumed_v1_20261104'),(42,'ac_seed42_v1_20261106')]:
        d=read(ROOT/f'execution/{directory}/completed_evidence/comparison.json')
        followups[str(seed)]={k:d[k] for k in ['passed','global_step','summary','paired','accounting']}
        for key in ['training_seconds','resumed_training_seconds_since_20261104']:
            if key in d:followups[str(seed)][key]=d[key]
    cd=read(ROOT/'execution/cd_prm_7b_pair_20261028/artifacts/summary.json')
    formats=read(ROOT/'execution/answer_order_audit_20261108/results.json')
    status=read(ROOT/'execution/ac_status_cpu_20261109/summary.json')
    initial={}
    for ds,n,expected in [('math500',500,245),('gsm8k',1319,750)]:
        p=ROOT/f'execution/final_four_eval_20261029/evidence/runs/Final_four_eval_seed17_1M_20261029/initial/{ds}.jsonl'
        INPUTS[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        rr=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        # These shared saved prediction files use reward for final correctness.
        assert len(rr)==n
        key='reward' if 'reward' in rr[0] else 'outcome_reward'
        correct=sum(r[key] for r in rr)
        assert correct==expected
        initial[ds]={'n':n,'correct':correct,'accuracy':correct/n,'protocol':'V1'}
    result={'passed':True,'GPU_used':False,'new_model_calls':0,'optimizer_updates':0,
        'initial_V1':initial,'followups':followups,
        'figure':{'actual_V0_B_loss_inputs':True,'groups':dict(kinds),
            'n_by_group':{k:len(v) for k,v in distributions.items()},
            'wrong_positive_advantages':dict(wrong_positive),
            'maximum_formula_replay_difference':max(errors),
            'scope':'All curves are empirical distributions of saved training advantages.'},
        'PRM_adaptation_probe':cd['processbench'],
        'placeholder_protocol':formats['analysis_spec'],
        'placeholder_rates':{k:formats['summaries'][k]['rates']['leading_placeholder_box_with_body']
            for k in ['initial/initial/math500','17/A/math500','17/C_bounded/math500']},
        'placeholder_counterfactual':status['placeholder_counterfactual'],
        'inputs_sha256':INPUTS}
    (HERE/'report_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'groups':dict(kinds),'wrong_positive':dict(wrong_positive),
        'initial_V1':initial,'GPU_used':False}))

if __name__=='__main__':main()
