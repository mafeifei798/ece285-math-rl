"""CPU-only aggregation of fixed report evidence and a matched-sample replay.

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
    p = ROOT / "execution/abc_seed17_resumed_v1_20261104/completed_evidence/C_mix/compact_signals.jsonl"
    INPUTS[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[(r['optimizer_step_before_update'],r['problem_id'])].append(r)
    kinds, actual, replay, errors = Counter(), [], [], []
    for g in groups.values():
        assert len(g) == 8
        ys = [r['outcome_reward'] for r in g]
        kind = 'all_wrong' if sum(ys) == 0 else 'all_correct' if sum(ys) == 8 else 'mixed'
        kinds[kind] += 1
        if kind == 'all_wrong':
            ms = [r['min_score'] for r in g]
            mu, sd = statistics.mean(ms), statistics.stdev(ms)
            for r, m in zip(g, ms):
                a = r['advantage']
                errors.append(abs(a-(m-mu)/(sd+5e-4)))
                actual.append(a)
                replay.append(0.)
    assert len(actual) == 1696 and sum(a > 0 for a in actual) == 524
    assert max(errors) < 1e-5
    figdir = HERE / 'figures'
    figdir.mkdir(exist_ok=True)
    a = np.sort(actual)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(6.6,2.15),layout='constrained')
    ax.axvspan(0,2.8,color='#fcf0e6',zorder=0)
    ax.step(np.r_[a[0]-.1,a],np.r_[0,np.arange(1,len(a)+1)/len(a)],
            where='post',color='#bd572b',lw=1.9,label='Direct mix: actual loss advantages')
    ax.plot([-2.8,0,0,2.8],[0,0,1,1],color='#246a8d',lw=1.9,
            label='C-bounded: same-score offline replay')
    ax.text(.98,.28,'Positive: 524 / 1,696 (30.9%)\nBounded replay: 0 / 1,696',
            ha='right',transform=ax.transAxes,fontsize=9)
    ax.set(xlim=(-2.8,2.8),ylim=(-.025,1.035),xlabel='Reward advantage',ylabel='Empirical CDF')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    ax.legend(loc='upper left',frameon=False,fontsize=8.6)
    fig.savefig(figdir/'all_wrong_advantage_ecdf.pdf')
    fig.savefig(figdir/'all_wrong_advantage_ecdf.png',dpi=180)
    plt.close(fig)
    with (figdir/'all_wrong_advantage_ecdf.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['sorted_direct_mix_advantage','empirical_cdf'])
        w.writerows(zip(a,np.arange(1,len(a)+1)/len(a)))
    followups = {}
    for seed, directory in [(17,'abc_seed17_resumed_v1_20261104'),(42,'ac_seed42_v1_20261106')]:
        d=read(ROOT/f'execution/{directory}/completed_evidence/comparison.json')
        followups[str(seed)]={k:d[k] for k in ['passed','global_step','summary','paired','accounting']}
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
        'figure':{'same_rollouts':True,'groups':dict(kinds),'n_all_wrong':len(actual),
            'positive_actual':sum(a>0 for a in actual),'positive_bounded_replay':0,
            'maximum_formula_replay_difference':max(errors),
            'scope':'Counterfactual advantage formula only; not an independently trained policy.'},
        'PRM_adaptation_probe':cd['processbench'],
        'placeholder_protocol':formats['analysis_spec'],
        'placeholder_rates':{k:formats['summaries'][k]['rates']['leading_placeholder_box_with_body']
            for k in ['initial/initial/math500','17/A/math500','17/C_bounded/math500']},
        'placeholder_counterfactual':status['placeholder_counterfactual'],
        'inputs_sha256':INPUTS}
    (HERE/'report_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'all_wrong_rows':len(actual),'positive_actual':524,
        'initial_V1':initial,'GPU_used':False}))

if __name__=='__main__':main()
