"""CPU replay and numerical checks; no models, optimizer, or training."""
import argparse
import ast
from collections import Counter
from pathlib import Path
import random
import sys
import torch
from common import HERE, ROOT, N, BATCH, GROUP, read, read_jsonl, sha, atomic, write_jsonl
from signals import make_signals, micro_loss, task_rewards
from alignment import native_input, align_actor, canonical_view
from vendor import pure_core as official

torch.set_num_threads(1)

def independent_reference(rewards, mask, groups):
    # Different accumulation implementation from the official Python reverse loop.
    returns = (rewards * mask).flip(-1).cumsum(-1).flip(-1) * mask
    a = torch.zeros_like(returns)
    for i, group in enumerate(groups):
        other = [j for j, g in enumerate(groups) if g == group and i != j]
        baseline = returns[other].double().sum() / (len(other) * rewards.shape[1])
        a[i] = returns[i] - baseline
    selected = a[mask.bool()].double()
    expected = (a.double() - selected.mean()) / torch.sqrt(selected.var(unbiased=True) + 1e-8)
    return (expected * mask).float(), returns

def numerical_checks():
    rng = random.Random(285)
    groups = [i // 8 for i in range(BATCH)]
    lengths = [rng.randint(10, 100) for _ in groups]
    ys = [i % 2 for i in groups]
    positions = [[2, l // 2, l - 1] for l in lengths]
    ps = [[rng.uniform(.01, .99) for _ in p] for p in positions]
    old = torch.zeros((BATCH, N)) - 2
    ref = old + torch.linspace(-.5, .5, N)[None, :]
    checks = {}
    for arm in ['P_VR','P_PRM']:
        batch = make_signals(arm, lengths, ys, ps, positions, groups, old, ref)
        expected, ret = independent_reference(batch.rewards, batch.mask, groups)
        torch.testing.assert_close(batch.advantages, expected, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(batch.returns, ret, atol=3e-6, rtol=3e-6)
        # Scalar parameter drives varying ratios, including both clipping sides.
        theta = torch.tensor(.4, requires_grad=True)
        features = torch.linspace(-2, 2, N)[None, :].expand(BATCH, -1)
        for i in range(BATCH):
            new = old[i:i+1, :lengths[i]] + theta * features[i:i+1, :lengths[i]]
            loss, _ = micro_loss(batch, i, new)
            loss.backward()
        actual_gradient = float(theta.grad)
        oracle_theta = torch.tensor(.4, requires_grad=True)
        ratio = (oracle_theta * features).exp()
        raw = torch.maximum(-batch.advantages * ratio, -batch.advantages * ratio.clamp(.8,1.2))
        # Official static mb1 mean over each response then mean over 32.
        loss = ((raw * batch.mask).sum(1) / batch.mask.sum(1)).mean()
        loss.backward()
        assert abs(actual_gradient - float(oracle_theta.grad)) < 2e-6
        changed = make_signals(arm, lengths, [1-y for y in ys], ps, positions, groups, old, ref)
        assert torch.equal(changed.advantages, batch.advantages) == (arm == 'P_PRM')
        batch.advantages[0,0] += 1
        caught = False
        try:
            micro_loss(batch, 0, old[0:1, :lengths[0]])
        except ValueError:
            caught = True
        assert caught
        checks[arm] = dict(parity=True, fixed_width=N, whitening_calls=1,
            accumulated_gradient=actual_gradient, reference_gradient=float(oracle_theta.grad),
            outcome_independence=(arm=='P_PRM'), mutation_rejected=caught)
    # Retain official padded-width behavior (not ordinary sequence RLOO).
    r=torch.zeros((2,6));r[0,1]=1
    m=torch.tensor([[1,1,0,0,0,0],[1,1,1,1,0,0]])
    a,_=official.compute_rloo_outcome_advantage(r,m,[0,0],adv_norm=False)
    assert abs(float(a[1,0])+1/3)<1e-6
    vendor = read(HERE / 'vendor/manifest.json')
    assert sha(HERE / 'vendor/pure_core.py') == vendor['file_sha256']
    tree = ast.parse((HERE / 'vendor/pure_core.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            import hashlib
            source=ast.get_source_segment((HERE/'vendor/pure_core.py').read_text(),node)
            assert hashlib.sha256(source.encode()).hexdigest()==vendor['functions'][node.name]['sha256']
    return dict(passed=True, arms=checks, official_functions_unchanged=True,
        vr_padding_retained=True, no_optimizer=True)

def tokenizer_checks(tok, prm_tok):
    cases = ['First.\n\nSecond.', 'First.\r\n\r\nSecond.',
        ' x = 2. \n\n  y = 4. \n', 'a\n\n\n\nb\n\n\n\n',
        '中文 α² = 4.\n\n答案是 2。', '😀 + 1 = 2\r\n\r\nDone.',
        'No box and no separator', '\n\n\n\n', '', ' \r\n\t ',
        'x=2\n\n\\boxed{2}', 'First\n\nSecond without EOS']
    results=[]
    for i,raw in enumerate(cases):
        ids=tok.encode(raw,add_special_tokens=False)
        if i != len(cases)-1:ids.append(tok.eos_token_id)
        enc=native_input(prm_tok,'Compute 1+1.',raw)
        mapped=align_actor(tok,ids,raw,enc['spans'],empty_terminal_step=enc['empty_terminal_step'])
        # A normal nonfinal reward must include the separator itself.
        for end in mapped['actor_positions'][:-1]:
            prefix=tok.decode(ids[:end+1],skip_special_tokens=True,clean_up_tokenization_spaces=False).replace('\r\n','\n')
            assert '\n\n' in prefix
        results.append(dict(case=i,raw=raw,**mapped,crlf=enc['crlf_replacements'],
            native_positions=enc['positions'], empty=enc['empty_terminal_step']))
    failures={}
    for key, func in {
        'text_mismatch':lambda:align_actor(tok,[tok.eos_token_id],'x',[dict(raw_start=0,raw_end=1)]),
        'after_eos':lambda:align_actor(tok,[tok.eos_token_id,1],'',[dict(raw_start=0,raw_end=0)],empty_terminal_step=True),
        'context_overflow':lambda:native_input(prm_tok,'q','x',max_length=1),
        'marker_injection':lambda:native_input(prm_tok,'q','x<extra_0>')
    }.items():
        try:func()
        except ValueError:failures[key]=True
        else:raise AssertionError(key)
    return dict(passed=True,cases=results,fail_closed=failures)

def replay(tok, prm_tok, root, out):
    source=root/'runs/B_corrected_seed17_6M_20261022/rollouts.jsonl'
    rows=read_jsonl(source)
    inventory=Counter(); mapped_rows=[]; complete=[]; errors=[]
    for i,r in enumerate(rows):
        raw=r['completion']; ids=tok.encode(raw,add_special_tokens=False)
        if r['terminated']:ids.append(tok.eos_token_id)
        view=canonical_view(raw,True)
        spans=view['spans'] or [dict(raw_start=0,raw_end=len(raw))]
        try:
            mapped=align_actor(tok,ids,raw,spans,empty_terminal_step=not view['steps'])
        except (ValueError,AssertionError) as exc:
            errors.append(dict(row=i,error=str(exc)));mapped=None
        inventory['rows']+=1
        inventory['reconstructed_length_matches_logged']+=len(ids)==r['completion_tokens']
        inventory['crlf_responses']+=view['crlf_replacements']>0
        inventory['scored_in_old_B']+=bool(r['step_scores'])
        inventory['truncated']+=not r['terminated']
        if mapped:
            inventory['aligned_rows']+=1
            inventory['mapped_steps']+=len(spans)
        good=(mapped is not None and bool(r['step_scores']) and len(r['step_scores'])==len(spans))
        mapped_rows.append(dict(source_row=i,problem_id=r['problem_id'],length=len(ids),
            outcome=r['outcome_reward'],scores=r['step_scores'],
            positions=mapped['actor_positions'] if mapped else [],usable_for_saved_score_replay=good))
    for i in range(0,len(rows),GROUP):
        sub=mapped_rows[i:i+GROUP]
        if len(sub)==GROUP and len(set(r['problem_id'] for r in sub))==1 and all(r['usable_for_saved_score_replay'] for r in sub):
            complete.append(sub)
    # Fixed first 32 complete scored groups, never selected by advantage/result.
    chosen=complete[:32]
    summaries=[]; ledger=[]
    for offset in range(0,len(chosen)-3,4):
        batch=sum(chosen[offset:offset+4],[])
        groups=[i//GROUP for i in range(BATCH)]
        zeros=torch.zeros((BATCH,N))
        arms={}
        for arm in ['P_VR','P_PRM']:
            s=make_signals(arm,[r['length'] for r in batch],[r['outcome'] for r in batch],
                [r['scores'] for r in batch],[r['positions'] for r in batch],groups,zeros,zeros)
            expected,_=independent_reference(s.rewards,s.mask,groups)
            torch.testing.assert_close(s.advantages,expected,atol=4e-6,rtol=4e-6)
            vals=s.advantages[s.mask.bool()]
            arms[arm]=dict(valid_tokens=int(s.mask.sum()),mean=float(vals.mean()),std=float(vals.std()),
                positive_tokens=int((vals>0).sum()),normalization_calls=s.whitening_calls,
                reward_sum=float(s.task_rewards.sum()), max_abs_reference_difference=float((expected-s.advantages).abs().max()))
            for j,r in enumerate(batch):
                ledger.append(dict(arm=arm,batch=offset//4,source_row=r['source_row'],
                    positions=r['positions'],terminal_task_reward=float(s.task_rewards[j,r['length']-1]),
                    step_task_rewards=s.task_rewards[j,r['positions']].tolist(),
                    advantage_at_steps=s.advantages[j,r['positions']].tolist(),
                    mean_valid_advantage=float(s.advantages[j,:r['length']].mean())))
        summaries.append(arms)
    assert summaries
    write_jsonl(out/'historical_replay_ledger.jsonl',ledger)
    return dict(passed=not errors,source=str(source),source_sha256=sha(source),inventory=dict(inventory),
        errors=errors[:20],available_complete_scored_groups=len(complete),replayed_groups=len(chosen),
        replayed_responses=len(chosen)*GROUP,batches=summaries,
        limitations=['Historical raw token IDs absent: re-tokenized text, not original sampled IDs.',
            'Reuse only existing nonempty PRM scores; missing-score rows are audited for alignment but not assigned invented scores.',
            'Historical KL set to zero for math replay: old/ref log-probs were not logged. Real preflight computes both.',
            'These signal checks do not estimate trained model accuracy.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokenizers',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    atomic(args.output/'core_checks.json',numerical_checks())
    if args.tokenizers:
        from transformers import AutoTokenizer
        assets=read(ROOT/'receipts/assets.json')
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        pt=AutoTokenizer.from_pretrained(ROOT/'models/qwen25_math_prm_7b',local_files_only=True)
        atomic(args.output/'tokenizer_checks.json',tokenizer_checks(tok,pt))
        atomic(args.output/'offline_replay.json',replay(tok,pt,ROOT,args.output))
    print('CPU checks completed; no model loaded or optimizer created',flush=True)

if __name__=='__main__':main()
