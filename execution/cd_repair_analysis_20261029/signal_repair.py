"""Experimental pure-CPU signal functions. Not connected to any trainer.

This is a repair/control design, not an algorithm-novelty claim. Preserve old
A/B/C/D. The returned combined advantage MUST NOT be normalized a second time.
"""
import math
import statistics
import struct


def outcome_advantages(outcomes,epsilon=1e-4):
    assert len(outcomes)>=2 and all(v in (0,1) for v in outcomes)
    values=[struct.unpack('f',struct.pack('f',float(v)))[0] for v in outcomes]
    mu=statistics.fmean(values);sd=statistics.stdev(values)
    return [(v-mu)/(sd+epsilon) for v in values]


def bounded_correct_process(outcomes,process_scores,coefficient=.2,epsilon=1e-4):
    assert len(outcomes)==len(process_scores) and 0<=coefficient
    assert all(math.isfinite(v) and 0<=v<=1 for v in process_scores)
    base=outcome_advantages(outcomes,epsilon)
    correct=[m for y,m in zip(outcomes,process_scores) if y]
    center=statistics.fmean(correct) if correct else 0.
    bonus=[coefficient*(m-center) if y else 0. for y,m in zip(outcomes,process_scores)]
    assert abs(math.fsum(bonus))<1e-10
    assert all(abs(v)<=coefficient+1e-12 for v in bonus)
    final=[a+b for a,b in zip(base,bonus)]
    assert all(a<=0 for y,a in zip(outcomes,final) if not y)
    return dict(outcome_advantage=base,process_bonus=bonus,advantage=final,
                correct_process_mean=center,normalization_after_combination=False)


def reference_retention(student,reference):
    """Bernoulli KL(B||student) and analytic dL/d(student logit), no updates.

    Prospective PRM-only loss component. A frozen reference is not an oracle.
    Derivatives are at output logits, not gradients with respect to LoRA weights.
    """
    assert student and len(student)==len(reference)
    assert all(math.isfinite(v) and 0<v<1 for v in student+reference)
    terms=[q*math.log(q/p)+(1-q)*math.log((1-q)/(1-p)) for p,q in zip(student,reference)]
    return dict(mean_KL=statistics.fmean(terms),d_student_logit=[(p-q)/len(student) for p,q in zip(student,reference)],
                optimizer_updates=0)
