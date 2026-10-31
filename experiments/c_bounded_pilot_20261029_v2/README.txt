Authorized three-arm seed17 policy pilot, approximately1M actual generated completion tokens per arm.

V2 recovery: retain V1 source/failure unchanged. V1 failed in the first generation
diagnostic before loss/backward/any optimizer update, with no checkpoint. A's
"PRM not requested" flag was incorrectly passed to the C invalid-output guard.
V2 fixes only that A diagnostic. A keeps the actual original TRL outcome tensor;
C_mix and C_bounded reward/objective formulas are unchanged. CPU regression
checks A's no-PRM case and that C's invalid-output guard remains strict.
Persist every actual generation/token pack before signal checks. V1's first
generated batch was not saved: count its unknown runtime/token overhead separately
(at most65536 completion tokens), never claim those samples were recovered.
All library/runtime caches for V2 are explicitly routed under the project cache/.

A: original outcome GRPO. C_mix: frozen adapted C PRM, original mixed reward GRPO.
C_bounded: same frozen C PRM, audited correct-only bounded process advantage.
The bounded rule is an algorithm change; C_mix remains the original mixed baseline.
No D retention adaptation, extra seeds, 6M expansion, PRM updates, new prompts or verifier rules.

All new remote code, logs, optimizer states and checkpoints live under /srv/research/ece285_math_rl_project.
Old A/B/C/D source and result hashes are verified before/after each stage.
All arms first execute2 real optimizer updates, save complete state, and continue from
that checkpoint only after all smoke gates pass. These tokens count toward1M.
Resume validates LoRA, optimizer, scheduler, PyTorch RNG and actual next prompt group.
The pinned colocated vLLM sampling RNG restarts on process reload: no uninterrupted
bitwise-equivalence claim. All arms use the same restart schedule and fixed seeds.

Keep pinned original actor args, LoRA16, group8, generation32, accumulation32,
cap2048, temperature1, LR5e-6, beta.04. Shared audit cache fraction.15 is used
for all arms on the previously authorized shared GPU1. C scoring alone disables
inherited TrainingArguments tf32=True and restores actor precision immediately.
PRM parameters never enter the policy optimizer. Full C parameter hash is checked.

Only existing500-question held-out MATH validation is evaluated for this pilot,
with greedy original prompts/cap2048/repaired verifier. No test-set selection.
Report paired bootstrap intervals, prompt overlap audit, actual token overshoot,
updates and PRM cost; no claim of seed robustness or improved reasoning from
the guaranteed zero wrong-positive-advantage rate alone.

controller.py is a finite authorized dependency pipeline, not a recurring monitor.
It never retries failed stages, never kills jobs, and never restarts an existing
attempt from base. Failure evidence and checkpoint must be inspected for recovery.
