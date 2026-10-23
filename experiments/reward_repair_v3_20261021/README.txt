Reward repair v3: percent-unit verification and CRLF-only native PRM input repair

This is an independent opt-in version, not installed in the original trainers.
It never loads an actor checkpoint, generates answers, resumes RL, updates a
PRM, or modifies the original A/B/Bprime sources, data, checkpoints or results.
It reuses the previous version's reviewed unit rules and archived legacy code.

Production-facing interfaces in repairs.py

1. repaired_outcome(text, gold, terminated, question=..., problem_id=...,
                    unit_metadata=...)
   A fixed question/gold-hash-bound manifest specifies requested percentage
   units. The explicit suffix '%' or '\%' is removed only within those typed
   questions, from BOTH candidate and gold. 15 and 15% match gold15 when the
   requested answer unit is percent-value; 0.15 does not. Probability questions
   retain dimensionless semantics, so 15% and0.15 may match there, while15 does
   not. Merely mentioning a discount/rate does not make the output a percent.
   Unknown questions retain the archived verifier; metadata mismatches raise.
   The reviewed manifest here has67 questions across train and evaluation,
   including one percentage-point question. It is not a universal unit parser.

2. CanonicalPRMAdapter(frozen_scorer).score(questions, raw_completions)
   Makes a separate view replacing CRLF with LF. It preserves actor text and
   raw-character span mappings, then uses the existing native double-LF steps.
   It introduces no sentence splitting, extra boundaries, PRM parameters or
   scoring head. Validity is determined before scoring. Unscored invalid rows
   are not treated as true zero-valued PRM judgments.

3. Use valid_for_prm and mixed_reward from repairs.py together with the new
   verifier. unit_exact/unit_symbolic are valid statuses: using the old
   training_b status whitelist would incorrectly suppress their process reward.
   The bug-fix reward remains y+0.2*min(p), invalid=0. GRPO group8, sample std,
   epsilon1e-4, token cap, seed, actor prompt, and all original data stay fixed.

Replay variants on identical stored 7B-B responses and original eight partners

logged: original labels and original frozen score cache.
matched_legacy: original labels, paired re-inference of old PRM inputs.
units_only: fixed unit labels, same old-input scores as matched_legacy.
crlf_only: original labels, normalized-input scores on matched backend/padding.
combined: both engineering repairs. No objective change is enabled.

Only original four-response PRM microbatches containing an affected input are
re-inferred. All partners and ordering are preserved and old/new inputs share
padding length. Unchanged native microbatches reuse their frozen score cache.
logged->matched_legacy measures backend/padding drift, NOT a repair benefit.
matched_legacy->other variants isolates the engineering changes. Installed
TRL0.23.1 reward tensor algebra is executed without constructing a trainer.
More steps can lower a minimum mechanically; changed ranks are not evidence of
improved mathematical judgments or trained-policy accuracy.

Reproduction, after inspecting existing completion receipts

test_repairs.py
prepare.py --inputs PROJECT --out NEW_AUDIT --metadata FROZEN_UNIT_METADATA
           --original-manifest PRIOR_SOURCE_MANIFEST
score.py --out NEW_AUDIT
analyze.py --out NEW_AUDIT
render.py

No repeat scoring after scoring_complete.json. Explicit --resume is required
to reuse verified complete microbatch receipts after a partial inference run.
Only an idle GPU is used; no running job is stopped or displaced.

Separate proposal for PRM-driven homogeneous groups (NOT ENABLED IN TRAINING)

A_i = (y_i-mean(y))/(sample_std(y)+1e-4)
      +0.2*(m_i-mean(m)), m=minimum native probability or0 for invalid output.

This keeps the ordinary outcome-only GRPO advantage and adds a bounded process
residual without normalizing the residual's tiny variance. For group8 and
m in[0,1], its absolute process term is bounded by0.2*7/8=0.175. In mixed binary
groups, the smallest absolute nonzero outcome advantage is about0.35345, so
this residual cannot reverse the correctness sign at the chosen coefficient.
For homogeneous groups it is small rather than rescaled to order-one size.
This is an engineering safeguard, not a new research contribution. It does not
prove that a PRM's ordering is mathematically correct; a bad proof may still get
a small positive residual. It is measured only as a separate offline proposal.

Four-arm controls and PRM failure plan

Historical A, H4-B, and7B-Bprime remain historical results. Retrospective unit
rescoring cannot undo their past gradients. If unit-corrected training is later
authorized, every arm must use the same fixed verifier, data, actor initial
checkpoint, budget, and seeds. Do not use an old-rule A as the strict matched
baseline for new-rule B/C/D. No new training is authorized by this release.

A: outcome-only GRPO under the common repaired verifier.
B: the common public PRM initialization frozen during RL.
C: that same PRM plus L_fail+L_success+L_step adaptation, then frozen in RL.
D: identical PRM initialization/batches/anchors/optimizer/steps as C, except
   c=0: L_fail+0*L_success+L_step, then frozen in RL.

If Qwen7B is selected, B/C/D must ALL use that7B initialization; the old H4
C/D pilot cannot be compared as an adaptation control for7B-Bprime. Percent
repairs apply to adaptation weak labels as well; CRLF applies to its input view.
No case-specific bonuses, problem-specific masks, gold-based answer selection,
or special treatment only for D is introduced.

For PRM mathematical misses, code is insufficient. Use existing public step
labels as anchors and outcome failures as weak evidence of at least one bad
step, not as a claim that every step is wrong. C/D retain their planned loss
difference. The earlier H4 small adaptation pilot did NOT show a reliable
repair: absolute scores fell without fixing known wrong-answer rankings.
Therefore do not promise that removing success supervision fixes the scorer.
For labeled anchors, preserve their structured step/label mapping rather than
re-splitting a concatenated trace and reassigning labels. Canonicalize line
endings within each labeled step, keep the original step boundaries, and assert
that native scoring positions match the label count. Unlabeled policy rollouts
use the common repaired splitter. These rules must be identical for C and D.

Before any later full RL, compare B/C/D on a disjoint, fixed labeled process
set: valid-process acceptance, first-error detection, and answer-correct
valid-vs-invalid AUROC with paired intervals. For training-signal diagnostics,
report mixed/all-correct/all-wrong groups separately, including invalid-partner
strata. The known20 reviewed B traces are regression diagnostics, not unseen
test evidence or extra PRM training labels. Prefer a public labeled validation
split for choices, freeze it, and keep final ProcessBench evidence untouched.

The bounded-objective proposal, if later authorized, must be applied to ALL
B/C/D together (A has zero process residual). Test it as a shared protocol
factor, not an extra trick given only to D. Keep D-C isolating c=0 vs1.
