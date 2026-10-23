Reward repair v2 — independent, offline-only implementation

Scope
This directory implements the agreed engineering repairs and counterfactual
replay on existing B-6M completions. It does not generate new completions, run an
optimizer, resume B, or start A/B/C/D/S training. No original training source,
checkpoint, dataset, or result is overwritten.

Implemented components
1. v2core.outcome_v2: question/gold hash-bound requested-unit verification.
   A fixed, reviewed manifest identifies 45 explicit percent-output questions
   in the existing 6993-question training split. All others retain the archived
   original verifier. Merely mentioning a rate/discount/% does not change units.
   Bare decimals in percent-output questions are interpreted in the requested
   percent-value unit (20 means 20%, not 0.2%); probability questions retain the
   original dimensionless interpretation. Positive and negative tests cover both.
   This is a scoped repair, not a universal semantic verifier: composite gold,
   unknown units and unsupported percent expressions are not silently rewritten.
   In particular, the composite gold '0.56 = 56%' remains on the legacy path.
   New datasets need their own frozen, reviewed unit metadata before use.

2. v2core.canonical_view/encode_native: a separate PRM-only LF view, native
   double-newline steps, raw-character spans, exact prefix token checks and
   explicit context-length failure. Raw actor output and actor prompt are kept.
   There is no sentence model, new head, code execution or semantic segmentation.

3. admission.admit_question: shared prospective question quarantine with hash
   binding. A separate manifest lists eligible IDs; original data is untouched.
   Quarantine never means changing one answer's reward to zero or resampling a
   convenient partner. Replay retains every original group and separately
   reports the common subset after removing the entire invalid-question group.

4. v2core.audit_advantages: a read-only, same-device/dtype reference for actual
   TRL advantages. It saves evidence before raising, checks finite values and
   group identity, and returns the exact same actual tensor object. Float64 CPU
   results are diagnostics. Pinned FP32 rewards are required; sample std,
   epsilon=1e-4, coefficient=0.2 and min aggregation are unchanged.
   b_audit_patch.patched_b_trainer_class is an opt-in subclass factory, not a
   launcher. It verifies the original B method AST against the archived source,
   replaces only its cross-device assert_close statement, and leaves all other
   statements identical. A stub-only test checks outputs and accounting without
   creating a policy model. The factory is not installed into the original run.

5. trl_kernel.py extracts the reward tensor algebra from the installed TRL
   0.23.1 source. Offline checks execute only those tensor operations, with no
   GRPO trainer construction, generation, optimizer or backward call.

Reproduction order
build_inputs.py -> test_v2.py -> prepare_replay.py -> score_frozen.py ->
replay_offline.py -> admission.py -> replay_failure_guard.py ->
protect_originals.py (after) -> verify_results.py -> make_report.py.
Build/freeze inputs once. Completed replay/scoring scripts refuse to overwrite
their main completion artifacts. score_frozen.py can resume only a verified
partial prefix, with explicit prior cost accounting; completed inference is not
repeated automatically. All PRM scoring uses the pinned original H4 model in
BF16, eval mode, no gradients, the same old/new batch order and padding shape.

Comparison interpretation
logged: original recorded rewards/scores.
logged_labels_replayed_scores: old labels, paired re-inference of old inputs.
legacy_replayed: reproduced legacy outcomes and old-input scores.
units_only: new unit verifier; identical old-input PRM scores.
split_only: normalized CRLF inputs; identical legacy outcomes.
combined: new unit verifier plus normalized CRLF inputs.

logged -> logged_labels_replayed_scores measures backend/batching drift, not a
repair gain. legacy_replayed -> units_only isolates unit handling.
legacy_replayed -> split_only isolates the input change on the same backend.
legacy_replayed -> combined is the matched joint counterfactual. Advantages are
recomputed on all eight original partners; their deltas are not assumed additive.
Unchanged groups reuse original frozen-PRM scores. Every eligible partner in a
prespecified affected/review group is re-inferred. Invalid outputs have no real
PRM score; null is used in analysis instead of misreading the old placeholder 0.

Reward midranks use the actual FP32 rewards consumed by TRL and exact ties;
pairwise-change reporting and advantage sign reporting
use a fixed 1e-6 dead band. Also report |delta advantage|>0.01. A change in a
score/rank is not necessarily an improvement in mathematical correctness.
More step boundaries can lower a min score even without better recognition.

Four-arm compatibility
A: outcome only. B: original frozen PRM. C: symmetric adaptation. D: no positive
outcome weak-supervision term. C/D still differ only in c=1 versus c=0; no new
loss, PRM parameters, reward weights, normalization or masks are introduced.

Diagnostic-only patch: original A/B results can remain comparable; later B
continuation needs a separate approved launcher and recovery receipt.
PRM-input-only change: A can be reused after compatibility checks; B/C/D must
share the new input rule. Training-verifier or training-data changes: all four
arms are affected; old A cannot be presented as a matched baseline to new-rule
B/C/D. Common rescoring cannot change earlier training gradients.

The previously reviewed 50 cases and prior PRM pilot are regression/diagnostic
evidence, not unseen test evidence or adaptation data. No PRM is improved by
this code patch alone. Whether C/D improves mathematical ranking remains an
unresolved model-adaptation question.

Outputs
The execution folder stores frozen inputs, unit changes, paired PRM scores,
per-variant rewards/ranks/advantages, group/case CSVs, installed-TRL provenance,
positive/negative tests, numerical-audit receipts, independent arithmetic QA,
original A/B protection receipts and a short Chinese REPORT.txt.
