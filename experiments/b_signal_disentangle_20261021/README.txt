Fixed-rollout separation of PRM ranking and advantage amplification

analyze.py reads the completed v3 repaired7B-B replay and prior fixed scores.
It preserves all9056 responses, invalid partners, and original groups. It does
not load or call a model, generate an answer, train a PRM, or run an optimizer.

New diagnosis: review all eight partners rather than inferring rank errors from
one bad proof with positive advantage. Two groups are uniform draws from the
remaining31 all-correct groups (seed286); two are known diagnostic groups.
The32 mathematical reviews are by one Codex assistant, some old scores known.
They are not independent human ground truth or population ranking estimates.
Explicit erroneous claims, recovered errors, proof gaps and wording ambiguity
are separated. Across-question sparse-review AUROC is not same-question rank
accuracy. All-error groups cannot establish clean-vs-error rank inversions.

Original saved actual advantages are retained. A local FP32 replay differs in
last digits across CPU platforms; both are compared to independent FP64. The
maximum observed difference is under1e-5 with no sign change for |A|>1e-3.
No trainer's numerical guard or actual advantage tensor is modified.

bounded_signal.py implements the proposed engineering guard:
A = outcome_GRPO_advantage +0.2*centered_process_minimum.
The process term is not standardized again. Existing validity rules apply:
invalid responses have zero placeholders, not a true zero PRM judgment.
Return values are detached, finite, group-preserving sequence advantages.
The integration contract is to feed these to the existing clipped policy loss
without a second reward/advantage normalization, preserving clipping and KL.
No trainer integration or launch is included. This is not claimed as novel.

verify.py checks every binary outcome/extreme-probability combination for an
eight-response group (65536 groups). It independently verifies the0.175 bound,
mixed-outcome sign protection, exact recovery of outcome-GRPO at coefficient0,
invalid-input rejection, arithmetic labels, and actual fixed-response replay.
render.py creates a readable report and provenance receipts.

The bounded rule leaves PRM within-outcome rankings unchanged. It can reduce a
mistaken signal's amplitude but cannot make a poor scorer rank correctly. A
stronger scorer and a safer advantage construction address different defects.
Offline checks cannot prove that the repaired policy will beat trained A or
identify causal shares of the observed B-minus-A accuracy difference.

Alternative scorer materials were read from primary sources only. ActPRM-X is
a possible offline candidate based on the current Qwen7B PRM. Its author
benchmark is not a local RL result. No new weights were downloaded or executed.

Reproduction: analyze.py (once) -> verify.py -> render.py.
Existing original and v3 outputs are never overwritten. All new artifacts are
under execution/b_signal_disentangle_20261021; code lives in this directory.
