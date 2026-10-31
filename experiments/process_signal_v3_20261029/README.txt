Process-signal revision V3, 2026-10-29

Purpose: compare HOW a fixed discriminative process scorer enters small-model
mathematical GRPO. Do not confuse this estimator study with a new PRM model,
a full PAPO reproduction, or a proof that answer-correct derivations are valid.

Current running experiments/c_bounded_pilot_20261029_v2 is immutable. No active
controller imports this folder. No new GPU training is dispatched by V3.

Changes inspired by inspected official repositories:
1. PAPO: add an explicit correct-subset-standardized process estimator, preserving
   the original outcome component, fixed C scorer, lambda.2 and common epsilon.
   The official native function is included as a CPU reference, with source/license
   notice; parity is checked at its native epsilon1e-6 and lambda.2/1.0.
2. TRL: retain ACTUAL parent advantage tensors for A and C-mix. Replace only the
   estimator in PAPO/bounded before buffer/shuffle. Token/mask fingerprints and
   row tags are verified at the actual pinned loss entry; never re-standardize.
3. PURE: report outcome strata, signal magnitudes, token/PRM costs, truncations,
   step counts, short wrong outputs and screened refusal candidates. These
   diagnostics never modify rewards. None certifies a mathematical process error.
4. EasyR1/SimpleRL: separate data/scorer/reward/estimator/metrics; use one frozen
   machine-readable protocol. Do not copy their default format/length rewards.
5. LCA: keep the completed C adapter frozen. No D retention run, online PRM
   learning, extra labels, new judge, longer responses or new model is introduced.

File map:
  protocol.json        four-arm FUTURE controlled design + current budget boundary
  signals.py           pure outcome/mixed/PAPO/bounded sequence estimators
  diagnostics.py       metrics separated from the learning signal
  trainer_adapter.py   pinned TRL audit mixin; formal train/optimizer entry locked
  papo_reference.py    verbatim official function with isolated imports/registry
  validate.py          CPU official-function/legacy/buffer/loss/edge/mutation checks
  replay_rollouts.py   existing scored C rollout comparison; no model calls

Run CPU acceptance from the project root:
  python3 experiments/process_signal_v3_20261029/validate.py --output <NEW_OUTPUT_DIR>

Run scored-rollout replay:
  python3 experiments/process_signal_v3_20261029/replay_rollouts.py \
    --input <C_mix_or_C_bounded_run>/rollouts.jsonl --output <NEW_REPORT.json>
Requires the adjacent run_config.json with the exact C adapter provenance.
Never feed outcome-only A logs and invent missing C scores as zeros.

Acceptance limits:
  The CPU actor tokens/masks/logits are algebra fixtures. Frozen C probabilities,
  outcomes and response texts are real historical data. No GPU/model generation,
  actual PRM call, optimizer or checkpoint update is part of this acceptance.
  The real-model audit and checkpoint acceptance of the currently running V2 do
  not automatically transfer to V3's added PAPO path. Train entries remain locked.

Fair-comparison boundaries:
  Existing A/C-mix/C-bounded each1M seed17 continue untouched. New PAPO training
  has zero allocated tokens and is not queued. Future PAPO-on-C would be an
  estimator-level comparison, not the paper's 8GPU+generative-grader recipe.
  Same lambda does not imply equal gradient norm or tuning to each method's best.
  Correct-subset gating and avoiding std scaling have existing antecedents;
  mathematical accuracy/independent process validity are the real outcomes.
  Single-seed no-significant-difference is not evidence of equivalence.

References and immutable source commits:
  PAPO: tanzelin430/PAPO @7d260f0a7a6e23d289e32bbc96f80408dddfa1d5
  PURE: CJReinforce/PURE @b174e508a4469663543cfcb366be78cf199d20bf (verl)
  TRL: huggingface/trl @4529a1c8b1813480a85b02c9c8e7f75a29085d65 (v0.23.1)
  LCA: JiaTianyu20031016/MIL @d8dc4593a95f8ed142d5c9166140e8843cb208fc
  Full11-project review: execution/github_reference_audit_20261029/
