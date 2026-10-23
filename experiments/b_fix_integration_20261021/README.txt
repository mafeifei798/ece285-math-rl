B-fix independent training integration, 2026-10-21

Baseline interpretation
  Original B is the mixed-reward/group-standardization baseline and remains
  untouched. B-fix is a named revised baseline, not an implementation-equivalent
  reproduction of B. Any future B-fix/C/D comparison must share this exact
  advantage rule; D-C can then isolate PRM adaptation again. Compare the final
  policy against outcome-only A too. This engineering guard is not claimed novel.

Exact rule
  A_i = (y_i - mean_group(y))/(sample_std_group(y) + 1e-4)
        + 0.2 * (m_i - mean_group(m)).
  y uses shared v3 percent verification; m is frozen Qwen7B native minimum step
  probability with v3 PRM-only CRLF normalization. Invalid partners remain in
  their original groups with zero placeholders. A zero placeholder is not a PRM
  correctness judgment. The process component is bounded by 0.175 for group8;
  PRM ranking errors are not corrected by this change.

Integration
  rewards.py: same callable used by live trainer and replay, including actual
    v3 verifier, canonical input view, validity, cached/native step alignment.
  trainer.py: independent GRPOTrainer subclass, never changes A/B. Replace the
    historical tensor once after generation and before shuffle/split/buffering.
    Replace the corresponding advantage logs. Carry integer provenance tags
    and assert bitwise equality to expected FP32 values immediately before loss.
    Keep original TRL ratio clipping, KL, token masks and loss reduction.
  build_training_version.py: lazy factory for the live model/trainer. There is
    no CLI/launcher, queue, or call to train. This factory has not been executed
    with actor/PRM weights; a future explicitly authorized pilot remains needed.
  vendor/: source copies from the actual previously used TRL0.23.1 environment,
    not a different upstream release. Full installed trainer SHA is checked.

No-update replay
  python replay.py --out <new independent output directory>
  Run in the existing remote Python environment with CUDA_VISIBLE_DEVICES=''
  and HF_HUB_OFFLINE=1. Requires the existing frozen v3 evidence, no downloads.
  Uses all9056 stored completions, original groups, cached PRM probabilities,
  actual v3 answer verification, and real saved lengths/EOS. All native steps
  are aligned. Executes exact TRL preparation/shuffle/buffer/compute_loss bodies,
  stripping profiling decorators only. Current/old/reference log probabilities
  and token identity tensors are explicit synthetic algebra fixtures because
  those were not stored; no model forward is performed. Real masks/lengths,
  clipping at three probability-ratio settings, and KL are exercised.
  It checks that loss inputs equal the builder outputs exactly and deliberately
  injects another normalization/rescaling/misalignment to verify rejection.

Evidence boundary
  A successful CPU replay proves this tensor integration on fixed evidence, not
  successful live GPU training or improved MATH/GSM accuracy. No optimizer,
  backward, actor sampling, PRM model inference or parameter update is run.
  Original A/B/proposal code and historical results are retained. No C/D run.
