Corrected-only Qwen7B-PRM B baseline

Selection: this is the current baseline implementation. The earlier bounded
advantage version remains archived as an unselected method experiment under
experiments/b_fix_integration_20261021. No original A/B artifact is overwritten.

Fixes included, shared consistently across applicable prospective arms:
1. Frozen question/gold-bound percentage-unit answer verification.
2. CRLF-to-LF canonical view for PRM input only; actor text stays unchanged.
3. unit_exact/unit_symbolic remain eligible for ordinary process scoring.

Algorithm unchanged:
  R_i = y_i + 0.2 * min_t p_i,t for complete valid outputs; otherwise 0.
  A_i = (R_i - mean_group(R)) / (sample_std_group(R) + 1e-4).
  Group8, process coefficient0.2, original pinned TRL0.23.1, original clipping,
  KL, masks, optimizer/sampling settings, frozen Qwen2.5-Math-PRM-7B.
  No decomposed advantages, process-signal caps, group filtering, new judge,
  threshold change, homogeneous-group suppression, or PRM adaptation.

trainer.py only audits the actual TRL tensor and adds integer trace metadata.
It does not assign output['advantages'] or edit the already-recorded TRL log.
The trace metadata verifies no numerical change before/inside loss. The source
and runtime pin checks are diagnostics, not mathematical objective changes.

rewards.py is an exact copy of the shared v3 reward callback. replay.py uses
all9056 real saved answers and cached native PRM probabilities, recomputes
outcome labels with the actual verifier, and executes pinned TRL preparation,
shuffle/split/buffer/loss method bodies. Unsaved token identity and model log
probabilities are explicit numeric fixtures; no actual model is loaded.

build_training_version.py constructs the new isolated version only on explicit
future invocation. It has no launcher/CLI, train call, or queue. No constructor
with actor/PRM weights was executed in this task. No training is authorized.

Remaining PRM misranking and homogeneous-group variance amplification are
retained as baseline behavior and reported, not silently patched as code bugs.
Historical A/B remain historical. A prospective fair comparison must share the
unit-verifier fixes; do not treat rescored historical outputs as a retrained run.
