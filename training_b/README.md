# B: frozen public process reward, seed 17

User authorization on 2026-10-21: “咱们能顺便把b也做了吗”. This authorizes B/17, its correctness pilot, 24M completion-token training, and retained 6M/12M/24M evaluations. No C/D or additional seeds are queued.

The actor starts from the **original base model**, not A's trained adapter. The saved A GRPOConfig is reused with only output paths changed. Pilot max_steps is an explicit plumbing-only exception. Outcome verification, group size 8, normalization, LoRA rank 16, generation cap 2048, dataset manifest, learning rate and greedy evaluation stay fixed.

Complete parseable responses receive `outcome + 0.2 * min(native_step_probabilities)`; invalid/truncated responses receive zero. The original H4 PRM is frozen and evaluated at its native double-newline step boundaries. The scorer checks that each independently tokenized prefix is an exact token prefix of the full input. It stops on overlength/invalid PRM scoring instead of silently truncating. Step-count/delimiter diagnostics and all-correct/all-wrong/mixed outcome-group diagnostics are saved. PRM time/input tokens are additional costs, separately reported.

Files under `training/` and `operations/` are imported read-only. This avoids invalidating the currently running A controller's source hashes. No environment or library upgrade is required.

The one-shot B controller waits for A's shared pipeline lock, including A's milestone evaluations. It uses only a GPU with no active compute process; it never stops the RNA workload. The wait limit is 48 hours. Gates: CPU unit tests, real native PRM scoring/repeat checks, two optimizer steps, exact checkpoint-resume plus one step, and real LoRA evaluation/compiler preflight. Failure stops the pipeline and retains evidence. Pilot work is extra, not included in the 24M training count.

Formal schedule: fresh actor to 6M; preserve/evaluate checkpoint; resume the full state to 24M with a retained 12M checkpoint; then evaluate 12M and 24M. This matches A's 6M restart boundary and its separate vLLM RNG restart limitation. Full checkpoints are outside rolling rotation. Tests are never used to select the endpoint.

Staging amendment: B/17 is required under both replication branches and is now authorized before C/D preparation. The C/D probe gate has **not** run. Its thresholds and later seed choice must not be changed using B accuracy. C/D and extra seeds require separate launches.

Status: `receipts/B_pipeline.json`; detailed progress under `runs/B_seed17_6M/` then `runs/B_seed17_24M/`. Model asset checksum receipt: `receipts/B_prm_assets.json`. Hardware/implementation gate: `receipts/B_pilot_gate.json`. Logs: `logs/B_seed17_*.log`.
