Authorized paired corrected-only A/B, seed17, 6M generated tokens per arm.

Shared: Qwen2.5-Math-1.5B base, identical seed17 LoRA initialization, fixed data
order and original saved GRPOConfig. Only output paths differ; pilots also use
max_steps=2. Both formal runs start from base, not old policy checkpoints.
A: shared v3 outcome reward only; no PRM loaded.
B: same outcome verifier +0.2*minimum frozen Qwen7B PRM step probability,
CRLF canonical PRM view and corrected unit-status eligibility.
Original group standardization, clipping, KL, optimizer and generation unchanged.

Stages: read-only preflight -> real A/B two-update pilots -> real eight-token
LoRA inference check -> gate -> A6M -> B6M -> greedy MATH500/GSM8K for both ->
paired question bootstrap comparison. Pilot tokens are extra and separately
reported. Formal runs reset initialization. Stop after first full update at or
above6M and report overshoot. PRM inference cost is extra, not actor token count.

launch_pair.py acquires shared pipeline and GPU1 cooperative locks, checks actual
GPU occupancy, preserves old A/B source/result hashes and large-file inventories,
and fails without automatic retraining on a stage error. Inspect evidence and
checkpoint before any recovery. GPU0/RNA jobs are untouched.

Old bounded-advantage and corrected-only offline releases remain immutable.
This new explicitly authorized release is independent; no C/D or new method.
