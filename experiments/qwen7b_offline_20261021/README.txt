Qwen2.5-Math-PRM-7B offline scoring audit — 2026-10-21

Authorization: frozen offline scoring of existing B rollouts, comparison with old B,
GPU memory and throughput measurement. No training or new generation authorized.
Original training/, training_b/, operations/, protocol and A/B outputs are untouched.

Input: original B first 6M, 9120 responses / 1140 original eight-response groups,
SHA256 2d5309f259bf454d9e446dea750a9bcbea6e3c1349ae483c9862e0beac540228.
7472 complete parseable responses are scored. All eight partners remain in reward
and advantage replay, including invalid completions receiving zero reward.

New PRM: Qwen/Qwen2.5-Math-PRM-7B, revision
0610740060112df12585d00a1c5f4624d2f59051. All assets are verified against metadata
retrieved from the official HF endpoint. Secondary transport cannot change assets.
Old PRM: original H4 checkpoint, unchanged and freshly scored on the same A100.

Native encodings: H4 question + double-LF joined steps + trailing double LF,
class-1 probability at each native prefix endpoint. Qwen uses its official chat
template, system instruction, and <extra_0> after every step; class-1 at markers.
This compares native scorer pipelines, not a pure identical-token weight swap.
No silent truncation. Both context limits are 4096. Original actor text is preserved.

Primary model-only comparison: unchanged raw double-LF segmentation, original
outcome labels, R=y+0.2*min(p). Secondary: both scorers share the previously fixed
CRLF normalization and typed-percent outcome repairs. The contradictory question
is excluded as a whole group in summaries, never edited or selectively zeroed.

Actual installed TRL 0.23.1 tensor algebra is extracted without constructing a
Trainer. FP32 sample standard deviation and epsilon 1e-4; all group partners kept.
Counterfactual outcome-only variants use the same B trajectories and are NOT
measured A-policy results. Advantage/rank changes alone do not establish improvement.

Error evaluation: fixed previous 30 uniform + 20 targeted reviewed responses.
These are Codex single-reviewer labels, not independent human gold. Targeted cases
were selected using OLD H4 scores. Their aggregate is a regression checklist,
not an unbiased model comparison. Random counts also have very small denominators.
Only certain valid prefixes count as false step penalties; local correct steps
after prior errors are excluded. First-error location excludes prior uncertain steps.
Threshold 0.5 fixed; threshold-free solution AUROC and paired bootstrap also saved.
No threshold, reward coefficient, model checkpoint or data tuning.

Costs: BF16 weights, FP32 softmax, SDPA, microbatch 8, length buckets. Production
forward time excludes model loading, tokenizer preprocessing and warm-up; wall
time and controls are separate. PyTorch allocated/reserved peaks are reported
separately. These are scorer-only costs, not a measurement of complete RL training.

Operational notes: HF CDN timeouts caused a transport-only fallback to ModelScope,
with exactly the same official hashes. The new model folder uses a valid Python
identifier because dotted local directory names break Transformers dynamic imports.
H4 was scored with the archived score_h4_source.py.txt before adding Qwen strict
loading-info validation and additional resource metadata; production numerical
forward, dtype, softmax, sample ordering and batch settings are unchanged.

Reproduce on the authorized server using the project venv, distinct audit directory:
python prepare.py ROOT OUT
python test_core.py OUT/tests_before_inference.json
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python score.py ROOT OUT h4
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python score.py ROOT OUT qwen7b
CUDA_VISIBLE_DEVICES=0 python analyze.py ROOT OUT
python protect_originals.py ROOT OUT/originals_after.json OUT/originals_before.json
python verify.py ROOT OUT
python report.py ROOT OUT

Scripts refuse to overwrite complete inference or silently resume partial runs.
All existing outputs and failed preparation attempts are retained.
