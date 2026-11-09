Prefix protection: independent implementation and validation
==========================================================

Scope
-----
This directory does not alter existing A/B/C/D experiments. It implements a
conservative first-error-prefix experiment on top of the frozen C-bounded
policy advantage. Importing a module does not dispatch a job. The supplied
validation programs perform no optimizer step; formal training is separate.

Three matched variants
----------------------
bounded: original C-bounded advantage, broadcast to original response tokens.
uniform: for an eligible wrong response, scale its negative policy advantage
         by unprotected_tokens / total_response_tokens at every token.
prefix:  zero the negative policy advantage before the first low-scoring step;
         retain the original negative advantage on the suffix, including EOS.

Only complete, PRM-scored wrong responses in mixed-outcome groups are eligible.
The first score below 0.8 is used. First-step errors, no low score, invalid or
truncated responses, and failed byte/token alignment retain original signals.
Correct responses are unchanged. All-wrong groups retain zero policy advantage;
their KL term is still present. No new positive reward is given to wrong answers.
The PRM-estimated prefix is not asserted to be mathematically correct.

Controls and caveats
-------------------
All variants retain the original response-length denominator, EOS/padding mask,
clipping, KL, verifier, prompts, frozen C PRM, and outcome normalization. There
is no normalization after the C-bounded scalar is constructed. Uniform and
prefix match the sum of absolute advantage coefficients per response, not the
gradient norm. Prefix must outperform uniform before attributing a benefit to
localizing errors instead of simply weakening negative updates.

Token alignment
---------------
signals.py reconstructs bytes from the actual emitted Qwen byte-BPE token IDs.
It validates complete decode and UTF-8 reconstruction, maps the unchanged CRLF
step splitter's raw character boundary, and retains any boundary-crossing token
in the penalized suffix. It does not locate boundaries using re-encoded output.
runtime_mixin.py fingerprints each response/token/mask/advantage record and
checks the fingerprint at the actual loss call after microbatch shuffling.

Loss provenance
---------------
token_loss.py is derived from TRL 0.23.1 GRPOTrainer._compute_loss. Four scalar
advantage unsqueezes are removed; the rest of the objective is retained. CPU
tests compare scalar and broadcast-matrix losses and log-probability gradients,
including clipping. Extra normalization injected before the loss is rejected.

Sampling provenance
-------------------
The installed TRL 0.23.1 file hash is
f217183c84be9b38b23454920eb1b914726aa3c4b9dd4c076d455bdaa67bf1b8.
Its legacy synchronization merges/unmerges LoRA into the BF16 actor base. Actual
GPU diagnostics found a non-identical base hash after this round trip and
differences between native and merged sampled models. This is not evidence of
a particular benchmark accuracy effect.

native_grpo_trainer.py is an isolated copy of the pinned Apache-licensed source:
enable native vLLM LoRA, disable prefix caching, use eager execution, and pass an
explicit LoRARequest. Eager execution passed the fixed-probe repeatability check;
its measured generation throughput is substantially slower than the CUDA-graph
diagnostic. This is a runtime control, not a new reward or loss.
native_sync.py overrides synchronization, saving a per-step adapter snapshot and
changing its ID without merging actor weights. It supports only single-GPU,
colocated, non-FSDP/non-DeepSpeed PEFT. Existing installed libraries are untouched.
Any subsequent three-arm pilot must use this SAME sampling path for every arm;
historical merged-path scores are not its matched algorithm control.

Validation programs
-------------------
cpu_validate.py: existing 9,216-response replay and scalar/matrix gradient checks.
test_plumbing.py: generation-hook integration, shuffled batches, rejection of
                  intentionally renormalized advantages.
no_update.py / merged_probe.py / validation_controller.py: fixed HF/native/merged
                  model comparisons and real backward, no optimization.
native_trainer_no_update.py: real TRL generation, frozen 7B PRM, all three token
                  losses, and same-weight adapter hot-reload repeatability.

Evidence
--------
Remote: /path/to/ece285_math_rl_project/audits/prefix_protection_20261107
Local: execution/prefix_protection_20261107
Failed diagnostic attempts are retained. Check result.json and the final report,
not merely a dispatch receipt, before treating an acceptance gate as passed.

No accuracy improvement is claimed by these implementation checks. A proposed
next pilot branches the same seed17 C-bounded checkpoint-288 into three variants,
48 additional updates each, matched question order, validation-only screening.
This is a continuation experiment, not a new from-scratch 6M comparison.
