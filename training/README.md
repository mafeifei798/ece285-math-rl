# ECE285 outcome-only GRPO baseline

This implements only arm **A**, starting with seed **17**. The user extended the
planned budget to **24M generated completion tokens per run**, with retained
6M, 12M and 24M milestones. The full planned comparison is therefore
4 arms × 3 paired seeds × 24M = **288M**. Only the existing A/17 continuation is
authorized for this launch; other arms and seeds are not automatically queued.

The original 6M run, its configuration, full checkpoint and completed evaluations
remain immutable. `baseline_config.json` records that original contract, not the
new stopping budget. The explicit extension is `operations/continuation_24m.json`.

Remote root: `/srv/research/ece285_math_rl_project`.
SSH: `ssh -p 11032 -i ~/.ssh/research_server_ed25519 researcher@server.example.invalid`.
The key is used by SSH; its contents are not stored in this project.

## Method and fairness

- Qwen2.5-Math-1.5B **base**, BF16, rank-16 LoRA over attention and MLP projections.
- Plain Question/Answer CoT prompt, requesting a final boxed answer.
- One scalar reward: 1 for a complete correct final answer, otherwise 0.
  No PRM, process reward, format bonus, or length bonus in A.
- Pinned TRL 0.23.1: `loss_type=grpo`, `scale_rewards=group`, epsilon 0.2,
  KL beta 0.04, one update per rollout batch, 8 responses per question.
- A generation batch has 32 completions (4 questions); microbatch 1 and
  accumulation 32. Completion cap 2048; temperature/top-p 1.0; no top-k filtering.
- LoRA learning rate 5e-6, constant schedule. These are shared starting settings,
  not tuned using test results. Subsequent B/C/D should reuse the frozen contract.
- The custom trainer does not replace TRL's objective. It audits computed
  advantages, logs completion-level outcomes, and implements token-budget stopping.
- All generated completion tokens, including EOS, wrong answers, and truncations,
  count. Stop at the first completed optimizer update reaching the declared budget; record overshoot
  (at most 32 × 2048 = 65,536 tokens). Prompts, evaluation, compilation and PRM
  work are separate cost counters. TRL's `num_tokens` includes prompts and is
  therefore **not** the project completion-token budget.
- TRL's colocated vLLM engine uses seed 0. The training seed controls adapter
  initialization, data shuffling and torch RNG. This limitation is recorded in
  every run rather than falsely claiming all RNGs vary across seeds.

## Data and evaluation

Original MATH training split: 7500 questions before duplicate and overlap checks.
A deterministic 500-question validation split is held out using split seed 285.
Known evaluation and ProcessBench question overlaps are excluded from training.
Data/model revisions, normalized-question hashes and exact JSONL hashes are saved.
PRM800K training question overlap is audited for the later PRM-based comparisons;
unknown checkpoint pretraining exposure remains a limitation.

The first prepared manifest contains 6,993 training questions and 500 validation
questions. From the 7,500 original records, four lack a boxed gold answer, two
are duplicate questions with mathematically equivalent answers, and one overlaps
an evaluation/diagnostic question. Maximum observed prompt length is 1,674 tokens.
MATH500 and GSM8K have zero exact normalized question overlap with the public
12,000-question PRM800K training inventory. ProcessBench has 847 overlapping
unique questions and is therefore an exposure-aware diagnostic, not evidence
of unseen-question generalization. The manifest records all exclusions and hashes.

The initial policy and final adapter use identical greedy single-generation
evaluation on MATH500 and GSM8K (1319 test questions), with the same verifier and
2048-token cap. No voting, reward model or reranking is used in evaluation.
No test-based checkpoint or hyperparameter selection is performed.

## Execution order

1. Independent project `.venv` and pinned public assets.
2. `prepare_data.py` and `pytest training/test_reward.py`.
3. Three-update pilot from the initial model in its own output directory;
   `check_pilot.py` checks real finite updates, a mixed-reward group, token
   accounting and checkpoint saving. Pilot weights are not used for formal A.
   A fresh C compiler check and real vLLM LoRA generation using this saved pilot
   adapter must also pass before formal training. This generates at most 16
   tokens from two validation prompts and never selects by answer quality.
4. Initial full policy evaluation on GPU 1. It uses immutable pre-RL weights and
   can continue while the pilot or formal A run uses GPU 0; test scores do not
   determine training settings.
5. `run_baseline.py` on GPU 0: fresh A/seed17, 6M tokens, final saved-adapter evaluation.

Step 5 is the completed historical run. **Do not relaunch it for the extension.**
Use `operations/run_continuation.py`, which resumes `runs/A_seed17/checkpoint-280`
into `runs/A_seed17_24M`. It inherits every saved GRPO argument except output and
logging paths, verifies exact adapter/optimizer/scheduler and Python/NumPy/torch
RNG restoration, and checks that the first new batch follows the 1,120 already
trained questions. Cumulative accounting starts at **6,020,163**, not zero.

The continuation saves full 12M and 24M checkpoints under its `milestones/`
directory, outside the rolling-checkpoint rotation. Their actual token counts
and hashes appear in `milestones.json`. After training, the controller evaluates
both checkpoints with the unchanged greedy evaluator; 6M predictions are reused
and hash-protected. It uses only an idle GPU and does not disturb unrelated jobs.
Training progress and actual restore checks are `progress.json` and
`resume_audit.json`; controller state is `controller.json`.

The original checkpoint does not contain vLLM's separate sampling RNG stream.
That seed-0 engine restarts, while model/optimizer/scheduler, data position and
Python/NumPy/torch RNG are restored. Do not claim bitwise identity to an
uninterrupted run. Record and match this 6M continuation boundary in subsequent
controlled comparisons. No extra pilot or fresh baseline training is launched.

Use `CUDA_VISIBLE_DEVICES`, `HF_HOME`, `HF_ENDPOINT`, `VLLM_CACHE_ROOT`,
`TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` explicitly. No global packages,
other project environments or other users' GPU jobs are changed.

Receipts and logs distinguish setup, pilot, formal training and evaluation.
Inspect live processes and locks before re-launching. Interrupted runs require
explicit checkpoint resume and matching code/config/data provenance; do not
overwrite output directories. A 48-hour per-run wall cap preserves a checkpoint
and marks the run incomplete instead of running indefinitely.

## Final evaluation recovery

The first A/seed17 training completed at 6,020,163 completion tokens and 280
updates. Its initial final-evaluation attempt stopped during vLLM LoRA kernel
compilation because the server lacked `Python.h`; it produced no predictions.
Matching CPython 3.11.11 headers were generated from the official source archive
under `cache/python_dev_headers`, without installing or replacing Python.
`operations/recover_evaluation.py` supplies those directories through
`C_INCLUDE_PATH` and reruns only the unchanged final evaluation after checking
model/config/source hashes and process locks. Original failure receipts and logs
are retained. Recovery status is in `runs/A_seed17/evaluation_recovery.json`;
its log is `logs/A_seed17_evaluation_recovery.log`.
The normal launcher now propagates the project header paths and runs
`operations/lora_preflight.py` after the pilot. Its fresh compile check cannot
be bypassed by an already-warm Triton cache. These checks do not change training
or evaluation prompts, rewards, model weights, or decoding settings.

## Upstream references

- [Pinned TRL GRPO implementation](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py)
- [Qwen model](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
- [Math-Verify](https://github.com/huggingface/Math-Verify)
- [Original MATH mirror](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
- [PRM800K exposure inventory](https://github.com/openai/prm800k/tree/main/prm800k/math_splits)
