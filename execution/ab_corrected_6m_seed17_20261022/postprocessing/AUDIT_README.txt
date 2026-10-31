Independent completed-training audit, corrected A/B seed17 6M

Run only after the selected formal arm has status=completed:
  python -B audit_completed_training.py --root /srv/research/ece285_math_rl_project --arm A
  python -B audit_completed_training.py --root /srv/research/ece285_math_rl_project --arm B --output /a/new/output/path.json

No model, torch, safetensors, numpy or project modules are imported. The verifier reads project files only.
Without --output it writes stdout only. With --output it uses exclusive creation, never overwrites,
and does not create parent directories. Exit0 means all implemented checks passed; exit1 means an
audit/schema/file error, reported as JSON. An existing output path is an explicit error.

Checks:
- pinned source/runtime hashes, seed/budget/reward/GRPO arguments and dataset/actor receipts;
- formal completed status and stop at the first whole update reaching6M;
- every32 rows form an update, each consecutive8 rows share question/gold;
- recomputed completion counts/tokens, outcomes/statuses, group types, PRM counts/input/padded tokens;
- B PRM padding cost reconstructed in actual selected-row microbatches of4;
- A outcome-only and B original y+0.2*min(PRM) reward; invalid-output handling;
- canonical CRLF step counts, raw hash and corrected eligibility;
- stored advantage versus independently computed sample-standardized original GRPO advantage;
- accounting reconciled with status/progress/last batch/final checkpoint;
- final live loss coverage and original-advantage/frozen-PRM receipts;
- all final checkpoint hashes and checkpoint trainer step;
- safetensors payloads parsed with standard library, comparing final_adapter and checkpoint
  tensors independently of harmless safetensors-header metadata/serialization differences.

Limits:
- Counts come from saved token counts, not retokenization. Runtime checks separately compare
  completion masks against these counts.
- Outcome correctness is NOT rejudged here. Evaluation needs a separate frozen-v3 verifier replay.
- Advantage comparison uses FP64 arithmetic on FP32-rounded rewards and a stated FP32 rounding
  allowance, largest in nearly tied groups. Exact tensor preservation is already guarded live.
- Frozen PRM, actor-updated and loss-coverage facts are checked against existing live receipts;
  this offline audit cannot independently reexecute their GPU checks.
- PRM elapsed time is receipt-consistent, not remeasured. It includes encoding, padding,
  transfer, inference and score extraction. PyTorch allocator memory is not whole-process GPU memory.
- Final adapter linkage is tensor exact; source hashes prove receipt consistency, not external trust.
- Optional group_diagnostics* files are inventoried, while group metrics are recomputed directly
  from rollouts. Their arbitrary external schema is not trusted to define results.
- Comparing A and B remains separate: shared configuration, initialization and common question
  sequence should match; different update counts are allowed under the fixed output-token budget.
- No old run, new run, training source, reward/loss, or controller is modified by this script.

Synthetic tests:
  python -B test_audit_completed_training.py
Eight tests independently specify four group types and analytic expected advantages. Full fixtures
have94 updates,3,008 completions,6,016,000 output tokens and16,000 token overshoot. Corrupt reward,
token count, advantage, grouping/update order, final adapter and attempted overwrite are rejected.
Fixtures live temporarily inside this postprocessing folder and are deleted on test exit.
Synthetic optimizer/RNG bytes are hashed, never unpickled or executed.
