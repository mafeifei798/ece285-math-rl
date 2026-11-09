Bounded 7B C/D PRM adaptation — 2026-10-28

User authorization: 512 initial-policy questions x4, 2048 shared public step
anchors, matched C/D settings, rank and advantage checks, preserve A/B, no RL.
User additionally allowed sharing physical GPU1 after live resource inspection.

C = Lfail + Lsuccess + Lstep; D = Lfail + 0*Lsuccess + Lstep.
Each component has a separate mean. There is no division by the count of active
components. Both arms execute the same success forwards/backwards (zero weight
in D). Native Qwen 7B classifier stays frozen; only rank8 LoRA is updated.

Data-dependent optimizer count is predeclared as max(512,ceil(fail/2),ceil(success/2)).
Every eligible weak trajectory is covered at least once, with smaller classes
cycled. All 2048 anchors are used; all inputs/schedules are shared by both arms.
No outcome-based resampling or test-based checkpoint selection is allowed.

Primary probe: 64 complete original corrected-B groups selected by hash within
outcome strata (16 all-correct,16 all-wrong,32 mixed), not by PRM scores.
All probe/ProcessBench/validation/test questions are excluded from new PRM
buffer and anchors by normalized exact question text. Three previously seen
old error groups, if available, are disclosed as illustrations only.
192 fixed ProcessBench solutions are diagnostics, not the full benchmark.
Noisy final-answer labels remain noisy, even after parser repairs.

All server work and caches stay below /path/to/ece285_math_rl_project.
Run root: runs/CD_PRM7B_pair_seed17_20261028.
Controller is finite, locks against duplicates, saves PRM checkpoints and fails
closed. It never launches actor training, other seeds, or automatic retries.
Original project source files and corrected A/B artifacts are hash-guarded.
