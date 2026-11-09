Bprime: frozen Qwen2.5-Math-PRM-7B, seed17, 6M completion-token pilot

Actor: original Qwen2.5-Math-1.5B base + rank16 LoRA. The 7B model only scores; it is frozen.
Changed experimental factor: H4 PRM -> official pinned Qwen7B PRM/native encoding.
Reward and GRPO: original outcome + 0.2 * minimum native-step probability; invalid/truncated=0.
Original dataset, outcome verifier, raw double-LF split, group8, generation batch32,
max completion2048, saved GRPOConfig and initial seed17 are retained.
Only numerical checking uses the already-tested same-device audit; actual TRL advantages are untouched.
Semantic v2 repairs (percent verifier, CRLF normalization, data quarantine) are OFF in this matched comparison.

Remote project: /path/to/ece285_math_rl_project
Formal run: runs/B_Qwen7B_seed17_6M
Smoke run: runs/pilot_B_Qwen7B_seed17_6M (2 optimizer steps; extra cost recorded)
Controller: experiments/b_qwen7b_6m_20261021/launch.py
Original source/results are hashed/inventoried and never overwritten.
Exclusive controller/run locks and existing-attempt checks prevent duplicate launch.
Failures stop the pipeline; there is no automatic retry/retraining or extension to24M.

Pipeline: 36 regression checks -> source-preservation snapshot -> 2-step smoke run
-> finite-gradient/frozen-PRM/initial-sampling/advantage/accounting gate -> independent
6M training from original base -> preserved milestone checkpoint -> unchanged greedy
MATH500 (500) and GSM8K (1319) evaluation. No PRM reranking at evaluation.

Interpretation: compare to the original A6M and H4 B6M. This is a scorer replacement
pilot, not the novel C/D adaptation intervention. Capacity and native PRM format both
change. A good offline score is not evidence of a downstream accuracy improvement.
One seed cannot establish a small gain. Do not start C/D from this controller.

Read status/progress/controller JSON and stage logs before any later intervention.
