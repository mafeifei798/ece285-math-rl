Independent C bounded-process-advantage integration, 2026-10-29

Authorized scope: real initial1.5B actor + existing frozen7B C PRM, actual pinned
TRL generation/buffer/loss, forward/backward only. No optimizer, formal RL,
D retention adaptation, extra seed, checkpoint update or scheduled automation.

A = normalized binary outcome advantage +0.2*y*(m-mean among correct responses).
The combined tensor is NOT normalized again. TRL's historical mixed-reward
advantage is checked, then discarded before shuffle/split/buffering. Its scalar
mixed reward remains callback compatibility/diagnostic data, not the optimized
advantage. Row tags and token/mask fingerprints enforce exact loss input.

C is the completed512-update PRM adapter with SHA2564917be38966c8438f58e283eb06c7fa818fed2bb13b726152078c19af013c356.
All actor/C parameters are byte-hashed before and after. Original A/B source and
results plus C/D final adapters/receipts and frozen score files are protected.

CPU replay: all67 saved groups536 responses, pinned actual TRL method bodies;
actor logits/tokens are synthetic fixtures. Live audit: real vLLM32 new samples,
original group8/temp1/cap2048/prompt/verifier; raw generated IDs retained. Saved
text phase:40 real saved responses/real C scoring/real actor forward+backward;
actor text is explicitly retokenized and NOT claimed exact historical tokens.

All new server files are under /srv/research/ece285_math_rl_project.
Physical GPU1 sharing was authorized. Audit-only vLLM cache fraction0.15 instead
of baseline0.3, for headroom with existing jobs. Do not compare this one-batch
runtime as a dedicated-card benchmark or claim full training-memory acceptance.

Formal train, optimizer construction and save_model entrypoints are locked.
No training launcher or train-through callback is provided in this release.
