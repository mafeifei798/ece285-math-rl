Two independent48-update branches from C-bounded checkpoint288, seed17.
grpo retains the frozen per-response length reduction.
fixed_length selects the frozen TRL dr_grpo reduction, with denominator2048.
All advantage/reward/PRM/LR/KL/sampling/length-limit settings are shared.
The shared native-LoRA/eager dependency remains immutable.
Controller order: CPU gates -> real-model no-update -> grpo48 -> fixed_length48 -> validation -> CPU comparison.
No GPU job is started by importing these files. No automatic retry or expansion.
