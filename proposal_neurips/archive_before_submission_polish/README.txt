ECE285 proposal: asymmetric offline process-reward adaptation

Main document: main.tex
Style: neurips_2025.sty, unmodified official NeurIPS 2025 style
Style source: https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip
Mode: preprint (named authors; standard template margins and font sizes)

The course limit is two pages excluding references. The delivered PDF must
contain exactly two pages of proposal text followed by references.

Authors, in the supplied order: Hongyi Ding, Jasmine Jiang, Yang Zeng.
Hongyi Ding: GRPO and GPU profiling.
Jasmine Jiang: PRM data and adaptation.
Yang Zeng: data splitting, evaluation, and statistics.
All three contribute to the report. Experimental groups A/B/C/D are methods,
not team-member labels.

To edit: upload main.tex and neurips_2025.sty to Overleaf, or open this folder
in the Codex LaTeX editor. Standard LaTeX compilation is supported. With a
local TeX installation, compile main.tex twice with pdflatex, or run tectonic
main.tex. No BibTeX run is needed: references are embedded in main.tex.

This is a proposal, not a report of completed training. The model and compute
settings are provisional until the shared hardware pilot is completed.
No training or course submission has been performed as part of document creation.

The project follows protocol_simple.json (in the parent workspace directory;
included alongside this README in the downloadable source archive). Older
project_plan.html/protocol.json describe a superseded online-PRM variant.

Review update (2026-10-19):
The direction and four arms are retained. Main evaluation is MATH500 after
checkpoint-corpus overlap audit; full MATH is exposure-aware supplementary
evaluation because PRM800K expanded training into 4,500 original test items.
The dataset audit and GPU pilot are planned, not completed.
REVIEW_NOTES.txt records citation, equation, comparison, and course checks.
check_proposal_math.py / math_checks.json provide independent algebra checks,
not evidence of training performance. Main proposal and protocol_simple.json
contain the reviewed specification.

Optimization update (2026-10-20):
The direction, equations, four training arms, model choices, and team roles are
unchanged. The primary contrast is D-C MATH500 policy accuracy. The initial
policy is also evaluated, without adding another training arm. All policy
tests use identical greedy decoding without PRM reranking or voting.

The same ProcessBench MATH scores support a secondary valid-chain AUROC on
answer-correct solutions. This helps distinguish ranking quality from a
uniform score reduction and uses existing annotations, not a new judge.
The diagnostic is never used for tuning; overlap audits remain pending.
The paired-seed protocol and resource fallback are recorded in
protocol_simple.json, version 2.2, and summarized in REVIEW_NOTES.txt.
