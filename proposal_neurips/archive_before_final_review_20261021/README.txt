ECE285 proposal: asymmetric offline process-reward adaptation

Main document: main.tex
Style: neurips_2025.sty, unmodified official NeurIPS 2025 style
Style source: https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip
Mode: preprint (named authors; standard template margins and font sizes)

The course limit is two pages excluding references. The delivered PDF must
contain exactly two pages of proposal text followed by references.

Authors, in the supplied order: Hongyi Ding, Jasmine Jiang, Yang Zeng.
Experimental groups A/B/C/D are methods, not team-member labels.
Individual role assignments are omitted from the submission. A short tentative
timeline remains because the course announcement explicitly requires it.

To edit: upload main.tex and neurips_2025.sty to Overleaf, or open this folder
in the Codex LaTeX editor. Standard LaTeX compilation is supported. With a
local TeX installation, compile main.tex twice with pdflatex, or run tectonic
main.tex. No BibTeX run is needed: references are embedded in main.tex.

This is a proposal for controlled comparisons, not a claim that the proposed
method has improved accuracy. Outcome-only baseline training and evaluation
have been verified on one A100. PRM adaptation and the B/C/D comparisons remain
pending. The user will submit the PDF; no course submission was performed.

The project follows protocol_simple.json (in the parent workspace directory;
included alongside this README in the downloadable source archive). Older
project_plan.html/protocol.json describe a superseded online-PRM variant.

Review update (2026-10-19):
The direction and four arms are retained. Main evaluation is MATH500 after
checkpoint-corpus overlap audit; full MATH is exposure-aware supplementary
evaluation because PRM800K expanded training into 4,500 original test items.
That earlier review preceded the dataset audit and A/17 training; see the
submission-polish update below for the current execution status.
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
The diagnostic is never used for tuning. This historical update preceded the
completed question-inventory audit; new-anchor filtering remains to be done.
The paired-seed protocol and resource fallback are recorded in
protocol_simple.json, version 2.2, and summarized in REVIEW_NOTES.txt.

Submission-polish update (2026-10-20):
The research direction, losses and four-arm design are retained. The standalone
"Three-person plan and tentative timeline" section is removed; the required
four-week timeline is a short closing paragraph. The prose and table emphasize
the research question, supervision change, and primary D-C comparison.

The user-authorized budget is now 24M generated completion tokens per run:
four arms x three paired seeds = 288M planned tokens, with 6M/12M/24M milestones.
The primary endpoint is the fixed 24M checkpoint; intermediate test scores do
not choose it. A/17 already completed 6M and is being continued from its full
checkpoint. No proposed-method gains are reported. The exact normalized
question audit found zero overlap between MATH500/GSM8K and the public PRM800K
training inventory; unknown pretraining exposure and paraphrases are not ruled
out. ProcessBench overlap is reported separately as a diagnostic limitation.
The source archive contains the current protocol and verification receipt.

Peer-review update (2026-10-20; current version):
protocol_simple.json is now version 2.4-peer-review-diagnostics. The core loss,
four RL arms, group normalization and 24M budget are unchanged. This review
changes documents and the prospective analysis plan only; it does not change
running training or launch additional runs.

The proposal now states the same-outcome normalization caveat, an illustrative
single-seed power calculation (about 4 percentage points at 10% disagreement,
500 paired questions, two-sided alpha=.05, 80% power), step-format monitoring,
and a PRM-only anchor control S. Policy accuracy remains primary. ProcessBench
MATH has 1,000 solutions from 955 questions; the answer-correct AUROC subset is
500 solutions (406 valid, 94 flawed), before inherited-overlap stratification.
The source archive includes peer_review_response.json with the decisions and
source verification. The saved source archive remains self-contained.
