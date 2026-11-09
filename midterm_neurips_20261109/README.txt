ECE285 midterm report — November 9, 2026

Authors: Hongyi Ding, Yang Zeng, Jasmine Jiang
Source: main.tex
Style: original NeurIPS 2025 style, copied unchanged from the proposal.
The preprint notice text identifies the course report, not a conference submission.

Build with Tectonic (tested with 0.17.0):
  tectonic --keep-logs --keep-intermediates main.tex
Or use a standard LaTeX installation and run pdflatex twice.
No BibTeX step is needed; the nine author-year references are in main.tex.

Scope:
  Five body pages plus one reference page.
  Main result: corrected frozen-PRM A/B comparison at about 6M completion tokens.
  C-bounded: a separate, explicitly preliminary experiment, briefly described.
  Original results and training files are read-only inputs, not rewritten.
  No GPU use, model generation, training, or submission for report preparation.

Verification:
  citation_verification.json records the primary-source citation checks.
  verify_report.py reads local saved evidence under the parent project and
  writes verification.json. It recounts saved outcomes and checks scalar math;
  historical model/tensor/verifier acceptance receipts remain separate.
  The script imports no GPU or model libraries. It requires the original local
  evidence folders and does not download data or launch jobs.
  pdf_qa.json records pagination, font/style checks and visual inspection.

The final report is ece285_midterm_report_20261109.pdf. The user submits it.
