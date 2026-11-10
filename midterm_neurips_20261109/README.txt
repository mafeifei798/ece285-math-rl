ECE285 midterm report — November 9, 2026
Authors: Hongyi Ding, Yang Zeng, Jasmine Jiang
Format: five body pages plus one reference page, original NeurIPS 2025 style.

Build: tectonic --keep-logs --keep-intermediates main.tex
Or run pdflatex twice; all 15 author-year references are inline, no BibTeX needed.
Keep figures/all_wrong_advantage_ecdf.pdf alongside the source.

Content: corrected historical V0 A/B comparison; separate V1 initial/A/C-mix/
C-bounded results including both completed seeds, paired confidence intervals,
matched-sample advantage replay, C/D reward-model adaptation, and future controls.
V0/V1 are distinct protocols and cannot be pooled into one controlled comparison.
No operational latest-status narrative is included.

CPU verification (inside original project checkout):
  python verify_report.py
  python make_report_figure.py
The second command requires numpy and matplotlib. It reads saved evidence,
generates the plot and report_evidence.json, and never calls a model.
Source evidence paths and SHA256 hashes are in the verification JSON files.
Full raw data stay in the original project, outside this source bundle.

No GPU use, training, model calls, verifier changes, or submission during revision.
Historical source/results are preserved; previous_draft retains the earlier report.
The stable final PDF is ece285_midterm_report_20261109.pdf. The user submits it.
