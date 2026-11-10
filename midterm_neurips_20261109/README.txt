ECE285 midterm report — November 9, 2026
Authors: Hongyi Ding, Yang Zeng, Jasmine Jiang
Format: five body pages plus one reference page, unchanged NeurIPS 2025 style.
Build: tectonic --keep-logs --keep-intermediates main.tex
Or run pdflatex twice. All 15 author-year references are inline.
Keep figures/b_advantage_by_group.pdf alongside main.tex.

Scope: corrected historical V0 A/B; separate V1 initial/A/C-mix/C-bounded
results with both completed seeds and paired intervals; actual B advantage
distributions by outcome group; C/D PRM adaptation counts; V1 measured costs;
full-run future controls and an explicit approximate detection limit.
V0/V1 are distinct protocols and cannot be pooled as one controlled comparison.
Seed42 remains inconclusive. Planned training is not completed work.

CPU verification (inside original project checkout):
  python verify_report.py
  python make_report_figure.py
  python verify_revision.py
The figure script requires numpy and matplotlib; no script calls a model.
The scripts read original saved evidence. Raw problem/response files are not
bundled. Source paths and SHA256 hashes are retained in the JSON receipts.

No GPU use, training, model calls, verifier changes, or submission during editing.
Original experiment results and earlier drafts are preserved.
The user submits the final PDF.
