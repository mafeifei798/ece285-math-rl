# Third-party source provenance

This repository contains an experimental course-project snapshot, including upstream source excerpts needed to audit step/token alignment and reward calculations. Original upstream headers are retained where present. No blanket license is granted over third-party code by this repository.

| Source | Pinned revision / context | Locations |
|---|---|---|
| [CJReinforce/PURE](https://github.com/CJReinforce/PURE) OpenRLHF branch | `41f9363b8cc4af044b7fc5f7f9bbbf62139e9553` | `execution/pure_official_fix_audit_20261023/source_openrlhf/`, `source_openrlhf_extra/`, OpenRLHF `vendor.py` and vendor manifests |
| CJReinforce/PURE veRL extraction | `b174e508a4469663543cfcb366be78cf199d20bf` | `experiments/pure_integration_20261022/vendor/` and manifests |
| [Hugging Face TRL](https://github.com/huggingface/trl/tree/v0.23.1) | 0.23.1; dependency and preserved audit source | Apache-2.0 upstream; consult [upstream LICENSE](https://github.com/huggingface/trl/blob/v0.23.1/LICENSE) |
| NeurIPS 2025 style | Official course proposal template | `proposal_neurips/neurips_2025.sty`; retain source notices |

The preserved PURE source snapshot did not include a verified top-level license file. Consult the corresponding upstream repository and individual source-file notices for applicable terms; this document does not assign an inferred Apache/MIT license to it.

Data and pretrained model weights are not distributed here; obtain them from their original publishers under their applicable terms.
