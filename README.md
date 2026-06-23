# FoodAudit-AG

FoodAudit-AG is an auditable workflow for food-additive compliance-state assessment under GB 2760-2024.

## Partial dataset release

**This repository publishes part of the benchmark dataset: 150 of 450 recipes.**

**本仓库公布部分数据集：完整基准共 450 条配方，本次公开其中 150 条。**

The public release contains 150 recipe-level records and 403 additive-level judgments with labels, category paths, and regulatory evidence fields. The remaining 300 records are maintained by a co-author and are not included in this repository.

### Public-slice selection

The released slice is the 150-record portion held by the releasing author at revision time; its scope was determined by data custody rather than by post-hoc model performance. It preserves one third of every predefined difficulty stratum in the full benchmark: L1, 30 of 90; L2, 45 of 135; L3, 60 of 180; and L4, 15 of 45. The slice is intended for transparency and mechanism-level verification and is not claimed to be a random or commercially representative sample.

See data/partial_dataset/v1/README.md and DATA_CARD.md.

## Repository structure

- data/partial_dataset/v1/: public partial dataset in CSV and JSONL formats.
- data/partial_dataset/v1/hard_noise_probe_30.csv: 30-case hard-noise diagnostic subset derived from the public partial dataset.
- results/robustness/: aggregate hard-noise robustness results.
- resources/kg_snapshot/v1/: frozen category tree, permission rules, and category synonyms.
- src/: FoodAudit-AG backend snapshot used as the revision baseline.
- app/: review-oriented Streamlit prototype.
- scripts/: graph import, candidate generation, evaluation, statistics, and release builders.
- docs/category_anchoring.md: exact candidate-generation, ranking, threshold, output-resolution, and failure-handling specification.
- docs/robustness_hard_noise.md: hard-noise perturbation construction and summary results.
- CODE_FREEZE.json: checksums for the revision-baseline code and resources.

## Release status

- Public dataset: partial-dataset-v1.0.0.
- Full benchmark identifier: foodaudit-benchmark-450-v1.0.0.
- Full 450-record freeze status: partial. The public 150-record slice is frozen; checksums for the co-author's 300-record slice still need to be added.
- Code freeze: revision baseline dated 2026-06-22. All manuscript metrics must be rerun after the major-revision changes.

## Environment

Copy .env.example to .env and set credentials locally. Do not commit API keys or database passwords.

    python -m pip install -r requirements.txt

## Rebuild the partial release

Run `scripts/build_partial_dataset.py` with the source workbook and the frozen CSV exports under `resources/kg_snapshot/v1/`. The release builder fails when category, amount, evidence, or label-rule validation issues remain.
