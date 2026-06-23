# FoodAudit-AG

FoodAudit-AG is an auditable workflow for food-additive compliance-state assessment under GB 2760-2024.

## Partial dataset release

**This repository publishes part of the benchmark dataset: 150 of 450 recipes.**

**本仓库公布部分数据集：完整基准共 450 条配方，本次公开其中 150 条。**

The public release contains 150 recipe-level records and 403 additive-level judgments with labels, category paths, and regulatory evidence fields. The remaining 300 records are part of the internally curated benchmark and have not been cleared for unrestricted redistribution.

### Public-slice selection

The released slice preserves one third of every predefined difficulty stratum in the full benchmark: L1, 30 of 90; L2, 45 of 135; L3, 60 of 180; and L4, 15 of 45. The slice is intended for transparency and mechanism-level verification and is not claimed to be a random or commercially representative sample or a complete substitute for the full benchmark.

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
- Full 450-record freeze status: partial public release. The public 150-record slice is frozen; the remaining 300 records are part of the internally curated benchmark and have not been cleared for unrestricted redistribution.
- Code freeze: revision baseline dated 2026-06-22. The manuscript metrics reported in the revised submission correspond to the frozen evaluation snapshot described in the paper and Online Resource 1.

## Environment

Copy .env.example to .env and set credentials locally. Do not commit API keys or database passwords.

    python -m pip install -r requirements.txt

## Rebuild the partial release

Run `scripts/build_partial_dataset.py` with the source workbook and the frozen CSV exports under `resources/kg_snapshot/v1/`. The release builder fails when category, amount, evidence, or label-rule validation issues remain.
