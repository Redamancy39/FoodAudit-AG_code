# Frozen knowledge-graph resources

This directory contains the GB 2760-2024 resource snapshot used to build the public partial dataset release.

- `allowed_rules.csv`: normalized additive permission rules used by the release builder (2,300 rows).
- `category_nodes.csv`: normalized food-category nodes (369 rows).
- `parent_edges.csv`: category hierarchy edges (353 rows).
- `category_synonyms.csv`: normalized category aliases (1,328 rows).
- `GB2760_*.jsonl`: corresponding source JSONL resources retained for provenance.

The source snapshot and the live Neo4j graph were not identical at freeze time: the source contains 2,300 rule rows, while the loaded graph inspection returned 2,277 `ALLOWED_IN` relationships. The public partial dataset was rebuilt and validated against the frozen source snapshot in this directory. This discrepancy must be resolved before the final manuscript evaluation is rerun.

File checksums are recorded in `CODE_FREEZE.json` and `data/partial_dataset/v1/dataset_manifest.json`.
