# Partial dataset release

This directory publishes a partial dataset from the FoodAudit-AG benchmark.

- Full benchmark described in the manuscript: 450 recipes.
- Publicly released here: 150 recipes.
- Remaining 300 recipes: part of the internally curated benchmark and not cleared for unrestricted redistribution.
- Regulatory reference: GB 2760-2024.
- Release version: partial-dataset-v1.0.0.

The released slice preserves one third of each predefined difficulty stratum in the full benchmark: L1, 30/90; L2, 45/135; L3, 60/180; and L4, 15/45. It is not claimed to be a random or commercially representative sample or a complete substitute for the full benchmark.

The release supports transparency and independent verification. These mechanism-oriented benchmark cases were initialized from generated candidates and manually reviewed. They are not presented as representative commercial or industrial formulations.

Files: recipes.csv, items.csv, recipes.jsonl, correction_log.csv, validation_issues.csv, dataset_manifest.json, full_benchmark_manifest.json.

Additional diagnostic file: hard_noise_probe_30.csv contains 30 perturbed inputs derived from the public partial dataset for robustness inspection under incomplete labels, ambiguous commercial names, compound foods, multilingual synonyms, and missing quantity units. Aggregate results are provided under results/robustness/.
