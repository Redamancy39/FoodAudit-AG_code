# Hard-noise robustness diagnostic subset

This note documents the hard-noise diagnostic subset released with the partial FoodAudit-AG benchmark.

## Scope

The subset contains 30 perturbed inputs derived from records that are already included in the public 150-recipe partial dataset. It is a diagnostic robustness probe, not a replacement for the 450-recipe benchmark.

The perturbations cover five practical input-noise conditions:

1. incomplete labels;
2. ambiguous commercial names;
3. compound foods;
4. multilingual synonyms;
5. missing quantity units.

Each condition contains six cases.

## Public files

- `data/partial_dataset/v1/hard_noise_probe_30.csv`: perturbed inputs, original sample IDs, noise type, gold labels, and difficulty metadata.
- `results/robustness/hard_noise_summary.json`: aggregate diagnostic metrics.
- `results/robustness/hard_noise_by_type.csv`: metrics grouped by perturbation type.

The complete per-run diagnostic logs are not included in the public release. The released files are intended to support inspection of the diagnostic input construction and the reported aggregate robustness results.

## Summary results

On the 30-case hard-noise subset, FoodAudit-AG obtained:

- recipe-level accuracy: 0.8333;
- binary F1: 0.8293;
- anchor accuracy: 0.6667;
- false-safe recipe-level cases: 0/30.

The strongest degradation occurred for incomplete labels. Ambiguous commercial names, compound-food descriptors, and multilingual synonym variants were fully correct in this diagnostic subset. Missing quantity units produced a smaller reduction in performance. The observed errors were conservative alerts rather than false-safe clearances.
