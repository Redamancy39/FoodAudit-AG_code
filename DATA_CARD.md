# Data card: FoodAudit-AG partial dataset

## Scope

This is a partial release of 150 records from the 450-recipe benchmark described in the manuscript. It contains 403 additive-level judgments.

## Intended use

The release supports transparency, error analysis, statistical verification, and reproduction of the FoodAudit-AG evaluation workflow.

## Provenance

Candidate formulations were generated to cover regulatory mechanisms and then reviewed against GB 2760-2024. The released cases are mechanism-oriented benchmark records. They are not claimed to be a representative sample of commercial products or industrial formulations.

## Public-slice selection

The public release preserves one third of each predefined difficulty stratum in the 450-record benchmark: L1, 30/90; L2, 45/135; L3, 60/180; and L4, 15/45. It should not be interpreted as a random or prevalence-representative sample or a complete substitute for the full benchmark.

## Contents

Each recipe includes the Chinese input question, anchored category, difficulty and mechanism bucket, additive-level labels, and recipe-level label. Each additive record includes the declared amount, category path, supporting rule category, normalized additive name, maximum amount, unit, remark, and calculation basis where available.

## Corrections

The release builder reconciled question text, declared category, additive names, labels, and evidence. Every change is listed in data/partial_dataset/v1/correction_log.csv. The original workbook remains unchanged.

## Privacy

The released fields contain no names, contact details, account identifiers, or enterprise-confidential formulation identifiers.

## Limitations

- Only 150 of 450 records are public.
- The remaining 300 records are part of the internally curated benchmark and have not been cleared for unrestricted redistribution.
- The benchmark is mechanism-oriented and should not be used to estimate real-world prevalence.
- GB 2760 interpretations should be reviewed by qualified regulatory personnel.
