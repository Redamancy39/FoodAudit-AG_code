# Data card: FoodAudit-AG partial dataset

## Scope

This is a partial release of 150 records from the 450-recipe benchmark described in the manuscript. It contains 403 additive-level judgments.

## Intended use

The release supports transparency, error analysis, statistical verification, and reproduction of the FoodAudit-AG evaluation workflow.

## Provenance

Candidate formulations were generated to cover regulatory mechanisms and then reviewed against GB 2760-2024. The released cases are mechanism-oriented benchmark records. They are not claimed to be a representative sample of commercial products or industrial formulations.

## Contents

Each recipe includes the Chinese input question, anchored category, difficulty and mechanism bucket, additive-level labels, and recipe-level label. Each additive record includes the declared amount, category path, supporting rule category, normalized additive name, maximum amount, unit, remark, and calculation basis where available.

## Corrections

The release builder reconciled question text, declared category, additive names, labels, and evidence. Every change is listed in data/partial_dataset/v1/correction_log.csv. The original workbook remains unchanged.

## Privacy

The released fields contain no names, contact details, account identifiers, or enterprise-confidential formulation identifiers.

## Limitations

- Only 150 of 450 records are public.
- The remaining 300 records are maintained by a co-author.
- The benchmark is mechanism-oriented and should not be used to estimate real-world prevalence.
- GB 2760 interpretations should be reviewed by qualified regulatory personnel.
