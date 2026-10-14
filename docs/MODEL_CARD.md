# Model card: Historical application-case classifier

## Purpose and scope

A configurable educational binary classifier for historical case data. The current example contract uses
`case_status` with `Denied=0`, `Certified=1` and excludes `case_id`. This contract alone does not establish
that the data represents visa grants, immigration eligibility or any particular jurisdiction.

## Data status

No source-verified, redistributable training dataset is committed or evaluated. Therefore there are
**no real-world accuracy claims**. Supply data you are entitled to use, with its original source, license,
collection period, target definition and feature dictionary. The code does not silently download an
unattributed third-party CSV. The provided inference example is fictional and has no target labels.

## Inputs and methodology

Numeric and categorical columns are supported. Excluded identifiers are checked for repeated entities
before removal; resolve repeated cases before random splitting. Preprocessing is fitted only on training
rows. Stratified 60/20/20 splits, seed 42, validation AP model selection and validation F1 threshold selection
are followed by a held-out test report. The trained bundle records labels, schema types and provenance.

## Interpretation and limits

`positive_class_score` is an uncalibrated score for the historical positive label. It is not a probability
of future legal approval. Sensitive features, proxies, policy changes, post-outcome columns and grouping
must be reviewed against the actual data contract. Do not use this baseline for automated applicant
eligibility decisions. No subgroup fairness evaluation or temporal validation has been performed.

## Reproduction after obtaining documented data

```bash
python app.py train --data data/EasyVisa.csv
python app.py predict --input examples/cases.csv --output artifacts/predictions.csv
```

Change `TARGET`, `LABELS` and `DROP` explicitly if your verified dataset uses different definitions.
Only load trusted Joblib artifacts.
