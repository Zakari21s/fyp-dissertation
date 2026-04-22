# Baseline Model Results (Random Forest) - exp_time_generalisation__train_eval

**Generated:** 2026-02-13T03:02:10.859101

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_30`
- **Class Weight:** `balanced`
- **Processing Time:** 314.47s

## Data Summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|----------|
| Train | 5,000,000 | 272,397 | 4,727,603 |
| Val | 2,000,000 | 47,092 | 1,952,908 |
| Test | 2,000,000 | 75,526 | 1,924,474 |

## Feature Information

- **Feature Count:** 55
- **Missingness (before imputation):** 40.19%

## Model Results

### Random Forest

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.050

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.050 | 0.0267 | 0.5068 | 0.0239 | 0.8499 | 0.0465 |
| TEST | 0.050 | 0.0381 | 0.5008 | 0.0379 | 0.9239 | 0.0729 |

