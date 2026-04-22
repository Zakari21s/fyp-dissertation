# Baseline Model Results (Random Forest) - exp_time_generalisation__train_eval

**Generated:** 2026-02-13T03:39:31.723206

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_14`
- **Class Weight:** `balanced`
- **Processing Time:** 307.78s

## Data Summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|----------|
| Train | 5,000,000 | 116,666 | 4,883,334 |
| Val | 2,000,000 | 27,555 | 1,972,445 |
| Test | 2,000,000 | 34,733 | 1,965,267 |

## Feature Information

- **Feature Count:** 55
- **Missingness (before imputation):** 40.19%

## Model Results

### Random Forest

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.300

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.300 | 0.0168 | 0.5194 | 0.0263 | 0.0641 | 0.0373 |
| TEST | 0.300 | 0.0181 | 0.5115 | 0.0192 | 0.0372 | 0.0254 |

