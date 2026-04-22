# Baseline Model Results (Random Forest) - exp_time_generalisation__train_eval

**Generated:** 2026-02-13T02:55:40.896658

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_30`
- **Class Weight:** `none`
- **Processing Time:** 332.89s

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

- **Class Weight:** `none`
- **Best Threshold (from VAL):** 0.580

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.580 | 0.0287 | 0.5145 | 0.0517 | 0.0504 | 0.0510 |
| TEST | 0.580 | 0.0388 | 0.5039 | 0.0436 | 0.0329 | 0.0375 |

