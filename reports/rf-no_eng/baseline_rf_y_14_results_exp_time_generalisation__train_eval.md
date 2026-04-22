# Baseline Model Results (Random Forest) - exp_time_generalisation__train_eval

**Generated:** 2026-02-13T03:33:25.454638

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_14`
- **Class Weight:** `none`
- **Processing Time:** 337.92s

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

- **Class Weight:** `none`
- **Best Threshold (from VAL):** 0.530

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.530 | 0.0227 | 0.5269 | 0.0726 | 0.0385 | 0.0503 |
| TEST | 0.530 | 0.0184 | 0.5048 | 0.0231 | 0.0119 | 0.0157 |

