# Baseline Model Results (Random Forest) - exp_time_generalisation__train_eval

**Generated:** 2026-02-13T04:02:43.855397

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_7`
- **Class Weight:** `none`
- **Processing Time:** 334.84s

## Data Summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|----------|
| Train | 5,000,000 | 58,010 | 4,941,990 |
| Val | 2,000,000 | 14,878 | 1,985,122 |
| Test | 2,000,000 | 18,328 | 1,981,672 |

## Feature Information

- **Feature Count:** 55
- **Missingness (before imputation):** 40.20%

## Model Results

### Random Forest

- **Class Weight:** `none`
- **Best Threshold (from VAL):** 0.370

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.370 | 0.0084 | 0.5161 | 0.0139 | 0.0306 | 0.0191 |
| TEST | 0.370 | 0.0098 | 0.4951 | 0.0142 | 0.0238 | 0.0178 |

