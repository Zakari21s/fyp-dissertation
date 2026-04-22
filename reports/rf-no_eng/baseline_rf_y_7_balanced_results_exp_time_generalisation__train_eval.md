# Baseline Model Results (Random Forest) - exp_time_generalisation__train_eval

**Generated:** 2026-02-13T04:10:50.819675

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_7`
- **Class Weight:** `balanced`
- **Processing Time:** 303.80s

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

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.200

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.200 | 0.0082 | 0.5033 | 0.0120 | 0.0547 | 0.0197 |
| TEST | 0.200 | 0.0095 | 0.4950 | 0.0097 | 0.0308 | 0.0147 |

