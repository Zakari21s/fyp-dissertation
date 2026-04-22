# Baseline Model Results (Random Forest) - exp_time_generalisation

**Generated:** 2026-02-13T03:50:59.966412

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_7`
- **Class Weight:** `none`
- **Processing Time:** 331.39s

## Data Summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|----------|
| Train | 5,000,000 | 2,459 | 4,997,541 |
| Val | 2,000,000 | 14,878 | 1,985,122 |
| Test | 2,000,000 | 18,328 | 1,981,672 |

## Feature Information

- **Feature Count:** 55
- **Missingness (before imputation):** 40.53%

## Model Results

### Random Forest

- **Class Weight:** `none`
- **Best Threshold (from VAL):** 0.020

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.020 | 0.0088 | 0.4928 | 0.0106 | 0.1340 | 0.0197 |
| TEST | 0.020 | 0.0095 | 0.4991 | 0.0098 | 0.0966 | 0.0178 |

