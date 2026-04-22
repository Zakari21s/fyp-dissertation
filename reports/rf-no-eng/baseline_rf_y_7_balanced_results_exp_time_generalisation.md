# Baseline Model Results (Random Forest) - exp_time_generalisation

**Generated:** 2026-02-13T03:56:15.799852

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_7`
- **Class Weight:** `balanced`
- **Processing Time:** 280.32s

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

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.020

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.020 | 0.0090 | 0.5081 | 0.0162 | 0.0519 | 0.0247 |
| TEST | 0.020 | 0.0092 | 0.4905 | 0.0107 | 0.0315 | 0.0160 |

