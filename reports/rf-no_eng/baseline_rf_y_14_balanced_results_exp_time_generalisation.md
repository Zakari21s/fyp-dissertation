# Baseline Model Results (Random Forest) - exp_time_generalisation

**Generated:** 2026-02-13T03:27:09.820926

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_14`
- **Class Weight:** `balanced`
- **Processing Time:** 298.12s

## Data Summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|----------|
| Train | 5,000,000 | 4,601 | 4,995,399 |
| Val | 2,000,000 | 27,555 | 1,972,445 |
| Test | 2,000,000 | 34,733 | 1,965,267 |

## Feature Information

- **Feature Count:** 55
- **Missingness (before imputation):** 40.53%

## Model Results

### Random Forest

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.020

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.020 | 0.0164 | 0.4828 | 0.0262 | 0.0730 | 0.0386 |
| TEST | 0.020 | 0.0174 | 0.4987 | 0.0184 | 0.0356 | 0.0243 |

