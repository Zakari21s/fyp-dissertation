# Baseline Model Results (Random Forest) - exp_time_generalisation

**Generated:** 2026-02-13T02:42:11.297654

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_30`
- **Class Weight:** `none`
- **Processing Time:** 374.35s

## Data Summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|----------|
| Train | 5,000,000 | 9,711 | 4,990,289 |
| Val | 2,000,000 | 47,092 | 1,952,908 |
| Test | 2,000,000 | 75,526 | 1,924,474 |

## Feature Information

- **Feature Count:** 55
- **Missingness (before imputation):** 40.53%

## Model Results

### Random Forest

- **Class Weight:** `none`
- **Best Threshold (from VAL):** 0.020

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.020 | 0.0278 | 0.5256 | 0.0303 | 0.2108 | 0.0530 |
| TEST | 0.020 | 0.0381 | 0.5040 | 0.0385 | 0.1977 | 0.0644 |

