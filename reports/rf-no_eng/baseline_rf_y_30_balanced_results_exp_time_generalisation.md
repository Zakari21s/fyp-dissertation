# Baseline Model Results (Random Forest) - exp_time_generalisation

**Generated:** 2026-02-13T02:48:38.086069

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_30`
- **Class Weight:** `balanced`
- **Processing Time:** 318.02s

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

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.010

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.010 | 0.0272 | 0.5022 | 0.0292 | 0.1971 | 0.0508 |
| TEST | 0.010 | 0.0374 | 0.4942 | 0.0365 | 0.1480 | 0.0586 |

