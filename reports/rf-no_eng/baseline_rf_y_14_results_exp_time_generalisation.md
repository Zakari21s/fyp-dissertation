# Baseline Model Results (Random Forest) - exp_time_generalisation

**Generated:** 2026-02-13T03:21:31.043021

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_14`
- **Class Weight:** `none`
- **Processing Time:** 348.94s

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

- **Class Weight:** `none`
- **Best Threshold (from VAL):** 0.090

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|-----------|--------|---------|-----------|--------|-----|
| VAL | 0.090 | 0.0175 | 0.5007 | 0.0348 | 0.0348 | 0.0348 |
| TEST | 0.090 | 0.0177 | 0.5051 | 0.0212 | 0.0212 | 0.0212 |

