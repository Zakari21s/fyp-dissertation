# Baseline Model Results (Balanced LR) - exp_time_generalisation

**Generated:** 2026-02-12T15:24:00.776441

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_30`
- **Class Weight:** `balanced`
- **Processing Time:** 813.81s

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

### Always-Negative Predictor

**Validation:**
- PR-AUC: 0.0235
- ROC-AUC: 0.5000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- Confusion Matrix: TN=1952908, FP=0, FN=47092, TP=0

**Test:**
- PR-AUC: 0.0378
- ROC-AUC: 0.5000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- Confusion Matrix: TN=1924474, FP=0, FN=75526, TP=0

### Logistic Regression (Balanced)

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.760

**Validation:**
- Threshold: 0.760
- PR-AUC: 0.0433
- ROC-AUC: 0.5997
- Precision: 0.1311
- Recall: 0.0616
- F1: 0.0838
- Confusion Matrix: TN=1933692, FP=19216, FN=44192, TP=2900

**Test:**
- Threshold: 0.760
- PR-AUC: 0.0400
- ROC-AUC: 0.4895
- Precision: 0.0595
- Recall: 0.0191
- F1: 0.0290
- Confusion Matrix: TN=1901642, FP=22832, FN=74081, TP=1445

