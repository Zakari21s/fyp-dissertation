# Baseline Model Results (Balanced LR) - exp_time_generalisation

**Generated:** 2026-02-12T16:43:44.097085

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_7`
- **Class Weight:** `balanced`
- **Processing Time:** 1107.68s

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

### Always-Negative Predictor

**Validation:**
- PR-AUC: 0.0074
- ROC-AUC: 0.5000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- Confusion Matrix: TN=1985122, FP=0, FN=14878, TP=0

**Test:**
- PR-AUC: 0.0092
- ROC-AUC: 0.5000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- Confusion Matrix: TN=1981672, FP=0, FN=18328, TP=0

### Logistic Regression (Balanced)

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.780

**Validation:**
- Threshold: 0.780
- PR-AUC: 0.0143
- ROC-AUC: 0.5850
- Precision: 0.0590
- Recall: 0.0687
- F1: 0.0635
- Confusion Matrix: TN=1968827, FP=16295, FN=13856, TP=1022

**Test:**
- Threshold: 0.780
- PR-AUC: 0.0096
- ROC-AUC: 0.5014
- Precision: 0.0126
- Recall: 0.0117
- F1: 0.0122
- Confusion Matrix: TN=1964870, FP=16802, FN=18113, TP=215

