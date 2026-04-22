# Baseline Model Results - exp_time_generalisation

**Generated:** 2026-02-12T15:56:57.098180

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_14`
- **Processing Time:** 457.70s

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

### Always-Negative Predictor

**Validation:**
- PR-AUC: 0.0138
- ROC-AUC: 0.5000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- Confusion Matrix: TN=1972445, FP=0, FN=27555, TP=0

**Test:**
- PR-AUC: 0.0174
- ROC-AUC: 0.5000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- Confusion Matrix: TN=1965267, FP=0, FN=34733, TP=0

### Logistic Regression

- **Best Threshold (from VAL):** 0.010

**Validation:**
- Threshold: 0.010
- PR-AUC: 0.0416
- ROC-AUC: 0.6020
- Precision: 0.3399
- Recall: 0.0188
- F1: 0.0356
- Confusion Matrix: TN=1971441, FP=1004, FN=27038, TP=517

**Test:**
- Threshold: 0.010
- PR-AUC: 0.0180
- ROC-AUC: 0.4972
- Precision: 0.0138
- Recall: 0.0004
- F1: 0.0008
- Confusion Matrix: TN=1964194, FP=1073, FN=34718, TP=15

