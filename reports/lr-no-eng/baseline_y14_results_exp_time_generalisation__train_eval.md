# Baseline Model Results - exp_time_generalisation__train_eval

**Generated:** 2026-02-12T16:02:21.039026

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_14`
- **Processing Time:** 241.76s

## Data Summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|----------|
| Train | 5,000,000 | 116,666 | 4,883,334 |
| Val | 2,000,000 | 27,555 | 1,972,445 |
| Test | 2,000,000 | 34,733 | 1,965,267 |

## Feature Information

- **Feature Count:** 55
- **Missingness (before imputation):** 40.19%

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

- **Best Threshold (from VAL):** 0.110

**Validation:**
- Threshold: 0.110
- PR-AUC: 0.0159
- ROC-AUC: 0.5219
- Precision: 0.0199
- Recall: 0.0744
- F1: 0.0314
- Confusion Matrix: TN=1871403, FP=101042, FN=25504, TP=2051

**Test:**
- Threshold: 0.110
- PR-AUC: 0.0190
- ROC-AUC: 0.5253
- Precision: 0.0202
- Recall: 0.3508
- F1: 0.0383
- Confusion Matrix: TN=1375231, FP=590036, FN=22547, TP=12186

