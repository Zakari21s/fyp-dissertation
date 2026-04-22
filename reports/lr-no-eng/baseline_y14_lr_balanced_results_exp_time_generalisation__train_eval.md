# Baseline Model Results (Balanced LR) - exp_time_generalisation__train_eval

**Generated:** 2026-02-12T17:12:09.929396

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_14`
- **Class Weight:** `balanced`
- **Processing Time:** 275.18s

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

### Logistic Regression (Balanced)

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.830

**Validation:**
- Threshold: 0.830
- PR-AUC: 0.0159
- ROC-AUC: 0.5204
- Precision: 0.0218
- Recall: 0.0659
- F1: 0.0328
- Confusion Matrix: TN=1891022, FP=81423, FN=25740, TP=1815

**Test:**
- Threshold: 0.830
- PR-AUC: 0.0192
- ROC-AUC: 0.5263
- Precision: 0.0200
- Recall: 0.3206
- F1: 0.0377
- Confusion Matrix: TN=1420573, FP=544694, FN=23596, TP=11137

