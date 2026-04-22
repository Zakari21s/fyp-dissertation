# Baseline Model Results (Balanced LR) - exp_time_generalisation

**Generated:** 2026-02-12T17:07:05.552692

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_14`
- **Class Weight:** `balanced`
- **Processing Time:** 877.58s

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

### Logistic Regression (Balanced)

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.770

**Validation:**
- Threshold: 0.770
- PR-AUC: 0.0273
- ROC-AUC: 0.6035
- Precision: 0.1033
- Recall: 0.0706
- F1: 0.0839
- Confusion Matrix: TN=1955554, FP=16891, FN=25610, TP=1945

**Test:**
- Threshold: 0.770
- PR-AUC: 0.0182
- ROC-AUC: 0.5027
- Precision: 0.0220
- Recall: 0.0117
- F1: 0.0153
- Confusion Matrix: TN=1947295, FP=17972, FN=34328, TP=405

