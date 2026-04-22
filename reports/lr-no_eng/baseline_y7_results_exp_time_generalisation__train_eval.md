# Baseline Model Results - exp_time_generalisation__train_eval

**Generated:** 2026-02-12T16:14:18.563085

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_7`
- **Processing Time:** 280.69s

## Data Summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|----------|
| Train | 5,000,000 | 58,010 | 4,941,990 |
| Val | 2,000,000 | 14,878 | 1,985,122 |
| Test | 2,000,000 | 18,328 | 1,981,672 |

## Feature Information

- **Feature Count:** 55
- **Missingness (before imputation):** 40.20%

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

### Logistic Regression

- **Best Threshold (from VAL):** 0.060

**Validation:**
- Threshold: 0.060
- PR-AUC: 0.0091
- ROC-AUC: 0.5191
- Precision: 0.0154
- Recall: 0.0448
- F1: 0.0230
- Confusion Matrix: TN=1942582, FP=42540, FN=14211, TP=667

**Test:**
- Threshold: 0.060
- PR-AUC: 0.0107
- ROC-AUC: 0.5223
- Precision: 0.0104
- Recall: 0.2661
- F1: 0.0200
- Confusion Matrix: TN=1517533, FP=464139, FN=13451, TP=4877

