# Baseline Model Results - exp_time_generalisation

**Generated:** 2026-02-12T16:09:14.877307

## Overview

- **Experiment Name:** `exp_time_generalisation`
- **Target Label:** `y_7`
- **Processing Time:** 261.96s

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

### Logistic Regression

- **Best Threshold (from VAL):** 0.010

**Validation:**
- Threshold: 0.010
- PR-AUC: 0.0241
- ROC-AUC: 0.5872
- Precision: 0.2552
- Recall: 0.0075
- F1: 0.0145
- Confusion Matrix: TN=1984798, FP=324, FN=14767, TP=111

**Test:**
- Threshold: 0.010
- PR-AUC: 0.0095
- ROC-AUC: 0.4960
- Precision: 0.0038
- Recall: 0.0002
- F1: 0.0003
- Confusion Matrix: TN=1980887, FP=785, FN=18325, TP=3

