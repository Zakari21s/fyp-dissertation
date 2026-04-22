# Baseline Model Results (Balanced LR) - exp_time_generalisation__train_eval

**Generated:** 2026-02-12T16:49:52.758890

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_7`
- **Class Weight:** `balanced`
- **Processing Time:** 305.19s

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

### Logistic Regression (Balanced)

- **Class Weight:** `balanced`
- **Best Threshold (from VAL):** 0.850

**Validation:**
- Threshold: 0.850
- PR-AUC: 0.0091
- ROC-AUC: 0.5200
- Precision: 0.0199
- Recall: 0.0308
- F1: 0.0242
- Confusion Matrix: TN=1962582, FP=22540, FN=14420, TP=458

**Test:**
- Threshold: 0.850
- PR-AUC: 0.0105
- ROC-AUC: 0.5236
- Precision: 0.0107
- Recall: 0.1775
- F1: 0.0202
- Confusion Matrix: TN=1680576, FP=301096, FN=15074, TP=3254

