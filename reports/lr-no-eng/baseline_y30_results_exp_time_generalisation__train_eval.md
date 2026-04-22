# Baseline Model Results - exp_time_generalisation__train_eval

**Generated:** 2026-02-12T04:40:55.833785

## Overview

- **Experiment Name:** `exp_time_generalisation__train_eval`
- **Target Label:** `y_30`
- **Processing Time:** 230.77s

## Data Summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|----------|
| Train | 5,000,000 | 272,397 | 4,727,603 |
| Val | 2,000,000 | 47,092 | 1,952,908 |
| Test | 2,000,000 | 75,526 | 1,924,474 |

## Feature Information

- **Feature Count:** 55
- **Missingness (before imputation):** 40.19%

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

### Logistic Regression

- **Best Threshold (from VAL):** 0.780

**Validation:**
- Threshold: 0.780
- PR-AUC: 0.0272
- ROC-AUC: 0.5242
- Precision: 0.0287
- Recall: 0.2740
- F1: 0.0519
- Confusion Matrix: TN=1515548, FP=437360, FN=34188, TP=12904

**Test:**
- Threshold: 0.780
- PR-AUC: 0.0428
- ROC-AUC: 0.5326
- Precision: 0.0442
- Recall: 0.5405
- F1: 0.0817
- Confusion Matrix: TN=1041522, FP=882952, FN=34704, TP=40822

