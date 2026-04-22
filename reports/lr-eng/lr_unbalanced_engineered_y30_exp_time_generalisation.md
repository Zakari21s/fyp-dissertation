# LR Unbalanced (Engineered) – exp_time_generalisation

**Generated:** 2026-02-20T03:00:45.575434
- **Label:** `y_30`
- **Data source:** data_engineered
- **Class weight:** none

## Data summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|-----------|
| Train | 5,000,000 | 4,637 | 4,995,363 |
| Val   | 2,000,000 | 7,595 | 1,992,405 |
| Test  | 2,000,000 | 13,657 | 1,986,343 |

**Features:** 72
**Missingness (before imputation):** 30.96%

## Always-negative

**Val:**
  - Accuracy: 0.9962
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
  - ROC-AUC: 0.5000
  - PR-AUC: 0.0038
  - Threshold: 0.500
  - TN/FP/FN/TP: 1992405/0/7595/0


**Test:**
  - Accuracy: 0.9932
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
  - ROC-AUC: 0.5000
  - PR-AUC: 0.0068
  - Threshold: 0.500
  - TN/FP/FN/TP: 1986343/0/13657/0


## Logistic Regression (unbalanced)

**Best threshold (from val):** 0.01

**Val:**
  - Accuracy: 0.9956
  - Precision: 0.1065
  - Recall: 0.0201
  - F1: 0.0339
  - ROC-AUC: 0.5776
  - PR-AUC: 0.0097
  - Threshold: 0.010
  - TN/FP/FN/TP: 1991121/1284/7442/153


**Test:**
  - Accuracy: 0.9925
  - Precision: 0.0173
  - Recall: 0.0018
  - F1: 0.0033
  - ROC-AUC: 0.4877
  - PR-AUC: 0.0075
  - Threshold: 0.010
  - TN/FP/FN/TP: 1984920/1423/13632/25


**Processing time:** 2667.42s