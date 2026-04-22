# LR Engineered (tabular-equivalent + FE) – exp_time_generalisation

**Generated:** 2026-04-10T03:34:26.995485
- **Label:** `y_30`
- **Data source:** data_engineered
- **Features:** `n_*` / `r_*` (same as tabular baseline) plus engineered columns only
- **Class weight:** `balanced` (matches pre-FE `train_baselines_y30` when `balanced`)
- **Row cap policy:** load parquet pool in hive order until cumulative rows ≥ cap; then keep all positives, sample negatives to cap, shuffle (same as pre-FE `train_baselines_y30` loaders).

## Data summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|-----------|
| Train | 5,000,000 | 4,516 | 4,995,484 |
| Val   | 2,000,000 | 9,738 | 1,990,262 |
| Test  | 2,000,000 | 13,708 | 1,986,292 |

**Features:** 147
**Missingness (before imputation):** 45.69%

## Always-negative

**Val:**
  - Accuracy: 0.9951
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
  - ROC-AUC: 0.5000
  - PR-AUC: 0.0049
  - Threshold: 0.500
  - TN/FP/FN/TP: 1990262/0/9738/0


**Test:**
  - Accuracy: 0.9931
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
  - ROC-AUC: 0.5000
  - PR-AUC: 0.0069
  - Threshold: 0.500
  - TN/FP/FN/TP: 1986292/0/13708/0


## Logistic Regression

**Best threshold (from val):** 0.4

**Val:**
  - Accuracy: 0.9860
  - Precision: 0.0020
  - Recall: 0.0038
  - F1: 0.0026
  - ROC-AUC: 0.5780
  - PR-AUC: 0.0058
  - Threshold: 0.400
  - TN/FP/FN/TP: 1971935/18327/9701/37


**Test:**
  - Accuracy: 0.9847
  - Precision: 0.0029
  - Recall: 0.0036
  - F1: 0.0032
  - ROC-AUC: 0.4838
  - PR-AUC: 0.0064
  - Threshold: 0.400
  - TN/FP/FN/TP: 1969427/16865/13659/49


**Processing time:** 29414.54s