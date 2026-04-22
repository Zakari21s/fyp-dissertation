# LR Engineered (tabular-equivalent + FE) – exp_time_generalisation

**Generated:** 2026-04-10T15:36:43.130956
- **Label:** `y_14`
- **Data source:** data_engineered
- **Feature set mode:** `baseline` (see `column_matches_feature_set` in script)
- **Class weight:** `balanced` (matches pre-FE `train_baselines_y30` when `balanced`)
- **Row cap policy:** load parquet pool in hive order until cumulative rows ≥ cap; then keep all positives, sample negatives to cap, shuffle (same as pre-FE `train_baselines_y30` loaders).

## Train negative downsampling

- **Status:** enabled
- **Requested ratio:** 1:2 (negatives per positive)
- **Original train pool (after row cap, before neg downsampling):** 5,000,000 rows — 2,307 pos / 4,997,693 neg
- **Sampled train (used for LR fit):** 6,921 rows — 2,307 pos / 4,614 neg

## Data summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|-----------|
| Train (used for fit) | 6,921 | 2,307 | 4,614 |
| Val   | 2,000,000 | 5,025 | 1,994,975 |
| Test  | 2,000,000 | 7,431 | 1,992,569 |

**Features:** 55
**Missingness (before imputation):** 43.99%

## Always-negative

**Val:**
  - Accuracy: 0.9975
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
  - ROC-AUC: 0.5000
  - PR-AUC: 0.0025
  - Threshold: 0.500
  - TN/FP/FN/TP: 1994975/0/5025/0


**Test:**
  - Accuracy: 0.9963
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
  - ROC-AUC: 0.5000
  - PR-AUC: 0.0037
  - Threshold: 0.500
  - TN/FP/FN/TP: 1992569/0/7431/0


## Logistic Regression

**Best threshold (from val):** 0.86

**Val:**
  - Accuracy: 0.9952
  - Precision: 0.0388
  - Recall: 0.0378
  - F1: 0.0383
  - ROC-AUC: 0.5898
  - PR-AUC: 0.0065
  - Threshold: 0.860
  - TN/FP/FN/TP: 1990274/4701/4835/190


**Test:**
  - Accuracy: 0.9943
  - Precision: 0.0059
  - Recall: 0.0032
  - F1: 0.0042
  - ROC-AUC: 0.4811
  - PR-AUC: 0.0038
  - Threshold: 0.860
  - TN/FP/FN/TP: 1988517/4052/7407/24


**Processing time:** 30.03s

## Feature names (reproducibility)

Sorted column names used in X (same order as training matrix):

```
n_1
n_12
n_170
n_171
n_172
n_173
n_175
n_177
n_180
n_181
n_182
n_183
n_184
n_187
n_190
n_194
n_195
n_196
n_199
n_232
n_233
n_241
n_242
n_5
n_9
r_1
r_12
r_170
r_171
r_172
r_173
r_174
r_175
r_177
r_180
r_181
r_182
r_183
r_184
r_187
r_188
r_190
r_192
r_194
r_195
r_196
r_197
r_198
r_199
r_206
r_241
r_242
r_244
r_5
r_9
```