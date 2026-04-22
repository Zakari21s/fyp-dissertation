# LR Engineered (tabular-equivalent + FE) – exp_time_generalisation

**Generated:** 2026-04-10T15:39:10.166265
- **Label:** `y_14`
- **Data source:** data_engineered
- **Feature set mode:** `rollvar` (see `column_matches_feature_set` in script)
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

**Features:** 70
**Missingness (before imputation):** 41.72%

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

**Best threshold (from val):** 0.87

**Val:**
  - Accuracy: 0.9888
  - Precision: 0.0177
  - Recall: 0.0637
  - F1: 0.0277
  - ROC-AUC: 0.6149
  - PR-AUC: 0.0068
  - Threshold: 0.870
  - TN/FP/FN/TP: 1977214/17761/4705/320


**Test:**
  - Accuracy: 0.9876
  - Precision: 0.0070
  - Recall: 0.0167
  - F1: 0.0099
  - ROC-AUC: 0.5001
  - PR-AUC: 0.0042
  - Threshold: 0.870
  - TN/FP/FN/TP: 1975090/17479/7307/124


**Processing time:** 30.86s

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
rollstd7_log1p_r_1
rollstd7_log1p_r_12
rollstd7_log1p_r_170
rollstd7_log1p_r_173
rollstd7_log1p_r_175
rollstd7_log1p_r_177
rollstd7_log1p_r_183
rollstd7_log1p_r_190
rollstd7_log1p_r_194
rollstd7_log1p_r_195
rollstd7_log1p_r_196
rollstd7_log1p_r_241
rollstd7_log1p_r_242
rollstd7_log1p_r_5
rollstd7_log1p_r_9
```