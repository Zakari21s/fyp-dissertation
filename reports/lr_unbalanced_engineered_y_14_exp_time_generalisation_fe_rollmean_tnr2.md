# LR Engineered (tabular-equivalent + FE) – exp_time_generalisation

**Generated:** 2026-04-10T15:38:08.717469
- **Label:** `y_14`
- **Data source:** data_engineered
- **Feature set mode:** `rollmean` (see `column_matches_feature_set` in script)
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
**Missingness (before imputation):** 39.10%

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

**Best threshold (from val):** 0.89

**Val:**
  - Accuracy: 0.9901
  - Precision: 0.0182
  - Recall: 0.0557
  - F1: 0.0274
  - ROC-AUC: 0.6165
  - PR-AUC: 0.0062
  - Threshold: 0.890
  - TN/FP/FN/TP: 1979858/15117/4745/280


**Test:**
  - Accuracy: 0.9882
  - Precision: 0.0083
  - Recall: 0.0182
  - F1: 0.0114
  - ROC-AUC: 0.4912
  - PR-AUC: 0.0040
  - Threshold: 0.890
  - TN/FP/FN/TP: 1976360/16209/7296/135


**Processing time:** 30.34s

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
rollmean7_log1p_r_1
rollmean7_log1p_r_12
rollmean7_log1p_r_170
rollmean7_log1p_r_173
rollmean7_log1p_r_175
rollmean7_log1p_r_177
rollmean7_log1p_r_183
rollmean7_log1p_r_190
rollmean7_log1p_r_194
rollmean7_log1p_r_195
rollmean7_log1p_r_196
rollmean7_log1p_r_241
rollmean7_log1p_r_242
rollmean7_log1p_r_5
rollmean7_log1p_r_9
```