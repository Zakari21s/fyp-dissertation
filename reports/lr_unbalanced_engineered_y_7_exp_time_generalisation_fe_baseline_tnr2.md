# LR Engineered (tabular-equivalent + FE) – exp_time_generalisation

**Generated:** 2026-04-10T15:22:54.759369
- **Label:** `y_7`
- **Data source:** data_engineered
- **Feature set mode:** `baseline` (see `column_matches_feature_set` in script)
- **Class weight:** `balanced` (matches pre-FE `train_baselines_y30` when `balanced`)
- **Row cap policy:** load parquet pool in hive order until cumulative rows ≥ cap; then keep all positives, sample negatives to cap, shuffle (same as pre-FE `train_baselines_y30` loaders).

## Train negative downsampling

- **Status:** enabled
- **Requested ratio:** 1:2 (negatives per positive)
- **Original train pool (after row cap, before neg downsampling):** 5,000,000 rows — 1,235 pos / 4,998,765 neg
- **Sampled train (used for LR fit):** 3,705 rows — 1,235 pos / 2,470 neg

## Data summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|-----------|
| Train (used for fit) | 3,705 | 1,235 | 2,470 |
| Val   | 2,000,000 | 1,661 | 1,998,339 |
| Test  | 2,000,000 | 3,858 | 1,996,142 |

**Features:** 55
**Missingness (before imputation):** 44.03%

## Always-negative

**Val:**
  - Accuracy: 0.9992
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
  - ROC-AUC: 0.5000
  - PR-AUC: 0.0008
  - Threshold: 0.500
  - TN/FP/FN/TP: 1998339/0/1661/0


**Test:**
  - Accuracy: 0.9981
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
  - ROC-AUC: 0.5000
  - PR-AUC: 0.0019
  - Threshold: 0.500
  - TN/FP/FN/TP: 1996142/0/3858/0


## Logistic Regression

**Best threshold (from val):** 0.99

**Val:**
  - Accuracy: 0.9961
  - Precision: 0.0090
  - Recall: 0.0337
  - F1: 0.0143
  - ROC-AUC: 0.5795
  - PR-AUC: 0.0019
  - Threshold: 0.990
  - TN/FP/FN/TP: 1992200/6139/1605/56


**Test:**
  - Accuracy: 0.9950
  - Precision: 0.0036
  - Recall: 0.0057
  - F1: 0.0044
  - ROC-AUC: 0.4910
  - PR-AUC: 0.0020
  - Threshold: 0.990
  - TN/FP/FN/TP: 1990075/6067/3836/22


**Processing time:** 31.60s

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