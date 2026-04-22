# LR Engineered (tabular-equivalent + FE) – exp_time_generalisation

**Generated:** 2026-04-10T15:26:34.004790
- **Label:** `y_7`
- **Data source:** data_engineered
- **Feature set mode:** `rollmean` (see `column_matches_feature_set` in script)
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

**Features:** 70
**Missingness (before imputation):** 39.05%

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

**Best threshold (from val):** 0.92

**Val:**
  - Accuracy: 0.9922
  - Precision: 0.0052
  - Recall: 0.0446
  - F1: 0.0094
  - ROC-AUC: 0.5783
  - PR-AUC: 0.0017
  - Threshold: 0.920
  - TN/FP/FN/TP: 1984292/14047/1587/74


**Test:**
  - Accuracy: 0.9898
  - Precision: 0.0054
  - Recall: 0.0236
  - F1: 0.0088
  - ROC-AUC: 0.4933
  - PR-AUC: 0.0023
  - Threshold: 0.920
  - TN/FP/FN/TP: 1979485/16657/3767/91


**Processing time:** 33.46s

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