# LR Engineered (tabular-equivalent + FE) – exp_time_generalisation

**Generated:** 2026-04-10T04:58:38.894667
- **Label:** `y_30`
- **Data source:** data_engineered
- **Feature set mode:** `delta` (see `column_matches_feature_set` in script)
- **Class weight:** `balanced` (matches pre-FE `train_baselines_y30` when `balanced`)
- **Row cap policy:** load parquet pool in hive order until cumulative rows ≥ cap; then keep all positives, sample negatives to cap, shuffle (same as pre-FE `train_baselines_y30` loaders).

## Train negative downsampling

- **Status:** enabled
- **Requested ratio:** 1:2 (negatives per positive)
- **Original train pool (after row cap, before neg downsampling):** 5,000,000 rows — 4,516 pos / 4,995,484 neg
- **Sampled train (used for LR fit):** 13,548 rows — 4,516 pos / 9,032 neg

## Data summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|-----------|
| Train (used for fit) | 13,548 | 4,516 | 9,032 |
| Val   | 2,000,000 | 9,738 | 1,990,262 |
| Test  | 2,000,000 | 13,708 | 1,986,292 |

**Features:** 85
**Missingness (before imputation):** 52.31%

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

**Best threshold (from val):** 0.81

**Val:**
  - Accuracy: 0.9876
  - Precision: 0.0326
  - Recall: 0.0541
  - F1: 0.0407
  - ROC-AUC: 0.5818
  - PR-AUC: 0.0090
  - Threshold: 0.810
  - TN/FP/FN/TP: 1974632/15630/9211/527


**Test:**
  - Accuracy: 0.9839
  - Precision: 0.0079
  - Recall: 0.0108
  - F1: 0.0091
  - ROC-AUC: 0.4814
  - PR-AUC: 0.0068
  - Threshold: 0.810
  - TN/FP/FN/TP: 1967734/18558/13560/148


**Processing time:** 33.65s

## Feature names (reproducibility)

Sorted column names used in X (same order as training matrix):

```
delta1_log1p_r_1
delta1_log1p_r_12
delta1_log1p_r_170
delta1_log1p_r_173
delta1_log1p_r_175
delta1_log1p_r_177
delta1_log1p_r_183
delta1_log1p_r_190
delta1_log1p_r_194
delta1_log1p_r_195
delta1_log1p_r_196
delta1_log1p_r_241
delta1_log1p_r_242
delta1_log1p_r_5
delta1_log1p_r_9
delta7_log1p_r_1
delta7_log1p_r_12
delta7_log1p_r_170
delta7_log1p_r_173
delta7_log1p_r_175
delta7_log1p_r_177
delta7_log1p_r_183
delta7_log1p_r_190
delta7_log1p_r_194
delta7_log1p_r_195
delta7_log1p_r_196
delta7_log1p_r_241
delta7_log1p_r_242
delta7_log1p_r_5
delta7_log1p_r_9
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