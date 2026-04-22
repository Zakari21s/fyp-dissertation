# LR Engineered (tabular-equivalent + FE) – exp_time_generalisation

**Generated:** 2026-04-10T04:55:37.659393
- **Label:** `y_30`
- **Data source:** data_engineered
- **Feature set mode:** `baseline` (see `column_matches_feature_set` in script)
- **Class weight:** `balanced` (matches pre-FE `train_baselines_y30` when `balanced`)
- **Row cap policy:** load parquet pool in hive order until cumulative rows ≥ cap; then keep all positives, sample negatives to cap, shuffle (same as pre-FE `train_baselines_y30` loaders).

## Train negative downsampling

- **Status:** enabled
- **Requested ratio:** 1:10 (negatives per positive)
- **Original train pool (after row cap, before neg downsampling):** 5,000,000 rows — 4,516 pos / 4,995,484 neg
- **Sampled train (used for LR fit):** 49,676 rows — 4,516 pos / 45,160 neg

## Data summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|-----------|
| Train (used for fit) | 49,676 | 4,516 | 45,160 |
| Val   | 2,000,000 | 9,738 | 1,990,262 |
| Test  | 2,000,000 | 13,708 | 1,986,292 |

**Features:** 55
**Missingness (before imputation):** 43.76%

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

**Best threshold (from val):** 0.79

**Val:**
  - Accuracy: 0.9905
  - Precision: 0.0551
  - Recall: 0.0586
  - F1: 0.0568
  - ROC-AUC: 0.5874
  - PR-AUC: 0.0100
  - Threshold: 0.790
  - TN/FP/FN/TP: 1980475/9787/9167/571


**Test:**
  - Accuracy: 0.9876
  - Precision: 0.0082
  - Recall: 0.0067
  - F1: 0.0074
  - ROC-AUC: 0.4848
  - PR-AUC: 0.0068
  - Threshold: 0.790
  - TN/FP/FN/TP: 1975177/11115/13616/92


**Processing time:** 30.63s

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