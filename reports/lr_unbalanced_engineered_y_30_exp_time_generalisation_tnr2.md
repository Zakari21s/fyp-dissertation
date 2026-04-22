# LR Engineered (tabular-equivalent + FE) – exp_time_generalisation

**Generated:** 2026-04-10T05:04:59.876701
- **Label:** `y_30`
- **Data source:** data_engineered
- **Feature set mode:** `all` (see `column_matches_feature_set` in script)
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

**Features:** 147
**Missingness (before imputation):** 45.85%

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

**Best threshold (from val):** 0.01

**Val:**
  - Accuracy: 0.9027
  - Precision: 0.0056
  - Recall: 0.1076
  - F1: 0.0107
  - ROC-AUC: 0.5727
  - PR-AUC: 0.0060
  - Threshold: 0.010
  - TN/FP/FN/TP: 1804317/185945/8690/1048


**Test:**
  - Accuracy: 0.9271
  - Precision: 0.0036
  - Recall: 0.0352
  - F1: 0.0066
  - ROC-AUC: 0.4843
  - PR-AUC: 0.0064
  - Threshold: 0.010
  - TN/FP/FN/TP: 1853629/132663/13225/483


**Processing time:** 71.48s

## Feature names (reproducibility)

Sorted column names used in X (same order as training matrix):

```
age_days
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
instab_log1p_r_1
instab_log1p_r_12
instab_log1p_r_170
instab_log1p_r_173
instab_log1p_r_175
instab_log1p_r_177
instab_log1p_r_183
instab_log1p_r_190
instab_log1p_r_194
instab_log1p_r_195
instab_log1p_r_196
instab_log1p_r_241
instab_log1p_r_242
instab_log1p_r_5
instab_log1p_r_9
log1p_r_1
log1p_r_12
log1p_r_170
log1p_r_173
log1p_r_175
log1p_r_177
log1p_r_183
log1p_r_190
log1p_r_194
log1p_r_195
log1p_r_196
log1p_r_241
log1p_r_242
log1p_r_5
log1p_r_9
model_code
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