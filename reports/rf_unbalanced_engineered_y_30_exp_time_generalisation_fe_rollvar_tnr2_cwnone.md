# Random Forest (engineered data) – exp_time_generalisation

**Generated:** 2026-04-10T16:31:52.844428
- **Label:** `y_30`
- **Data source:** data_engineered
- **Feature set mode:** `rollvar` (same policy as LR: `column_matches_feature_set`)
- **Class weight:** `none`
- **RF:** n_estimators=200, max_depth=None, min_samples_leaf=1, n_jobs=-1
- **Preprocessing:** median imputation + StandardScaler (same as LR engineered runner for fair comparison).
- **Row cap policy:** same as LR (hive-order pool, then keep positives + sample negatives to cap, shuffle).

## Train negative downsampling

- **Status:** enabled
- **Requested ratio:** 1:2 (negatives per positive)
- **Original train pool (after row cap, before neg downsampling):** 5,000,000 rows — 4,516 pos / 4,995,484 neg
- **Sampled train (used for RF fit):** 13,548 rows — 4,516 pos / 9,032 neg

## Data summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|-----------|
| Train (used for fit) | 13,548 | 4,516 | 9,032 |
| Val   | 2,000,000 | 9,738 | 1,990,262 |
| Test  | 2,000,000 | 13,708 | 1,986,292 |

**Features:** 70
**Missingness (before imputation):** 41.80%

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


## Random Forest

**Best threshold (from val, same scan as LR):** 0.62

**Val:**
  - Accuracy: 0.9921
  - Precision: 0.0491
  - Recall: 0.0344
  - F1: 0.0405
  - ROC-AUC: 0.6348
  - PR-AUC: 0.0128
  - Threshold: 0.620
  - TN/FP/FN/TP: 1983778/6484/9403/335


**Test:**
  - Accuracy: 0.9905
  - Precision: 0.0057
  - Recall: 0.0023
  - F1: 0.0032
  - ROC-AUC: 0.5300
  - PR-AUC: 0.0074
  - Threshold: 0.620
  - TN/FP/FN/TP: 1980918/5374/13677/31


**Processing time:** 48.95s

## Feature names (reproducibility)

Sorted column names used in X:

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