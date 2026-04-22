# Random Forest (engineered data) – exp_time_generalisation

**Generated:** 2026-04-10T16:27:56.948044
- **Label:** `y_30`
- **Data source:** data_engineered
- **Feature set mode:** `rollmean` (same policy as LR: `column_matches_feature_set`)
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
**Missingness (before imputation):** 39.23%

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

**Best threshold (from val, same scan as LR):** 0.61

**Val:**
  - Accuracy: 0.9928
  - Precision: 0.0573
  - Recall: 0.0309
  - F1: 0.0402
  - ROC-AUC: 0.6491
  - PR-AUC: 0.0122
  - Threshold: 0.610
  - TN/FP/FN/TP: 1985313/4949/9437/301


**Test:**
  - Accuracy: 0.9912
  - Precision: 0.0048
  - Recall: 0.0014
  - F1: 0.0022
  - ROC-AUC: 0.5194
  - PR-AUC: 0.0074
  - Threshold: 0.610
  - TN/FP/FN/TP: 1982378/3914/13689/19


**Processing time:** 41.31s

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