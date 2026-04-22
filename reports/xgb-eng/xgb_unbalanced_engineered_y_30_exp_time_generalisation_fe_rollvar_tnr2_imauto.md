# XGBoost (engineered data) – exp_time_generalisation

**Generated:** 2026-04-10T17:07:55.578285
- **Label:** `y_30`
- **Data source:** data_engineered
- **Feature set mode:** `rollvar`
- **Imbalance mode:** `auto` (`none` = no scale_pos_weight; `auto` = neg/pos on final train)
- **scale_pos_weight (used):** `2.0`
- **XGB:** n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
- **Preprocessing:** median imputation + **StandardScaler** (same as LR/RF engineered runners).
- **Row cap policy:** same as LR/RF.

## Train negative downsampling

- **Status:** enabled
- **Requested ratio:** 1:2
- **Train pool before neg downsampling:** 5,000,000 rows — 4,516 pos / 4,995,484 neg
- **Train used for fit:** 13,548 rows — 4,516 pos / 9,032 neg

## Data summary

| Split | Rows | Positives | Negatives |
|-------|------|-----------|-----------|
| Train (fit) | 13,548 | 4,516 | 9,032 |
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


## XGBoost

**Best threshold (from val):** 0.74

**Val:**
  - Accuracy: 0.9909
  - Precision: 0.0399
  - Recall: 0.0380
  - F1: 0.0389
  - ROC-AUC: 0.5964
  - PR-AUC: 0.0110
  - Threshold: 0.740
  - TN/FP/FN/TP: 1981349/8913/9368/370


**Test:**
  - Accuracy: 0.9890
  - Precision: 0.0041
  - Recall: 0.0026
  - F1: 0.0032
  - ROC-AUC: 0.4741
  - PR-AUC: 0.0063
  - Threshold: 0.740
  - TN/FP/FN/TP: 1977891/8401/13673/35


**Processing time:** 34.79s

## Feature names

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