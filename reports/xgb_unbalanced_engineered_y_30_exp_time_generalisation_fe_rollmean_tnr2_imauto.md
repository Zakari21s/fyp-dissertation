# XGBoost (engineered data) – exp_time_generalisation

**Generated:** 2026-04-10T17:05:42.126174
- **Label:** `y_30`
- **Data source:** data_engineered
- **Feature set mode:** `rollmean`
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


## XGBoost

**Best threshold (from val):** 0.78

**Val:**
  - Accuracy: 0.9933
  - Precision: 0.0673
  - Recall: 0.0298
  - F1: 0.0413
  - ROC-AUC: 0.5963
  - PR-AUC: 0.0116
  - Threshold: 0.780
  - TN/FP/FN/TP: 1986240/4022/9448/290


**Test:**
  - Accuracy: 0.9916
  - Precision: 0.0075
  - Recall: 0.0018
  - F1: 0.0028
  - ROC-AUC: 0.4543
  - PR-AUC: 0.0061
  - Threshold: 0.780
  - TN/FP/FN/TP: 1983106/3186/13684/24


**Processing time:** 33.89s

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