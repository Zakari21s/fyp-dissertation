# Labeling Strategy Design
## Binary Classification for Disk Failure Prediction

### 1. Unit of Observation

Each training sample represents a **single disk on a single day**, formally defined as:

```
sample = (disk_id, smart_day)
```

where:
- `disk_id`: Unique identifier for the disk drive
- `smart_day`: Date of the SMART log observation (from `ds` column, format: YYYYMMDD)

This granularity enables:
- **Temporal modeling**: Captures day-to-day changes in SMART attributes that may signal impending failure
- **Sufficient data volume**: With ~135M rows across 2018-2019, provides ample training samples
- **Practical prediction**: Aligns with daily monitoring schedules in production data centers

### 2. Prediction Horizon

We propose three prediction horizons: **H ∈ {7, 14, 30} days**.

**Justification:**

- **7 days (1 week)**: 
  - Enables proactive maintenance scheduling within a single maintenance cycle
  - Provides sufficient lead time for data migration and replacement planning
  - Balances early warning with actionable prediction window

- **14 days (2 weeks)**:
  - Allows for more flexible maintenance scheduling and resource allocation
  - Captures medium-term degradation patterns that may not be visible in shorter horizons
  - Reduces false positives from transient anomalies

- **30 days (1 month)**:
  - Enables strategic capacity planning and bulk procurement
  - Captures long-term degradation trends
  - Provides maximum lead time for critical infrastructure

**Selection rationale:**
These horizons span the range from immediate operational needs (7 days) to strategic planning (30 days), allowing evaluation of model performance across different use cases. The progression also enables analysis of how prediction accuracy degrades with increasing horizon length, which is expected in time-series forecasting tasks.

### 3. Label Definition

For each sample `(disk_id, smart_day)`, we compute:

```
days_to_failure = failure_date - smart_day
```

where `failure_date` is extracted from the `failure_time` column in the failure labels dataset, truncated to the date component.

The binary label for horizon H is defined as:

```
y_H = {
    1  if  0 ≤ days_to_failure ≤ H
    0  otherwise
}
```

**Interpretation:**
- **y_H = 1 (positive class)**: The disk will fail within H days from the observation date
- **y_H = 0 (negative class)**: The disk will not fail within H days (either it fails later, or never fails during the observation period)

**Special cases:**
- **No failure record**: If `disk_id` has no entry in the failure labels, `y_H = 0` for all samples
- **Post-failure observations**: Samples where `days_to_failure < 0` are excluded (see Section 5)
- **Future failures beyond H**: Samples where `days_to_failure > H` are labeled as `y_H = 0`

### 4. Required Columns

#### 4.1 SMART Logs Dataset

**Required columns:**
- `disk_id` (int64): Disk identifier for joining with failure labels
- `ds` (string/int, format: YYYYMMDD): Date of SMART log observation
- `model` (string): Disk model identifier (optional, for model-specific analysis)
- All SMART attribute columns: `n_*` and `r_*` columns (e.g., `n_1`, `n_5`, `r_1`, `r_5`, etc.)

**Usage:**
- `disk_id` and `ds` define the unit of observation
- SMART attributes serve as features for the classification model
- `model` may be used for stratified analysis or model-specific feature engineering

#### 4.2 Failure Labels Dataset

**Required columns:**
- `disk_id` (int64): Disk identifier for joining with SMART logs
- `failure_time` (datetime): Timestamp of disk failure (parsed from original string format)

**Usage:**
- `disk_id` enables left join with SMART logs
- `failure_time` is used to compute `days_to_failure` for label assignment
- **Critical**: `failure_time` and any derived fields (e.g., `days_to_failure`) must **never** be used as features (see Section 5.2)

### 5. Leakage Prevention

#### 5.1 Post-Failure Data Exclusion

**Rule:** All samples where `days_to_failure < 0` must be removed from the training dataset.

**Rationale:**
- These observations occur **after** the disk has already failed
- Post-failure SMART readings may reflect abnormal states (e.g., complete drive failure, error codes) that are not predictive but rather diagnostic
- Including post-failure data would create a trivial prediction task: "if the disk has already failed, predict it will fail"
- This violates the temporal ordering required for realistic prediction scenarios

**Implementation note:** After computing `days_to_failure`, filter: `df = df[df['days_to_failure'] >= 0]`

#### 5.2 Feature Exclusion

**Prohibited features:**
- `failure_time`: Direct failure timestamp
- `days_to_failure`: Computed time-to-failure
- Any binary indicator derived from failure status (e.g., `has_failure_record`, `is_failed`)

**Rationale:**
- These fields contain information that would not be available at prediction time
- Using `days_to_failure` as a feature would make the task trivial: the model would learn to predict "failure in H days" by directly reading "failure in X days"
- The model must learn to predict failure from SMART attributes alone, not from failure metadata

**Allowed features:**
- All SMART attributes (`n_*`, `r_*` columns)
- `disk_id` (for grouping/identification, not as a direct feature)
- `ds` (for temporal feature engineering, e.g., day-of-week, but not as direct failure predictor)
- `model` (for categorical encoding if used in feature engineering)

#### 5.3 Temporal Leakage Prevention

**Temporal ordering principle:** The model must only use information available **at or before** the observation time `smart_day`.

**Mechanisms:**
1. **Strict temporal split**: Train/test splits must be chronological (e.g., train on 2018 data, test on 2019 data)
2. **No future information**: When computing features for a sample at time `t`, only use SMART logs from dates ≤ `t`
3. **Label computation**: `days_to_failure` is computed using `failure_time` (known only after failure), but this is acceptable because:
   - Labels are used only for training, not as features
   - At inference time, labels are not available (we predict them)
   - The label computation respects temporal ordering: we only label past observations with future failure information

**Validation:**
- Ensure train/test split dates: `max(train_ds) < min(test_ds)`
- Verify that feature engineering for sample at time `t` uses only data from `≤ t`
- Cross-validation must use time-series splits, not random splits

### 6. Class Imbalance Expectations

**Expected imbalance:**
Given the dataset characteristics:
- Total SMART log observations: ~135M+ rows (2018-2019)
- Unique failed disks: ~15,823
- Average observations per disk: ~8,500+ (assuming ~2 years of daily logs)

The positive class (y_H = 1) is expected to be **highly imbalanced**, with positive samples representing a small fraction of total observations. For example:
- For H = 7 days: Each failed disk contributes at most 7 positive samples
- For H = 30 days: Each failed disk contributes at most 30 positive samples
- Estimated positive class ratio: < 0.1% to ~0.2% depending on horizon

**Implications:**
- Standard accuracy metrics will be misleading (e.g., 99.9% accuracy achieved by predicting all negatives)
- Appropriate metrics: Precision-Recall curve, F1-score, ROC-AUC, or cost-sensitive metrics
- Mitigation strategies (to be implemented later):
  - Class weighting in loss functions
  - Resampling techniques (SMOTE, undersampling, etc.)
  - Ensemble methods
  - Threshold tuning for precision-recall trade-off

**Note:** Class imbalance mitigation will be addressed during model development and is outside the scope of this labeling design document.

---

## Summary

This labeling strategy defines:
- **Unit**: (disk_id, day) pairs from SMART logs
- **Horizons**: 7, 14, 30 days for multi-horizon evaluation
- **Labels**: Binary classification based on `days_to_failure ≤ H`
- **Data sources**: SMART logs (features) + failure labels (targets)
- **Leakage prevention**: Post-failure exclusion, feature restrictions, temporal ordering
- **Imbalance**: Expected severe imbalance, mitigation deferred to modeling phase

This design ensures a realistic, temporally-consistent labeling approach suitable for production deployment scenarios.

