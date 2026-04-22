# Cleaning Pipeline Design: Step 4
## Labeled Dataset Quality Assurance and Standardization

### 1. Goals

**Primary objectives:**
- **Quality**: Ensure data quality standards for machine learning model training
- **Consistency**: Standardize schema, data types, and formats across partitioned datasets
- **Reproducibility**: Create deterministic, version-controlled cleaning pipeline

**Success criteria:**
- All partitions have identical schema and column ordering
- Data types are consistent and ML-ready (numeric features, proper date types)
- No duplicate (disk_id, smart_day) pairs
- Missingness and outlier policies documented and applied consistently
- QA metrics meet acceptance thresholds

### 2. Inputs and Outputs

**Inputs:**
- `data_clean/labeled_smartlog2018ssd/` (partitioned parquet: `year=YYYY/month=MM/data.parquet`)
- `data_clean/labeled_smartlog2019ssd/` (partitioned parquet: `year=YYYY/month=MM/data.parquet`)

**Input schema (expected):**
- Identifiers: `disk_id`, `ds`, `smart_day`, `failure_date`
- Labels: `y_7`, `y_14`, `y_30`
- SMART features: `n_*`, `r_*` columns (variable presence across partitions)
- Metadata: `model` (optional)

**Outputs:**
- `data_clean/cleaned_smartlog2018ssd/` (partitioned parquet, same structure)
- `data_clean/cleaned_smartlog2019ssd/` (partitioned parquet, same structure)
- `reports/cleaning_stage_<N>_<dataset>.json` (per-stage reports)
- `reports/cleaning_qa_<dataset>.md` (final QA report)
- `logs/cleaning_<dataset>.log` (processing logs)

### 3. Cleaning Stages

#### Stage 1: Schema Standardization

**Objective:** Ensure all partitions share a common, consistent column set.

**Process:**
1. Scan all partitions to identify the union of all columns across partitions
2. Define canonical column ordering: identifiers → labels → SMART features (sorted) → metadata
3. For each partition:
   - Add missing columns with `NaN` values
   - Reorder columns to match canonical schema
   - Ensure column names are consistent (no case/whitespace variations)

**Rationale:**
- Machine learning frameworks require consistent feature sets across train/test splits
- Prevents downstream errors from missing columns in specific partitions
- Enables efficient batch processing and feature engineering

**Output:** Standardized schema manifest (JSON) documenting column set and ordering.

---

#### Stage 2: Data Type Fixing

**Objective:** Ensure all columns have correct, ML-compatible data types.

**Type assignments:**
- `disk_id`: `int64` (standardize from mixed int/string)
- `ds`: `string` or `int64` (preserve original format)
- `smart_day`: `date` or `string` (ISO format: YYYY-MM-DD)
- `failure_date`: `date` or `string` (ISO format: YYYY-MM-DD)
- `y_7`, `y_14`, `y_30`: `int8` (binary labels: 0 or 1)
- `model`: `string` (categorical)
- SMART features (`n_*`, `r_*`): `float64` (numeric, allow NaN for missing)

**Process:**
1. Convert SMART features to numeric (coerce errors to NaN)
2. Validate label columns are binary (0/1), convert if needed
3. Standardize date columns to consistent format
4. Type coercion with error logging for failed conversions

**Rationale:**
- Prevents runtime errors during model training
- Optimizes memory usage (int8 for labels vs int64)
- Ensures numeric operations work correctly on SMART features

**Output:** Type conversion report (columns converted, errors encountered).

---

#### Stage 3: Deduplication Safety Check

**Objective:** Verify and enforce uniqueness of (disk_id, smart_day) pairs.

**Process:**
1. For each partition, check for duplicate (disk_id, smart_day) combinations
2. If duplicates found:
   - Log duplicate count and examples
   - Keep first occurrence (by row order)
   - Report duplicate rate per partition

**Rationale:**
- Labeling pipeline should have removed duplicates, but safety check ensures data integrity
- Prevents data leakage from duplicate samples in train/test splits
- Maintains one observation per disk per day

**Output:** Deduplication report (duplicate counts per partition, if any).

---

#### Stage 4: Missingness Policy

**Objective:** Apply consistent missing value handling strategy.

**Policy:**
1. **Column-level filtering:**
   - Compute missingness percentage per SMART feature column across entire dataset
   - Drop columns with >95% missingness (insufficient signal)
   - Log dropped columns and rationale

2. **Row-level filtering (optional):**
   - Flag rows with >50% missing SMART features (configurable threshold)
   - Option to drop or retain based on downstream requirements

3. **Imputation (deferred to training phase):**
   - Forward-fill per disk (temporal continuity): Apply during feature engineering, train-only
   - Median imputation: Compute medians on training set only, apply to train/test
   - **Critical**: No imputation in this cleaning stage to prevent leakage

**Rationale:**
- Extreme missingness (>95%) indicates sensor/attribute not available for most disks
- Forward-fill preserves temporal patterns for time-series models
- Train-only statistics prevent data leakage (test set statistics must not influence training)

**Output:** Missingness report (per-column percentages, dropped columns, row-level statistics).

---

#### Stage 5: Outlier Policy

**Objective:** Document and optionally handle extreme values in SMART features.

**Policy options:**
1. **No outlier treatment** (default):
   - Retain all values as-is
   - Log extreme values for analysis
   - Let model learn robust representations

2. **Quantile clipping** (optional, train-only):
   - Compute 1st and 99th percentiles on training set only
   - Clip values outside [Q1, Q99] to boundaries
   - Apply same boundaries to test set
   - **Critical**: Percentiles computed on train-only to prevent leakage

**Rationale:**
- Outliers may be genuine hardware anomalies (predictive signal)
- Clipping can stabilize training but may remove important failure signals
- Train-only statistics maintain temporal integrity

**Output:** Outlier analysis report (extreme values per column, clipping decisions if applied).

---

#### Stage 6: Invalid Record Filtering

**Objective:** Remove records that violate domain constraints.

**Filters:**
1. **Missing identifiers:**
   - Drop rows with null `disk_id` or `smart_day`
   - Drop rows with invalid date formats

2. **Invalid labels:**
   - Ensure `y_H ∈ {0, 1}` for all horizons
   - Drop rows with null labels (should not occur, but safety check)

3. **Domain constraints (optional):**
   - SMART attribute value ranges (e.g., temperature: -40°C to 100°C)
   - Negative counts (reallocated sectors, etc.) should be non-negative
   - Flag or drop violations based on severity

**Rationale:**
- Invalid identifiers break joins and grouping operations
- Invalid labels corrupt training objectives
- Domain constraints catch data corruption or parsing errors

**Output:** Invalid record report (counts by filter type, examples of violations).

---

#### Stage 7: QA Summary and Acceptance Criteria

**Objective:** Generate comprehensive quality assurance report and validate acceptance criteria.

**QA metrics:**
1. **Schema consistency:** All partitions have identical column set and ordering
2. **Type consistency:** All columns have expected data types
3. **Uniqueness:** Zero duplicate (disk_id, smart_day) pairs
4. **Missingness:** Column missingness <95% (dropped if exceeded)
5. **Label distribution:** Label counts match expected ranges (severe imbalance expected)
6. **Date range:** smart_day within expected bounds (2018-01-01 to 2019-12-31)
7. **Row counts:** Output row count matches input minus filtered rows

**Acceptance criteria:**
- ✅ Schema standardized across all partitions
- ✅ Data types correct and consistent
- ✅ Zero duplicates (disk_id, smart_day)
- ✅ No columns with >95% missingness (or documented rationale for retention)
- ✅ All labels are binary (0 or 1)
- ✅ Date ranges valid and consistent
- ✅ Processing logs complete with no fatal errors

**Output:** Final QA report (Markdown) with metrics, acceptance status, and recommendations.

### 4. Data Leakage Prevention

**Critical constraints:**
- **No test set statistics:** All summary statistics (medians, quantiles, means) computed on training partitions only
- **Temporal ordering:** Training partitions (2018) processed before test partitions (2019)
- **Feature exclusion:** `failure_date`, `days_to_failure`, and any failure-derived fields must not appear as features
- **Imputation boundaries:** Forward-fill and median imputation use training-set statistics only

**Implementation notes:**
- Stage 4 and Stage 5 explicitly defer imputation/outlier treatment to training phase
- Cleaning pipeline focuses on schema, types, and basic filtering only
- Advanced imputation applied during feature engineering with train/test split awareness

### 5. Deliverables

**Per-stage reports:**
- `reports/cleaning_stage_1_schema_<dataset>.json`
- `reports/cleaning_stage_2_dtypes_<dataset>.json`
- `reports/cleaning_stage_3_dedup_<dataset>.json`
- `reports/cleaning_stage_4_missingness_<dataset>.json`
- `reports/cleaning_stage_5_outliers_<dataset>.json`
- `reports/cleaning_stage_6_invalid_<dataset>.json`

**Final deliverables:**
- `reports/cleaning_qa_<dataset>.md` (human-readable QA summary)
- `reports/cleaning_qa_<dataset>.json` (machine-readable QA metrics)
- `logs/cleaning_<dataset>.log` (detailed processing logs)

**Cleaned datasets:**
- `data_clean/cleaned_smartlog2018ssd/` (partitioned parquet)
- `data_clean/cleaned_smartlog2019ssd/` (partitioned parquet)

---

## Summary

This cleaning pipeline transforms labeled parquet datasets into ML-ready format through seven sequential stages: schema standardization, type fixing, deduplication checks, missingness policies, outlier handling, invalid record filtering, and comprehensive QA. The design prioritizes reproducibility, prevents data leakage, and maintains temporal integrity while preparing data for downstream feature engineering and model training.

