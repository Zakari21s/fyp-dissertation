# Data Cleaning Specification
## AI-Driven Disk Failure Prediction Using Alibaba DCBrain SMART Data

### 1. Objectives

The primary objective of this data cleaning process is to prepare the Alibaba DCBrain SMART dataset for binary classification tasks, where the target variable distinguishes between:
- **Positive class**: Disks that will fail within a specified prediction window
- **Negative class**: Disks that remain operational

Secondary objectives include:
- Ensuring data consistency across multiple daily CSV files (2018-2019, ~360 files/year)
- Establishing a reliable temporal structure for time-series analysis
- Preserving data integrity while removing invalid or problematic records
- Creating reproducible cleaning pipelines for future data ingestion

### 2. Assumptions About Dataset Structure

Based on the Alibaba DCBrain SMART dataset characteristics, we assume:

**File Structure:**
- Daily CSV files with consistent naming convention (e.g., `YYYY-MM-DD.csv` or similar)
- Each file contains SMART attribute readings for multiple disk drives
- Files span calendar years 2018 and 2019 (~360 files per year)

**Data Schema:**
- Each row represents a single disk observation at a specific timestamp
- Key identifiers: disk serial number, model, and timestamp/date
- SMART attributes: numeric sensor readings (e.g., temperature, reallocated sectors, power-on hours)
- Target variable: failure indicator (binary or derived from failure date)
- Potential metadata: disk age, installation date, or failure date

**Temporal Characteristics:**
- Observations are recorded daily or at regular intervals
- Multiple observations per disk across time (longitudinal data)
- Failure events are timestamped or can be inferred from data patterns

**Data Volume:**
- Total dataset: >2 million rows across all files
- High cardinality: thousands of unique disk serial numbers
- Multiple disk models with varying SMART attribute availability

### 3. Key Data Quality Risks

#### 3.1 Missingness
- **Risk**: Incomplete SMART attribute readings due to sensor failures, data collection gaps, or disk disconnection
- **Impact**: Reduces feature availability, may bias failure prediction if missingness correlates with failure
- **Detection**: Per-attribute and per-record missing value analysis

#### 3.2 Duplicates
- **Risk**: Identical records appearing multiple times (same disk, same timestamp)
- **Impact**: Inflates dataset size, may cause data leakage in train/test splits
- **Detection**: Exact duplicate detection on key fields (serial, timestamp, attributes)

#### 3.3 Inconsistent Serial/Model Information
- **Risk**: Same disk serial number with different model identifiers, or serial number format inconsistencies
- **Impact**: Incorrect disk grouping, model-specific feature engineering failures
- **Detection**: Serial number format validation, model-serial consistency checks

#### 3.4 Outliers
- **Risk**: Extreme SMART attribute values due to sensor errors, data corruption, or genuine hardware anomalies
- **Impact**: Skews statistical distributions, may mislead model training
- **Detection**: Statistical outlier detection (IQR, Z-score) and domain knowledge validation

#### 3.5 Data Leakage
- **Risk**: Future information present in historical records (e.g., failure date in pre-failure observations)
- **Impact**: Overly optimistic model performance, non-generalizable predictions
- **Detection**: Temporal validation of feature-target relationships, exclusion of post-failure data from pre-failure windows

#### 3.6 Temporal Inconsistencies
- **Risk**: Out-of-order timestamps, gaps in time series, or inconsistent date formats
- **Impact**: Incorrect time-series feature engineering, invalid prediction windows
- **Detection**: Chronological ordering validation, timestamp format consistency checks

#### 3.7 Data Type Mismatches
- **Risk**: Numeric fields stored as strings, incorrect encoding, or mixed formats
- **Impact**: Feature extraction failures, type conversion errors
- **Detection**: Schema validation, type inference and conversion

### 4. Cleaning Decisions and Rationale

#### 4.1 Missing Value Handling

**Decision**: Multi-strategy approach based on missingness pattern:
- **Low missingness (<5% per attribute)**: Forward-fill for time-series continuity, then median imputation
- **High missingness (≥5% per attribute)**: Flag for exclusion or treat as separate "missing" category
- **Complete record missingness**: Remove records with >50% missing SMART attributes

**Rationale**: Preserves temporal continuity for time-series models while avoiding excessive imputation bias. Thresholds balance data retention with quality.

#### 4.2 Duplicate Removal

**Decision**: Remove exact duplicates based on (serial_number, timestamp, all_SMART_attributes). Retain first occurrence.

**Rationale**: Prevents data inflation and ensures one observation per disk per time point. First-occurrence retention maintains chronological order.

#### 4.3 Serial Number and Model Standardization

**Decision**:
- Normalize serial numbers: uppercase, remove whitespace, validate format
- Create canonical model mapping for variant spellings/abbreviations
- Flag or exclude records with serial-model mismatches across time

**Rationale**: Ensures consistent disk identity tracking and enables model-specific feature engineering.

#### 4.4 Outlier Treatment

**Decision**: 
- **Domain-based bounds**: Apply SMART attribute-specific valid ranges (e.g., temperature: -40°C to 100°C)
- **Statistical bounds**: Cap extreme values at 3 standard deviations from mean (per attribute, per model if applicable)
- **Flagging**: Retain outlier flags as potential features rather than removing records

**Rationale**: Preserves information while preventing extreme values from distorting model training. Outlier flags may be predictive of failure.

#### 4.5 Data Leakage Prevention

**Decision**:
- Strict temporal splitting: training data must precede test data chronologically
- Remove any features derived from post-failure information
- Exclude observations from the prediction window if failure date is known
- Validate that no SMART attributes contain future failure indicators

**Rationale**: Ensures realistic prediction scenarios and generalizable model performance.

#### 4.6 Temporal Alignment

**Decision**:
- Standardize timestamp format (ISO 8601: YYYY-MM-DD HH:MM:SS)
- Sort by (serial_number, timestamp) to ensure chronological order
- Identify and flag temporal gaps >30 days for potential exclusion from time-series features

**Rationale**: Enables accurate time-series feature engineering and prediction window calculations.

#### 4.7 Data Type Standardization

**Decision**:
- Convert all numeric SMART attributes to float64
- Standardize categorical fields (model, serial) to string type
- Validate and convert timestamps to datetime objects

**Rationale**: Ensures compatibility with machine learning libraries and prevents type-related errors.

### 5. Outputs Produced

#### 5.1 Cleaned Dataset
- **Format**: Parquet files (partitioned by year/month for efficient access)
- **Structure**: Single consolidated dataset with standardized schema
- **Metadata**: Data dictionary documenting all fields, types, and cleaning transformations applied

#### 5.2 Quality Assurance (QA) Report
- **Summary statistics**: Row counts (before/after cleaning), missing value percentages, duplicate counts
- **Data quality metrics**: Completeness score, consistency score, validity score
- **Visualizations**: Missing value heatmaps, temporal coverage plots, attribute distributions
- **Issue log**: Detailed records of all data quality issues identified and resolved

#### 5.3 Cleaning Logs
- **Execution log**: Timestamped record of all cleaning operations performed
- **Decision log**: Rationale for each cleaning decision, including manual interventions
- **Error log**: Any failures or warnings encountered during processing
- **Performance metrics**: Processing time, memory usage, file I/O statistics

#### 5.4 Validation Artifacts
- **Schema validation report**: Confirmation of expected data types and constraints
- **Temporal validation report**: Chronological ordering verification, gap analysis
- **Leakage detection report**: Confirmation of no temporal leakage in features

### 6. Acceptance Criteria

The data cleaning process is considered successful when all of the following criteria are met:

#### 6.1 Completeness
- **Criterion**: ≥95% of original records retained after cleaning (excluding intentional exclusions for quality reasons)
- **Measurement**: (Cleaned rows / Original rows) × 100%

#### 6.2 Data Quality Metrics
- **Criterion**: 
  - Missing value rate <5% for critical SMART attributes
  - Zero exact duplicates in final dataset
  - Serial number consistency: 100% of disks have consistent model associations
- **Measurement**: Automated quality checks in QA report

#### 6.3 Temporal Integrity
- **Criterion**: 
  - All timestamps in valid format and chronological order
  - No temporal gaps >30 days without explicit flagging
  - Prediction windows correctly defined (no leakage)
- **Measurement**: Temporal validation report passes all checks

#### 6.4 Schema Consistency
- **Criterion**: 
  - All numeric fields are numeric type (no string contamination)
  - All required fields present in every record
  - Data dictionary matches actual schema
- **Measurement**: Schema validation passes with zero errors

#### 6.5 Reproducibility
- **Criterion**: 
  - Cleaning pipeline can be re-executed on raw data to produce identical results
  - All transformations are logged and version-controlled
  - Random seed fixed for any stochastic operations
- **Measurement**: Re-run pipeline produces bit-identical output (or documented acceptable variance)

#### 6.6 Documentation Completeness
- **Criterion**: 
  - QA report includes all required sections and visualizations
  - Cleaning logs are comprehensive and human-readable
  - Data dictionary is complete and accurate
- **Measurement**: Manual review confirms all documentation artifacts are present and correct

#### 6.7 Model Readiness
- **Criterion**: 
  - Cleaned dataset can be directly ingested by binary classification pipeline
  - Target variable is correctly defined and balanced (or imbalance documented)
  - Feature engineering pipeline can execute without errors
- **Measurement**: Successful execution of downstream model training script on cleaned data

---

**Document Version**: 1.0  
**Last Updated**: [Date]  
**Author**: [Your Name]  
**Project**: AI-Driven Disk Failure Prediction - FYP

