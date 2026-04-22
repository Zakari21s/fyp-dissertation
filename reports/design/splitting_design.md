# Time-Based Dataset Splitting Design

**Document Version:** 1.0  
**Last Updated:** 2026-02-06  
**Author:** Data Engineering Team  
**Project:** AI-Driven Disk Failure Prediction - FYP

## Overview

This document describes the time-based train/validation/test splitting strategy for the disk failure prediction project. The splitting process is **non-destructive** - it reads from the final cleaned dataset (Stage 7 output) and creates new split datasets without modifying the source data.

## Objectives

1. **Temporal Generalization**: Evaluate model performance on future time periods to simulate real-world deployment scenarios
2. **Data Leakage Prevention**: Ensure strict entity-level disjointness between splits to prevent information leakage
3. **Reproducibility**: Create consistent, deterministic splits that can be regenerated from cleaned data

## Split Policy

### Time Windows

- **TRAIN**: All rows with `smart_day` in year 2018
- **VAL**: 2019-01-01 to 2019-06-30 (inclusive)
- **TEST**: 2019-07-01 to 2019-12-31 (inclusive)

### Rationale

- **Temporal Progression**: Train on historical data (2018), validate on first half of next year (2019 H1), test on second half (2019 H2)
- **Realistic Evaluation**: Simulates deployment where model is trained on past data and evaluated on future data
- **Sufficient Data**: Each split contains enough data for meaningful evaluation while maintaining temporal separation

## Entity Definition

**Entity ID** = `(disk_id, model)`

This composite key represents a unique disk entity because:
- `disk_id` alone may not be unique (same disk_id can appear with different models)
- Model reuse across different physical disks requires the combination to uniquely identify a disk

## Entity Disjointness Enforcement

To prevent data leakage, we enforce strict entity-level disjointness:

1. **TRAIN Exclusion**: Remove from TRAIN any entity `(disk_id, model)` that appears in VAL or TEST
   - **Rationale**: Prevents the model from seeing any information about disks that will be evaluated
   - **Impact**: Ensures training data contains only entities that will never appear in validation or test sets

2. **VAL Exclusion**: Remove from VAL any entity that appears in TEST
   - **Rationale**: Prevents validation set from containing entities that will be used for final evaluation
   - **Impact**: Ensures validation set is truly independent from test set

3. **TEST**: No exclusions applied
   - **Rationale**: Test set represents the final evaluation scenario and should remain unchanged
   - **Impact**: Test set contains all entities in its time range, providing realistic evaluation

### Why Entity Disjointness Matters

Without entity disjointness, a model could:
- **Memorize disk-specific patterns**: Learn characteristics of specific disks that appear in multiple splits
- **Overfit to entity-level features**: Exploit patterns that are specific to individual disks rather than generalizable
- **Inflate performance metrics**: Achieve artificially high performance that doesn't generalize to new disks

By enforcing entity disjointness, we ensure:
- **True generalization**: Model must learn patterns that generalize across different disk entities
- **Realistic evaluation**: Performance metrics reflect true predictive capability on unseen disks
- **Fair comparison**: Different models are evaluated on the same entity-disjoint splits

## Implementation Approach

### Multi-Pass Processing

The splitting process uses a memory-efficient multi-pass approach:

#### Pass 1: Entity Set Building
- Scan VAL time range (2019-01-01 to 2019-06-30) to build set of entity IDs
- Scan TEST time range (2019-07-01 to 2019-12-31) to build set of entity IDs
- Store entity sets in memory (reasonable size: ~100K-500K entities)

#### Pass 2: TRAIN Partition Writing
- Iterate through 2018 partitions
- Filter rows by date (2018-01-01 to 2018-12-31)
- Exclude rows where `(disk_id, model)` is in VAL or TEST entity sets
- Write filtered partitions to output

#### Pass 3: VAL Partition Writing
- Iterate through 2019 partitions
- Filter rows by date (2019-01-01 to 2019-06-30)
- Exclude rows where `(disk_id, model)` is in TEST entity set
- Write filtered partitions to output

#### Pass 4: TEST Partition Writing
- Iterate through 2019 partitions
- Filter rows by date (2019-07-01 to 2019-12-31)
- No entity exclusions applied
- Write filtered partitions to output

### Memory Safety

- **Partition-by-partition processing**: Never load entire dataset into memory
- **File-by-file processing**: Process individual parquet files within partitions
- **Entity set size**: Entity sets are small enough to fit in memory (~100K-500K tuples)
- **Streaming writes**: Write output partitions incrementally

## Input/Output Specification

### Input (Read-Only)

**Source**: Stage 7 cleaned output directory

**Location Detection** (in order):
1. `data_interim/clean_stage7_<dataset>/`
2. `data_clean/clean_stage7_<dataset>/`

**Structure**: Partitioned parquet files
```
<stage7_dir>/
  year=2018/
    month=01/
      data.parquet
    month=02/
      data.parquet
    ...
  year=2019/
    month=01/
      data.parquet
    ...
```

**Required Columns**:
- `disk_id` (nullable int)
- `model` (string)
- `smart_day` (datetime)
- Labels: `y_7`, `y_14`, `y_30` (int8)
- SMART features: `n_*` and `r_*` (float64)

### Output (Write-Only)

**Location**: `data_splits/exp_time_generalisation/<dataset>/<split>/`

**Structure**: Partitioned parquet files
```
data_splits/
  exp_time_generalisation/
    <dataset>/
      train/
        year=2018/
          month=01/
            data.parquet
          ...
      val/
        year=2019/
          month=01/
            data.parquet
          ...
      test/
        year=2019/
          month=07/
            data.parquet
          ...
```

**Format**: PyArrow parquet files with same schema as input

## Reports

### JSON Report

**Location**: `reports/splits_summary_<dataset>.json`

**Contents**:
- Input statistics (total rows read)
- Per-split statistics:
  - Rows before/after overlap removal
  - Entities before/after overlap removal
  - Date ranges
  - Label distributions (y_7, y_14, y_30)
- Processing time

### Markdown Report

**Location**: `reports/splits_summary_<dataset>.md`

**Contents**:
- Overview and split definitions
- Entity disjointness explanation
- Detailed per-split statistics
- Label distribution tables

## Usage

### Command Line Interface

```bash
# Default: smartlog2018ssd for train, smartlog2019ssd for eval
python -m src.make_time_splits

# Explicit dataset specification
python -m src.make_time_splits \
  --train_dataset smartlog2018ssd \
  --eval_dataset smartlog2019ssd

# Overwrite existing output
python -m src.make_time_splits --overwrite
```

### Arguments

- `--train_dataset`: Dataset name for training data (default: `smartlog2018ssd`)
- `--eval_dataset`: Dataset name for validation/test data (default: `smartlog2019ssd`)
- `--overwrite`: Delete existing output directory before writing (default: False)

## Validation

### Post-Split Checks

After splitting, verify:

1. **Entity Disjointness**:
   - TRAIN entities ∩ VAL entities = ∅
   - TRAIN entities ∩ TEST entities = ∅
   - VAL entities ∩ TEST entities = ∅

2. **Temporal Separation**:
   - All TRAIN rows have `smart_day` in 2018
   - All VAL rows have `smart_day` in 2019-01-01 to 2019-06-30
   - All TEST rows have `smart_day` in 2019-07-01 to 2019-12-31

3. **Data Integrity**:
   - All required columns present
   - No data corruption
   - Row counts match expected ranges

4. **Label Distribution**:
   - Positive labels present in all splits
   - Label nesting constraint satisfied (y_7 ≤ y_14 ≤ y_30)

## Performance Considerations

### Expected Runtime

- **Entity set building**: ~5-10 minutes (depends on dataset size)
- **Partition writing**: ~10-30 minutes (depends on dataset size and overlap)
- **Total**: ~15-40 minutes for typical dataset sizes

### Scalability

- **Memory**: O(E) where E = number of unique entities (~100K-500K)
- **Disk I/O**: O(N) where N = total number of rows
- **Time**: O(N) - linear in dataset size

## Error Handling

### Common Issues

1. **Stage 7 output not found**:
   - Error: `FileNotFoundError` with checked paths
   - Solution: Ensure Stage 7 has been run for both datasets

2. **Missing required columns**:
   - Error: Warning logged, partition skipped
   - Solution: Verify Stage 7 output schema

3. **Date parsing errors**:
   - Error: Warning logged, rows with invalid dates excluded
   - Solution: Verify `smart_day` column format

4. **Output directory exists**:
   - Error: Script fails if output exists and `--overwrite` not specified
   - Solution: Use `--overwrite` flag or manually delete output directory

## Integration with Pipeline

### Prerequisites

- Stage 7 QA must be completed for both train and eval datasets
- Stage 7 output must be available in expected location

### Dependencies

- Input: Stage 7 cleaned parquet files
- Output: Split parquet files for model training

### Downstream Usage

Split datasets are used by:
- Model training scripts
- Feature engineering pipelines
- Evaluation scripts

## Future Enhancements

Potential improvements:

1. **Stratified Splitting**: Ensure label distribution balance across splits
2. **Cross-Validation**: Support k-fold temporal cross-validation
3. **Entity Sampling**: Option to sample entities for faster iteration
4. **Validation Scripts**: Automated post-split validation checks
5. **Visualization**: Plots showing entity overlap and temporal distribution

## References

- **Cleaning Pipeline Design**: `reports/cleaning_pipeline_design.md`
- **Data Cleaning Specification**: `reports/Data_Cleaning_Specification.md`
- **Implementation**: `src/make_time_splits.py`

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-06  
**Status**: Final

