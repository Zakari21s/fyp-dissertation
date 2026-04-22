# Cleaning Stage 6: Invalid Record Filtering + Label/Key Sanity - smartlog2018ssd

**Generated:** 2026-02-06T18:30:19.747485

## Overview

- **Dataset:** smartlog2018ssd
- **Total partitions:** 12
- **Partitions processed:** 12
- **Partitions failed:** 0
- **Processing time:** 182.48 seconds

## Global Row Counts

- **Rows in:** 133,559,111
- **Rows out:** 133,559,111
- **Rows dropped:** 0
- **% dropped:** 0.0000%

## Drop Counts by Rule

| Rule | Count |
|------|-------|
| Missing disk_id | 0 |
| Missing model | 0 |
| Missing smart_day | 0 |
| Missing ds | 0 |
| Invalid labels | 0 |

## Corrections (Values Set to NaN)

- **Total r_* negative values corrected:** 0
- **Total n_* out-of-range values corrected:** 50,601,882

### Top 15 n_* Columns with Out-of-Range Values Fixed to NaN

| Column | Count |
|--------|-------|
| `n_195` | 27,093,647 |
| `n_1` | 11,755,131 |
| `n_180` | 11,753,104 |

## Key Sanity Check

**Total duplicate keys found:** 0

**Note:** Duplicates are reported for monitoring only. Deduplication was already performed in Stage 3.

| Partition | Duplicate Keys |
|-----------|----------------|

## Per-Month Breakdown

| Partition | Rows In | Rows Out | Missing disk_id | Missing model | Missing smart_day | Missing ds | Invalid Labels |
|-----------|---------|----------|-----------------|---------------|-------------------|------------|----------------|
| `year=2018/month=01` | 10,529,353 | 10,529,353 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=02` | 9,668,382 | 9,668,382 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=03` | 10,947,243 | 10,947,243 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=04` | 10,447,825 | 10,447,825 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=05` | 11,314,286 | 11,314,286 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=06` | 11,203,125 | 11,203,125 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=07` | 11,647,453 | 11,647,453 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=08` | 12,016,685 | 12,016,685 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=09` | 11,633,245 | 11,633,245 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=10` | 11,772,193 | 11,772,193 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=11` | 10,304,367 | 10,304,367 | 0 | 0 | 0 | 0 | 0 |
| `year=2018/month=12` | 12,074,954 | 12,074,954 | 0 | 0 | 0 | 0 | 0 |

## Important Notes

- **No outlier clipping applied:** Outlier clipping, if needed, will be applied during model training on train-only data.

- **No imputation applied:** Missing values (NaN) are preserved. Imputation decisions will be made during model training.

- **Corrections performed:** Invalid measurements (negative r_* values, out-of-range n_* values) were converted to NaN rather than dropping rows, to preserve potential failure signals.

