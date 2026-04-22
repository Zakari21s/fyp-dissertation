# Cleaning Stage 6: Invalid Record Filtering + Label/Key Sanity - smartlog2019ssd

**Generated:** 2026-02-06T18:33:31.587401

## Overview

- **Dataset:** smartlog2019ssd
- **Total partitions:** 12
- **Partitions processed:** 12
- **Partitions failed:** 0
- **Processing time:** 172.72 seconds

## Global Row Counts

- **Rows in:** 128,776,521
- **Rows out:** 128,776,521
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
- **Total n_* out-of-range values corrected:** 41,268,165

### Top 15 n_* Columns with Out-of-Range Values Fixed to NaN

| Column | Count |
|--------|-------|
| `n_195` | 26,305,202 |
| `n_1` | 7,482,342 |
| `n_180` | 7,480,621 |

## Key Sanity Check

**Total duplicate keys found:** 0

**Note:** Duplicates are reported for monitoring only. Deduplication was already performed in Stage 3.

| Partition | Duplicate Keys |
|-----------|----------------|

## Per-Month Breakdown

| Partition | Rows In | Rows Out | Missing disk_id | Missing model | Missing smart_day | Missing ds | Invalid Labels |
|-----------|---------|----------|-----------------|---------------|-------------------|------------|----------------|
| `year=2019/month=01` | 12,148,058 | 12,148,058 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=02` | 10,872,424 | 10,872,424 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=03` | 12,068,266 | 12,068,266 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=04` | 11,307,999 | 11,307,999 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=05` | 11,731,037 | 11,731,037 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=06` | 10,611,803 | 10,611,803 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=07` | 11,083,718 | 11,083,718 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=08` | 11,059,118 | 11,059,118 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=09` | 10,550,469 | 10,550,469 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=10` | 10,228,261 | 10,228,261 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=11` | 8,975,113 | 8,975,113 | 0 | 0 | 0 | 0 | 0 |
| `year=2019/month=12` | 8,140,255 | 8,140,255 | 0 | 0 | 0 | 0 | 0 |

## Important Notes

- **No outlier clipping applied:** Outlier clipping, if needed, will be applied during model training on train-only data.

- **No imputation applied:** Missing values (NaN) are preserved. Imputation decisions will be made during model training.

- **Corrections performed:** Invalid measurements (negative r_* values, out-of-range n_* values) were converted to NaN rather than dropping rows, to preserve potential failure signals.

