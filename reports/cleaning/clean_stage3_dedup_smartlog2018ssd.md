# Cleaning Stage 3: Deduplication Safety Check - smartlog2018ssd

**Generated:** 2026-02-06T15:24:29.883562

## Overview

- **Total partitions:** 12
- **Partitions processed:** 12
- **Partitions failed:** 0
- **Total files:** 12
- **Files processed:** 12
- **Files failed:** 0
- **Processing time:** 1500.88 seconds

## Global Row Counts

- **Rows in:** 133,559,111
- **Rows out:** 133,559,111
- **Rows removed:** 0
- **% of dataset affected:** 0.0000%

## Global Data Quality Issues

- **Rows dropped (missing disk_id):** 0
- **Rows dropped (missing smart_day):** 0
- **Rows with missing model (set to 'UNKNOWN'):** 0

## Global Duplicate Analysis

- **Total duplicate rows found:** 0

**Note:** Duplicates were identified by (disk_id, model, smart_day) key at the partition level. Only the first occurrence encountered in streaming order was kept.

## Per-Month Breakdown

| Partition | Rows In | Rows Out | Duplicate Rows | Rows Removed | Unique Keys | Missing disk_id | Missing smart_day | Missing model |
|-----------|---------|----------|----------------|--------------|------------|-----------------|-------------------|---------------|
| `year=2018/month=01` | 10,529,353 | 10,529,353 | 0 | 0 | 10,529,353 | 0 | 0 | 0 |
| `year=2018/month=02` | 9,668,382 | 9,668,382 | 0 | 0 | 9,668,382 | 0 | 0 | 0 |
| `year=2018/month=03` | 10,947,243 | 10,947,243 | 0 | 0 | 10,947,243 | 0 | 0 | 0 |
| `year=2018/month=04` | 10,447,825 | 10,447,825 | 0 | 0 | 10,447,825 | 0 | 0 | 0 |
| `year=2018/month=05` | 11,314,286 | 11,314,286 | 0 | 0 | 11,314,286 | 0 | 0 | 0 |
| `year=2018/month=06` | 11,203,125 | 11,203,125 | 0 | 0 | 11,203,125 | 0 | 0 | 0 |
| `year=2018/month=07` | 11,647,453 | 11,647,453 | 0 | 0 | 11,647,453 | 0 | 0 | 0 |
| `year=2018/month=08` | 12,016,685 | 12,016,685 | 0 | 0 | 12,016,685 | 0 | 0 | 0 |
| `year=2018/month=09` | 11,633,245 | 11,633,245 | 0 | 0 | 11,633,245 | 0 | 0 | 0 |
| `year=2018/month=10` | 11,772,193 | 11,772,193 | 0 | 0 | 11,772,193 | 0 | 0 | 0 |
| `year=2018/month=11` | 10,304,367 | 10,304,367 | 0 | 0 | 10,304,367 | 0 | 0 | 0 |
| `year=2018/month=12` | 12,074,954 | 12,074,954 | 0 | 0 | 12,074,954 | 0 | 0 | 0 |

