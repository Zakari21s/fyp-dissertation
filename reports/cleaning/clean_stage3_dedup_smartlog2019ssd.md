# Cleaning Stage 3: Deduplication Safety Check - smartlog2019ssd

**Generated:** 2026-02-06T15:49:19.842416

## Overview

- **Total partitions:** 12
- **Partitions processed:** 12
- **Partitions failed:** 0
- **Total files:** 12
- **Files processed:** 12
- **Files failed:** 0
- **Processing time:** 1437.69 seconds

## Global Row Counts

- **Rows in:** 128,776,521
- **Rows out:** 128,776,521
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
| `year=2019/month=01` | 12,148,058 | 12,148,058 | 0 | 0 | 12,148,058 | 0 | 0 | 0 |
| `year=2019/month=02` | 10,872,424 | 10,872,424 | 0 | 0 | 10,872,424 | 0 | 0 | 0 |
| `year=2019/month=03` | 12,068,266 | 12,068,266 | 0 | 0 | 12,068,266 | 0 | 0 | 0 |
| `year=2019/month=04` | 11,307,999 | 11,307,999 | 0 | 0 | 11,307,999 | 0 | 0 | 0 |
| `year=2019/month=05` | 11,731,037 | 11,731,037 | 0 | 0 | 11,731,037 | 0 | 0 | 0 |
| `year=2019/month=06` | 10,611,803 | 10,611,803 | 0 | 0 | 10,611,803 | 0 | 0 | 0 |
| `year=2019/month=07` | 11,083,718 | 11,083,718 | 0 | 0 | 11,083,718 | 0 | 0 | 0 |
| `year=2019/month=08` | 11,059,118 | 11,059,118 | 0 | 0 | 11,059,118 | 0 | 0 | 0 |
| `year=2019/month=09` | 10,550,469 | 10,550,469 | 0 | 0 | 10,550,469 | 0 | 0 | 0 |
| `year=2019/month=10` | 10,228,261 | 10,228,261 | 0 | 0 | 10,228,261 | 0 | 0 | 0 |
| `year=2019/month=11` | 8,975,113 | 8,975,113 | 0 | 0 | 8,975,113 | 0 | 0 | 0 |
| `year=2019/month=12` | 8,140,255 | 8,140,255 | 0 | 0 | 8,140,255 | 0 | 0 | 0 |

