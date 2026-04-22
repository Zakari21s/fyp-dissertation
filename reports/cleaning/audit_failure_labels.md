# Failure Labels Audit Report

**Generated:** 2026-02-05T16:59:34.130345

**File:** `data_raw/ssd_failure_label.csv`

## Summary

- **Total rows:** 16,305
- **Columns:** 3
- **Processing time:** 0.04s

## Schema

| Column | Data Type |
|--------|-----------|
| `model` | `str` |
| `failure_time` | `str` |
| `disk_id` | `int64` |

## Missingness

| Column | Missing % |
|--------|----------|
| `model` | 0.00% |
| `failure_time` | 0.00% |
| `disk_id` | 0.00% |

## Duplicate Detection

**Disk ID column:** `disk_id`

- **Duplicate rows:** 482
- **Duplicate rate:** 2.96%
- **Unique disks with failures:** 15,823

## Datetime Parsing

**Datetime column:** `failure_time`

- **Format used:** `%Y-%m-%d %H:%M:%S`
- **Inferred format:** `%Y-%m-%d %H:%M:%S`
- **Parse success rate:** 100.00%
- **Formats tried:** 5
  - %Y%m%d
  - %Y-%m-%d
  - %Y/%m/%d
  - %Y%m%d%H%M%S
  - %Y-%m-%d %H:%M:%S

### Parsed Failure Time Range

- **Min failure date:** 2018-01-02 19:15:32
- **Max failure date:** 2019-12-31 22:58:47

## Data Quality Summary

⚠️ **Issues detected:**

- Duplicate rate is 2.96%

