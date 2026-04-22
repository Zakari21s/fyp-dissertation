# Failure Labels Deduplication Summary

**Generated:** 2026-02-05T17:03:59.619394

## Overview

- **Rows before deduplication:** 16,305
- **Rows after deduplication:** 15,823
- **Rows removed:** 482
- **Total unique disks:** 15,823

## Duplicate Analysis

- **Disks with multiple records:** 467
- **Disks with conflicting failure_time:** 467

**Note:** Duplicates were resolved by keeping the record with the earliest `failure_time` per `disk_id`.

## Datetime Parsing

- **Parse success rate:** 100.00%
- **Successfully parsed:** 16,305 / 16,305

## Failure Date Range

- **Min failure date:** 2018-01-02 19:15:32
- **Max failure date:** 2019-12-31 21:57:36

## Output

Cleaned data has been saved to `data_interim/labels/failure_labels_dedup.parquet`

The output includes:
- All original columns
- `failure_time_parsed`: Parsed datetime version of `failure_time`
- One record per `disk_id` (earliest `failure_time` kept)
