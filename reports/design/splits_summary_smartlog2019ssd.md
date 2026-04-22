# Time-Based Dataset Splitting Summary - smartlog2019ssd

**Generated:** 2026-02-12T00:36:15.856752

## Overview

- **Train Dataset:** smartlog2018ssd
- **Eval Dataset:** smartlog2019ssd
- **Total Input Rows:** 262,335,632
- **Processing Time:** 1263.25s

## Split Definitions

- **TRAIN:** All rows with `smart_day` in year 2018
- **VAL:** 2019-01-01 to 2019-06-30 (inclusive)
- **TEST:** 2019-07-01 to 2019-12-31 (inclusive)

## Entity Disjointness

To prevent data leakage:
- Removed from TRAIN any entity (disk_id, model) that appears in VAL or TEST
- Removed from VAL any entity that appears in TEST
- TEST remains unchanged

## Split Statistics

### TRAIN

- **Rows (before overlap removal):** 133,559,111
- **Rows removed (overlap):** 126,605,553
- **Rows (final):** 6,953,558
- **Entities (before):** 455,294
- **Entities (after):** 35,773
- **Date Range:** 2018-01-01 00:00:00 to 2018-12-31 00:00:00

**Label Distribution:**

| Label | Positive | Negative | Total |
|-------|----------|----------|-------|
| `y_7` | 141,943 | 6,811,615 | 6,953,558 |
| `y_14` | 260,956 | 6,692,602 | 6,953,558 |
| `y_30` | 541,384 | 6,412,174 | 6,953,558 |

### VAL

- **Rows (before overlap removal):** 68,739,587
- **Rows removed (overlap):** 64,833,340
- **Rows (final):** 3,906,247
- **Entities (before):** 425,951
- **Entities (after):** 42,085
- **Date Range:** 2019-01-01 00:00:00 to 2019-06-30 00:00:00

**Label Distribution:**

| Label | Positive | Negative | Total |
|-------|----------|----------|-------|
| `y_7` | 85,310 | 3,820,937 | 3,906,247 |
| `y_14` | 160,082 | 3,746,165 | 3,906,247 |
| `y_30` | 317,158 | 3,589,089 | 3,906,247 |

### TEST

- **Rows (before overlap removal):** 60,036,934
- **Rows removed (overlap):** 0
- **Rows (final):** 60,036,934
- **Entities (before):** 395,032
- **Entities (after):** 395,032
- **Date Range:** 2019-07-01 00:00:00 to 2019-12-31 00:00:00

**Label Distribution:**

| Label | Positive | Negative | Total |
|-------|----------|----------|-------|
| `y_7` | 103,694 | 59,933,240 | 60,036,934 |
| `y_14` | 195,814 | 59,841,120 | 60,036,934 |
| `y_30` | 392,378 | 59,644,556 | 60,036,934 |

