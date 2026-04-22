# Time-Based Dataset Splitting Summary - exp_time_generalisation

**Generated:** 2026-02-12T01:39:24.650831

## Overview

- **Experiment:** exp_time_generalisation
- **Train Dataset:** smartlog2018ssd
- **Eval Dataset:** smartlog2019ssd
- **Entity Disjoint Policy:** `none`
- **Total Input Rows:** 262,335,632
- **Processing Time:** 2161.70s

## Split Definitions

- **TRAIN:** All rows with `smart_day` in year 2018
- **VAL:** 2019-01-01 to 2019-06-30 (inclusive)
- **TEST:** 2019-07-01 to 2019-12-31 (inclusive)

## Entity Disjointness Policy

**Policy: NONE** - No entity exclusions. Splits are purely time-based.


## Split Statistics

### TRAIN

- **Rows (before filtering):** 133,559,111
- **Rows (after date filter):** 133,559,111
- **Rows (after entity filter):** 133,559,111
- **Rows dropped (date filter):** 0
- **Rows dropped (entity overlap):** 0
- **Rows (final):** 133,559,111
- **Entities (before):** 455,294
- **Entities (after):** 455,294
- **Date Range:** 2018-01-01 00:00:00 to 2018-12-31 00:00:00

**Label Distribution:**

| Label | Positive | Negative | Total |
|-------|----------|----------|-------|
| `y_7` | 142,787 | 133,416,324 | 133,559,111 |
| `y_14` | 264,744 | 133,294,367 | 133,559,111 |
| `y_30` | 567,799 | 132,991,312 | 133,559,111 |

### VAL

- **Rows (before filtering):** 68,739,587
- **Rows (after date filter):** 68,739,587
- **Rows (after entity filter):** 68,739,587
- **Rows dropped (date filter):** 0
- **Rows dropped (entity overlap):** 0
- **Rows (final):** 68,739,587
- **Entities (before):** 425,951
- **Entities (after):** 425,951
- **Date Range:** 2019-01-01 00:00:00 to 2019-06-30 00:00:00

**Label Distribution:**

| Label | Positive | Negative | Total |
|-------|----------|----------|-------|
| `y_7` | 87,835 | 68,651,752 | 68,739,587 |
| `y_14` | 169,420 | 68,570,167 | 68,739,587 |
| `y_30` | 356,070 | 68,383,517 | 68,739,587 |

### TEST

- **Rows (before filtering):** 60,036,934
- **Rows (after date filter):** 60,036,934
- **Rows (after entity filter):** 60,036,934
- **Rows dropped (date filter):** 0
- **Rows dropped (entity overlap):** 0
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

