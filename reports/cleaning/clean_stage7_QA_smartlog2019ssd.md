# Stage 7: Final QA Summary + Acceptance Criteria Validation - smartlog2019ssd

**Generated:** 2026-02-06T23:46:39.613446

**IMPORTANT:** This is a read-only validation stage. No data was modified.

## Overview

- **Dataset:** smartlog2019ssd
- **Input path:** `data_interim/clean_stage6_smartlog2019ssd/`
- **Pipeline stages completed:** 1 (Schema), 2 (Types), 3 (Deduplication), 4 (Missingness), 5 (Outlier Analysis), 6 (Invalid Records), 7 (QA)

## Acceptance Criteria Checklist

| Criterion | Status | Notes |
|-----------|--------|-------|
| Schema Consistency | PASS | All 12 partitions have identical schema (62 columns) |
| Type Consistency | PASS | All types are correct (model dtype 'str' accepted as string-equivalent representation) |
| Uniqueness | PASS | No duplicate keys found for (disk_id, model, smart_day) |
| Missingness | PASS | All identifiers/labels have 0% missing; all other columns < 95% |
| Label Distribution | PASS | All labels are binary (0/1) with valid distributions |
| Label Nesting Consistency | PASS | All labels satisfy nesting constraint (y_7 ≤ y_14 ≤ y_30). |
| Date Coverage | PASS | Date range: 2019-01-01 00:00:00 to 2019-12-31 00:00:00 (within expected range) |
| Row Count Reconciliation | PASS | Total rows: 128,776,521 across 12 partitions |

## Dataset Summary

- **Total rows:** 128,776,521
- **Unique disk_id count:** 186,479
- **Unique (disk_id, model) count:** 407,424

**Note:** Disk identity is defined by (disk_id, model) due to model reuse. The unique (disk_id, model) count represents the true number of distinct disk entities.

- **Date range:** 2019-01-01 00:00:00 to 2019-12-31 00:00:00
- **Number of n_* features:** 25
- **Number of r_* features:** 30
- **Total partitions:** 12

## Schema & Types

- **Column count:** 62
- **Schema consistent:** Yes

### Column Types

| Column | Type |
|--------|------|
| `disk_id` | `int64` |
| `ds` | `int32` |
| `model` | `str` |
| `n_1` | `float64` |
| `n_12` | `float64` |
| `n_170` | `float64` |
| `n_171` | `float64` |
| `n_172` | `float64` |
| `n_173` | `float64` |
| `n_175` | `float64` |
| `n_177` | `float64` |
| `n_180` | `float64` |
| `n_181` | `float64` |
| `n_182` | `float64` |
| `n_183` | `float64` |
| `n_184` | `float64` |
| `n_187` | `float64` |
| `n_190` | `float64` |
| `n_194` | `float64` |
| `n_195` | `float64` |
| `n_196` | `float64` |
| `n_199` | `float64` |
| `n_232` | `float64` |
| `n_233` | `float64` |
| `n_241` | `float64` |
| `n_242` | `float64` |
| `n_5` | `float64` |
| `n_9` | `float64` |
| `r_1` | `float64` |
| `r_12` | `float64` |
| `r_170` | `float64` |
| `r_171` | `float64` |
| `r_172` | `float64` |
| `r_173` | `float64` |
| `r_174` | `float64` |
| `r_175` | `float64` |
| `r_177` | `float64` |
| `r_180` | `float64` |
| `r_181` | `float64` |
| `r_182` | `float64` |
| `r_183` | `float64` |
| `r_184` | `float64` |
| `r_187` | `float64` |
| `r_188` | `float64` |
| `r_190` | `float64` |
| `r_192` | `float64` |
| `r_194` | `float64` |
| `r_195` | `float64` |
| `r_196` | `float64` |
| `r_197` | `float64` |
| `r_198` | `float64` |
| `r_199` | `float64` |
| `r_206` | `float64` |
| `r_241` | `float64` |
| `r_242` | `float64` |
| `r_244` | `float64` |
| `r_5` | `float64` |
| `r_9` | `float64` |
| `smart_day` | `datetime64[us]` |
| `y_14` | `int8` |
| `y_30` | `int8` |
| `y_7` | `int8` |

## Missingness Summary

### Top 20 Columns by Missing %

| Column | Missing % |
|--------|----------|
| `n_182` | 79.57% |
| `r_182` | 79.57% |
| `n_181` | 79.57% |
| `r_181` | 79.57% |
| `r_244` | 79.57% |
| `n_177` | 79.57% |
| `r_177` | 79.57% |
| `n_233` | 77.54% |
| `n_232` | 77.54% |
| `r_192` | 77.54% |
| `n_175` | 71.73% |
| `r_175` | 71.73% |
| `n_241` | 57.12% |
| `r_241` | 57.12% |
| `n_242` | 57.12% |
| `r_242` | 57.12% |
| `n_190` | 52.90% |
| `r_190` | 52.90% |
| `n_1` | 48.70% |
| `r_206` | 48.70% |

### Identifiers & Labels Missingness

- `disk_id`: 0.00% missing ✓
- `model`: 0.00% missing ✓
- `smart_day`: 0.00% missing ✓
- `ds`: 0.00% missing ✓
- `y_7`: 0.00% missing ✓
- `y_14`: 0.00% missing ✓
- `y_30`: 0.00% missing ✓

**✓ All identifiers and labels have 0% missing values.**

## Label Distribution

| Label | Positive | Negative | Total | Positive Rate |
|-------|----------|----------|-------|---------------|
| `y_7` | 191,529 | 128,584,992 | 128,776,521 | 0.1487% |
| `y_14` | 365,234 | 128,411,287 | 128,776,521 | 0.2836% |
| `y_30` | 748,448 | 128,028,073 | 128,776,521 | 0.5812% |

**Note:** Severe class imbalance is expected for failure prediction tasks.

## Label Nesting Consistency

This check verifies temporal consistency of horizon-based labels: y_7 ≤ y_14 ≤ y_30.

**✓ PASS:** All labels satisfy nesting constraint.

## Final Statement

**✓ DATASET STATUS: CLEANED, VALIDATED, AND READY FOR MODELING**

All acceptance criteria have been met. The dataset has been successfully cleaned through Stages 1-6 and validated in Stage 7. It is now ready for model training.

