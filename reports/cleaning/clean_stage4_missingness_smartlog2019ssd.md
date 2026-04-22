# Cleaning Stage 4: Missingness Handling + Feature Coverage Filtering - smartlog2019ssd

**Generated:** 2026-02-06T16:14:28.407873

## Overview

- **Total partitions:** 12
- **Partitions processed:** 12
- **Partitions failed:** 0
- **Processing time:** 177.54 seconds

## Configuration

- **Batch size:** 250,000
- **Min feature coverage:** 0.01
- **Drop constant features:** True
- **Enable row filter:** False
- **Enable missing indicators:** False

## Global Row Counts

- **Rows in:** 128,776,521
- **Rows out:** 128,776,521
- **% kept:** 100.0000%

## Feature Filtering Summary

- **SMART features original:** 102
- **SMART features dropped (low coverage):** 36
- **SMART features dropped (constant):** 11
- **SMART features kept:** 55

### Dropped Features

Total dropped: 47

First 50 dropped features:

```
n_10
n_11
n_13
n_174
n_188
n_189
n_191
n_192
n_193
n_197
n_198
n_2
n_200
n_204
n_205
n_206
n_207
n_211
n_240
n_244
n_245
n_3
n_4
n_6
n_7
n_8
r_10
r_11
r_13
r_189
r_191
r_193
r_2
r_200
r_204
r_205
r_207
r_211
r_232
r_233
r_240
r_245
r_3
r_4
r_6
r_7
r_8
```

## Top 20 Most Missing SMART Features

| Feature | Coverage | Missing Rate |
|---------|----------|--------------|
| `n_10` | 0.0000 | 1.0000 |
| `n_11` | 0.0000 | 1.0000 |
| `n_13` | 0.0000 | 1.0000 |
| `n_189` | 0.0000 | 1.0000 |
| `n_191` | 0.0000 | 1.0000 |
| `n_193` | 0.0000 | 1.0000 |
| `n_2` | 0.0000 | 1.0000 |
| `n_200` | 0.0000 | 1.0000 |
| `n_204` | 0.0000 | 1.0000 |
| `n_205` | 0.0000 | 1.0000 |
| `n_207` | 0.0000 | 1.0000 |
| `n_240` | 0.0000 | 1.0000 |
| `n_3` | 0.0000 | 1.0000 |
| `n_4` | 0.0000 | 1.0000 |
| `n_6` | 0.0000 | 1.0000 |
| `n_7` | 0.0000 | 1.0000 |
| `n_8` | 0.0000 | 1.0000 |
| `r_10` | 0.0000 | 1.0000 |
| `r_11` | 0.0000 | 1.0000 |
| `r_13` | 0.0000 | 1.0000 |

## Per-Month Breakdown

| Partition | Rows In | Rows Out | Rows Dropped (min_features) |
|-----------|---------|----------|---------------------------|
| `year=2019/month=01` | 12,148,058 | 12,148,058 | 0 |
| `year=2019/month=02` | 10,872,424 | 10,872,424 | 0 |
| `year=2019/month=03` | 12,068,266 | 12,068,266 | 0 |
| `year=2019/month=04` | 11,307,999 | 11,307,999 | 0 |
| `year=2019/month=05` | 11,731,037 | 11,731,037 | 0 |
| `year=2019/month=06` | 10,611,803 | 10,611,803 | 0 |
| `year=2019/month=07` | 11,083,718 | 11,083,718 | 0 |
| `year=2019/month=08` | 11,059,118 | 11,059,118 | 0 |
| `year=2019/month=09` | 10,550,469 | 10,550,469 | 0 |
| `year=2019/month=10` | 10,228,261 | 10,228,261 | 0 |
| `year=2019/month=11` | 8,975,113 | 8,975,113 | 0 |
| `year=2019/month=12` | 8,140,255 | 8,140,255 | 0 |

