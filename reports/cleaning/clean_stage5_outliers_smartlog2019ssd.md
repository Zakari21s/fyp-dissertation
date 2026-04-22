# Stage 5: Outlier Analysis - smartlog2019ssd

**Generated:** 2026-02-06T18:08:56.408021

**IMPORTANT:** This is an analysis-only stage. No data was modified.

## Overview

- **Dataset:** smartlog2019ssd
- **Total rows analyzed:** 128,776,521
- **Total partitions:** 12
- **Partitions processed:** 12
- **Partitions failed:** 0
- **SMART features analyzed:** 55
- **Processing time:** 1771.25 seconds

## Global Summary

- **Features with negative values:** 0
- **Features with extreme skew:** 6
- **Features with heavy tails:** 15

## Top 30 Most Extreme Features

| Feature | Min | Max | P1 | P99 | P99.9 | % Negative | % Zero |
|---------|-----|-----|----|----|-------|------------|--------|
| `r_242` | 0.00 | 21668602808240.00 | 29423.00 | 1504901924431.00 | 3210749783760.00 | 0.00% | 0.00% |
| `r_241` | 0.00 | 14359022799097.00 | 37931.00 | 2070624817666.00 | 5107903169822.00 | 0.00% | 0.04% |
| `r_180` | 534.00 | 280817846714367.00 | 6474.00 | 4294967295.00 | 4294967295.00 | 0.00% | 0.00% |
| `r_1` | 0.00 | 55834574847.00 | 0.00 | 4294967295.00 | 4294967295.00 | 0.00% | 86.67% |
| `r_195` | 0.00 | 4294909784.00 | 0.00 | 910706951.00 | 2534536635.00 | 0.00% | 93.76% |
| `r_188` | 0.00 | 98260898.00 | 0.00 | 258.00 | 1825.00 | 0.00% | 7.84% |
| `r_184` | 0.00 | 98260840.00 | 0.00 | 0.00 | 0.00 | 0.00% | 99.96% |
| `r_9` | 0.00 | 6218787.00 | 2183.00 | 48549.00 | 53914.00 | 0.00% | 0.00% |
| `r_199` | 0.00 | 5840350.00 | 0.00 | 124.00 | 2420.00 | 0.00% | 91.87% |
| `r_187` | 0.00 | 5381988.00 | 0.00 | 0.00 | 0.00 | 0.00% | 99.91% |
| `r_183` | 0.00 | 155816.00 | 0.00 | 0.00 | 42.00 | 0.00% | 99.45% |
| `r_12` | 0.00 | 51209.00 | 4.00 | 76.00 | 903.00 | 0.00% | 0.00% |
| `r_192` | 1.00 | 51206.00 | 3.00 | 87.00 | 603.00 | 0.00% | 0.00% |
| `r_174` | 0.00 | 51206.00 | 2.00 | 71.00 | 368.00 | 0.00% | 0.35% |
| `r_171` | 0.00 | 46424.00 | 0.00 | 4.00 | 41.00 | 0.00% | 91.22% |
| `r_206` | 0.00 | 16980.00 | 0.00 | 3.00 | 16.00 | 0.00% | 89.29% |
| `r_170` | 0.00 | 11376.00 | 0.00 | 20.00 | 2169.00 | 0.00% | 86.45% |
| `r_196` | 0.00 | 11376.00 | 0.00 | 31.00 | 2177.00 | 0.00% | 81.41% |
| `r_5` | 0.00 | 11376.00 | 0.00 | 25.00 | 2168.00 | 0.00% | 87.58% |
| `r_177` | 0.00 | 7115.00 | 1.00 | 1860.00 | 4228.00 | 0.00% | 0.98% |
| `r_173` | 0.00 | 7790.00 | 0.00 | 901.00 | 1731.00 | 0.00% | 10.17% |
| `r_182` | 0.00 | 3767.00 | 0.00 | 0.00 | 1.00 | 0.00% | 99.92% |
| `r_181` | 0.00 | 3407.00 | 0.00 | 0.00 | 2047.00 | 0.00% | 99.45% |
| `r_172` | 0.00 | 2314.00 | 0.00 | 0.00 | 6.00 | 0.00% | 99.72% |
| `r_197` | 0.00 | 2143.00 | 0.00 | 0.00 | 2.00 | 0.00% | 99.84% |
| `r_175` | 4341367466.00 | 1417417982585.00 | 558963688058.00 | 1301841511038.00 | 1378966831732.00 | 0.00% | 0.00% |
| `r_198` | 0.00 | 286.00 | 0.00 | 0.00 | 2.00 | 0.00% | 99.58% |
| `n_195` | 1.00 | 200.00 | 100.00 | 200.00 | 200.00 | 0.00% | 0.00% |
| `r_194` | 0.00 | 139.00 | 13.00 | 38.00 | 48.00 | 0.00% | 0.00% |
| `r_190` | 0.00 | 128.00 | 10.00 | 35.00 | 39.00 | 0.00% | 0.00% |

## Interpretation

### Are Outliers Likely Noise or Signal?

SMART attributes are hardware sensor readings that may exhibit extreme values when:
- Hardware is approaching failure (predictive signal)
- Sensors malfunction (noise)
- Data collection errors occur (noise)

**Analysis:** 15 features show heavy-tailed distributions, suggesting that extreme values may be meaningful failure indicators rather than noise.

### Which SMART Attributes Look Physically Implausible?

No features contain negative values, which is expected for most SMART attributes.

### Are Distributions Model-Dependent?

This analysis aggregates across all disk models. Model-specific analysis would require grouping by the `model` column to identify if certain models exhibit different outlier patterns.

## Decision

### No Outlier Clipping Applied

**At this stage, no outlier clipping has been applied.**

### Future Clipping Considerations

If outlier clipping is deemed necessary during model training, it MUST:
1. Be computed on training data only (to prevent data leakage)
2. Use train-set percentiles (e.g., 1st and 99th) as clipping boundaries
3. Apply the same boundaries to test data
4. Be documented as a model-specific preprocessing step

### Recommendation

Given the prevalence of heavy-tailed distributions, consider:
- Using robust models (e.g., tree-based) that handle outliers naturally
- Applying log transformations for highly skewed features
- Training-set-only clipping if using linear models

