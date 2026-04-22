# Stage 5: Outlier Analysis - smartlog2018ssd

**Generated:** 2026-02-06T17:35:38.961293

**IMPORTANT:** This is an analysis-only stage. No data was modified.

## Overview

- **Dataset:** smartlog2018ssd
- **Total rows analyzed:** 133,559,111
- **Total partitions:** 12
- **Partitions processed:** 12
- **Partitions failed:** 0
- **SMART features analyzed:** 55
- **Processing time:** 1792.20 seconds

## Global Summary

- **Features with negative values:** 0
- **Features with extreme skew:** 7
- **Features with heavy tails:** 15

## Top 30 Most Extreme Features

| Feature | Min | Max | P1 | P99 | P99.9 | % Negative | % Zero |
|---------|-----|-----|----|----|-------|------------|--------|
| `r_242` | 0.00 | 18647553342824.00 | 22612.00 | 1205204145739.00 | 3045838092161.00 | 0.00% | 0.01% |
| `r_241` | 0.00 | 12615022612244.00 | 37451.00 | 1372053864096.00 | 2861915637163.00 | 0.00% | 0.04% |
| `r_180` | 95.00 | 281470681743359.00 | 6469.00 | 4294967295.00 | 4294967295.00 | 0.00% | 0.00% |
| `r_1` | 0.00 | 42949672959.00 | 0.00 | 4294967295.00 | 4294967295.00 | 0.00% | 77.32% |
| `r_195` | 0.00 | 4294884181.00 | 0.00 | 1078168553.00 | 2261021694.00 | 0.00% | 90.28% |
| `r_188` | 0.00 | 714988748.00 | 0.00 | 242.00 | 1581.00 | 0.00% | 12.50% |
| `r_184` | 0.00 | 98260840.00 | 0.00 | 0.00 | 0.00 | 0.00% | 99.96% |
| `r_187` | 0.00 | 9309631.00 | 0.00 | 0.00 | 15.00 | 0.00% | 99.81% |
| `r_199` | 0.00 | 5840350.00 | 0.00 | 133.00 | 4580.00 | 0.00% | 90.70% |
| `r_198` | 0.00 | 747508.00 | 0.00 | 1.00 | 2.00 | 0.00% | 98.91% |
| `r_183` | 0.00 | 155816.00 | 0.00 | 0.00 | 836.00 | 0.00% | 99.37% |
| `r_192` | 1.00 | 51206.00 | 3.00 | 85.00 | 1839.00 | 0.00% | 0.00% |
| `r_174` | 0.00 | 51206.00 | 4.00 | 72.00 | 596.00 | 0.00% | 0.12% |
| `r_12` | 1.00 | 51209.00 | 8.00 | 72.00 | 671.00 | 0.00% | 0.00% |
| `r_9` | 0.00 | 47283.00 | 482.00 | 41148.00 | 45524.00 | 0.00% | 0.00% |
| `r_171` | 0.00 | 46424.00 | 0.00 | 3.00 | 64.00 | 0.00% | 94.83% |
| `r_206` | 0.00 | 16980.00 | 0.00 | 1.00 | 11.00 | 0.00% | 94.64% |
| `r_170` | 0.00 | 11311.00 | 0.00 | 17.00 | 330.00 | 0.00% | 91.80% |
| `r_196` | 0.00 | 11311.00 | 0.00 | 38.00 | 2176.00 | 0.00% | 86.82% |
| `r_5` | 0.00 | 11311.00 | 0.00 | 21.00 | 2043.00 | 0.00% | 91.12% |
| `r_197` | 0.00 | 7342.00 | 0.00 | 0.00 | 2.00 | 0.00% | 99.81% |
| `r_177` | 0.00 | 6040.00 | 0.00 | 1069.00 | 3052.00 | 0.00% | 2.23% |
| `r_173` | 0.00 | 4510.00 | 0.00 | 493.00 | 896.00 | 0.00% | 17.28% |
| `r_181` | 0.00 | 3017.00 | 0.00 | 0.00 | 2082.00 | 0.00% | 99.45% |
| `r_182` | 0.00 | 2542.00 | 0.00 | 0.00 | 0.00 | 0.00% | 99.93% |
| `r_172` | 0.00 | 2314.00 | 0.00 | 0.00 | 1.00 | 0.00% | 99.76% |
| `r_244` | 0.00 | 873.00 | 0.00 | 0.00 | 0.00 | 0.00% | 100.00% |
| `r_175` | 4295295630.00 | 1207331258987.00 | 365090243207.00 | 1117189243511.00 | 1164433162845.00 | 0.00% | 0.00% |
| `n_195` | 1.00 | 200.00 | 100.00 | 200.00 | 200.00 | 0.00% | 0.00% |
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

