# Reports Index

Central navigation for everything under [`reports/`](../).  
Result files were reorganized into subfolders; this index reflects the current layout.

## Folder layout (quick map)

| Folder | Contents |
|--------|----------|
| [`index/`](./) | This file + [`baseline_inventory.md`](./baseline_inventory.md) |
| [`cleaning/`](../cleaning/) | Audit/clean stage JSON/MD |
| [`design/`](../design/) | Labeling, splitting, split summaries |
| [`feature_ranking/`](../feature_ranking/) | `feature_ranking_y30.*` |
| [`feature_engineering/`](../feature_engineering/) | `feature_engineering_summary_*` |
| [`metadata/`](../metadata/) | Run metadata JSON (disk-first-seen, model mapping, labeling summaries) |
| [`docs/`](../docs/) | Top-level documentation/spec notes (schema drift, labeling summaries, cleaning spec) |
| [`lr-no-eng/`](../lr-no-eng/) | LR + Naive baselines on raw splits (non-engineered) |
| [`rf-no-eng/`](../rf-no-eng/) | Random Forest on raw splits (non-engineered) |
| [`lr-eng/`](../lr-eng/) | LR / tabular outputs on engineered data |
| [`rf-eng/`](../rf-eng/) | RF outputs on engineered data |
| [`xgb-eng/`](../xgb-eng/) | XGBoost outputs on engineered data |
| [`xgb-no-eng/`](../xgb-no-eng/) | XGBoost on raw splits (reserved for future runs) |
| `reports/` (root) | Model/result subfolders only |

## 1) Baseline snapshot

- [`baseline_inventory.md`](./baseline_inventory.md) — frozen list of runs with **paths relative to `reports/`**

## 2) Design

- [`../design/labeling_design.md`](../design/labeling_design.md)
- [`../design/splitting_design.md`](../design/splitting_design.md)
- [`../design/splits_summary_exp_time_generalisation.md`](../design/splits_summary_exp_time_generalisation.md) (+ `__train_eval` variants if present)

## 3) Feature ranking

- [`../feature_ranking/feature_ranking_y30.md`](../feature_ranking/feature_ranking_y30.md)
- [`../feature_ranking/feature_ranking_y30.csv`](../feature_ranking/feature_ranking_y30.csv)
- [`../feature_ranking/feature_ranking_y30.json`](../feature_ranking/feature_ranking_y30.json)

## 4) Feature engineering summaries

- [`../feature_engineering/feature_engineering_summary_exp_time_generalisation_train.md`](../feature_engineering/feature_engineering_summary_exp_time_generalisation_train.md)
- [`../feature_engineering/feature_engineering_summary_exp_time_generalisation_val.md`](../feature_engineering/feature_engineering_summary_exp_time_generalisation_val.md)
- [`../feature_engineering/feature_engineering_summary_exp_time_generalisation_test.md`](../feature_engineering/feature_engineering_summary_exp_time_generalisation_test.md)
- (matching `.json` files in the same folder)

## 5) Non-engineered model results

### LR + Naive (`lr-no-eng/`)

- Pattern: `../lr-no-eng/baseline_y*_results_*.json` / `.md`
- Balanced LR: `../lr-no-eng/baseline_y*_lr_balanced_results_*.json` / `.md`

### Random Forest (`rf-no-eng/`)

- Pattern: `../rf-no-eng/baseline_rf_y_*_results_*.json` / `.md`
- Balanced RF: `../rf-no-eng/baseline_rf_y_*_balanced_results_*.json` / `.md`

Horizons appear in filenames (`y_7`, `y_14`, `y_30`). Experiments: `exp_time_generalisation`, `exp_time_generalisation__train_eval`.

## 6) Engineered model results (`lr-eng/`)

- [`../lr-eng/lr_unbalanced_engineered_y30_exp_time_generalisation.json`](../lr-eng/lr_unbalanced_engineered_y30_exp_time_generalisation.json) / [`.md`](../lr-eng/lr_unbalanced_engineered_y30_exp_time_generalisation.md)
- Tabular / pipeline summaries (engineered-related): `../lr-eng/ml_tabular_y30_summary_*.json` / `.md`

## 7) Engineered RF (`rf-eng/`)

- Pattern: `../rf-eng/rf_unbalanced_engineered_y_*_*.json` / `.md`

## 8) Engineered XGBoost (`xgb-eng/`)

- Pattern: `../xgb-eng/xgb_unbalanced_engineered_y_*_*.json` / `.md`

## 9) Metadata + docs

- `../metadata/disk_first_seen_exp_time_generalisation*.json`
- `../metadata/model_mapping_exp_time_generalisation*.json`
- `../metadata/labeling_summary_smartlog*.json`
- `../docs/Data_Cleaning_Specification.md`
- `../docs/failure_labels_dedup_summary.md`
- `../docs/schema_drift_smartlog*.md`
- `../docs/labeling_summary_smartlog*.md`

## 10) Suggested next additions

- Engineered LR/RF for `y_7`, `y_14`, and both experiments, under `lr-eng/` and `rf-eng/`.
- XGBoost non-engineered runs under `xgb-no-eng/` (currently empty).
- Deep-learning reports in a new folder e.g. `reports/dl-eng/` or `reports/dl-no-eng/` with a stable naming pattern.
- One consolidated comparison markdown (before vs after engineering).

## Reruns and output paths

Training scripts in `src/` still default to writing filenames directly under `reports/` (not into `lr-no-eng/` / `rf-no-eng/`). After a rerun, either move new files into the matching subfolder or change `reports_dir` in those scripts so new artifacts land in the right place automatically.
