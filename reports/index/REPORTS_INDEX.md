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
| [`lr-no_eng/`](../lr-no_eng/) | LR + Naive baselines on raw splits (non-engineered) |
| [`rf-no_eng/`](../rf-no_eng/) | Random Forest on raw splits (non-engineered) |
| [`lr-eng/`](../lr-eng/) | LR / tabular outputs on engineered data |
| [`rf-eng/`](../rf-eng/) | RF on engineered data (empty until you add runs) |
| `reports/*.md` (root) | Any docs left at top level (e.g. specs, drift) — see glob in repo |

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

### LR + Naive (`lr-no_eng/`)

- Pattern: `../lr-no_eng/baseline_y*_results_*.json` / `.md`
- Balanced LR: `../lr-no_eng/baseline_y*_lr_balanced_results_*.json` / `.md`

### Random Forest (`rf-no_eng/`)

- Pattern: `../rf-no_eng/baseline_rf_y_*_results_*.json` / `.md`
- Balanced RF: `../rf-no_eng/baseline_rf_y_*_balanced_results_*.json` / `.md`

Horizons appear in filenames (`y_7`, `y_14`, `y_30`). Experiments: `exp_time_generalisation`, `exp_time_generalisation__train_eval`.

## 6) Engineered model results (`lr-eng/`)

- [`../lr-eng/lr_unbalanced_engineered_y30_exp_time_generalisation.json`](../lr-eng/lr_unbalanced_engineered_y30_exp_time_generalisation.json) / [`.md`](../lr-eng/lr_unbalanced_engineered_y30_exp_time_generalisation.md)
- Tabular / pipeline summaries (engineered-related): `../lr-eng/ml_tabular_y30_summary_*.json` / `.md`

## 7) Engineered RF (`rf-eng/`)

- Empty until engineered RF reports are produced; place them here with a consistent naming scheme.

## 8) Suggested next additions

- Engineered LR/RF for `y_7`, `y_14`, and both experiments, under `lr-eng/` and `rf-eng/`.
- Deep-learning reports in a new folder e.g. `reports/dl-eng/` or `reports/dl-no_eng/` with a stable naming pattern.
- One consolidated comparison markdown (before vs after engineering).

## Reruns and output paths

Training scripts in `src/` still default to writing filenames directly under `reports/` (not into `lr-no_eng/` / `rf-no_eng/`). After a rerun, either move new files into the matching subfolder or change `reports_dir` in those scripts so new artifacts land in the right place automatically.
