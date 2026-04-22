# Baseline Inventory Snapshot

This file freezes the benchmark outputs that existed when reports were organized into subfolders.
**Paths in the tables below are relative to the `reports/` directory** (repo: `FYP-CODE/reports/…`).

It is intended as the fixed reference point before full engineered reruns and deep-learning runs.

## Protocol Snapshot

- Train rows cap: `5,000,000`
- Val rows cap: `2,000,000`
- Test rows cap: `2,000,000`
- Horizons: `y_7`, `y_14`, `y_30`
- Experiments: `exp_time_generalisation`, `exp_time_generalisation__train_eval`
- Metrics expected in reports: `pr_auc`, `roc_auc`, `precision`, `recall`, `f1`

## Non-Engineered Baselines (Frozen)

| Experiment | Horizon | Model Family | Mode | Train | Val | Test | Timestamp | JSON Report | MD Report |
|---|---|---|---|---:|---:|---:|---|---|---|
| exp_time_generalisation | y_7 | LR+Naive | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-12T16:09:14.874523 | `lr-no-eng/baseline_y7_results_exp_time_generalisation.json` | `lr-no-eng/baseline_y7_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_7 | LR+Naive | balanced | 5000000 | 2000000 | 2000000 | 2026-02-12T16:43:44.093673 | `lr-no-eng/baseline_y7_lr_balanced_results_exp_time_generalisation.json` | `lr-no-eng/baseline_y7_lr_balanced_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_7 | RF | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-13T03:50:59.965411 | `rf-no-eng/baseline_rf_y_7_results_exp_time_generalisation.json` | `rf-no-eng/baseline_rf_y_7_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_7 | RF | balanced | 5000000 | 2000000 | 2000000 | 2026-02-13T03:56:15.799130 | `rf-no-eng/baseline_rf_y_7_balanced_results_exp_time_generalisation.json` | `rf-no-eng/baseline_rf_y_7_balanced_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_14 | LR+Naive | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-12T15:56:57.096186 | `lr-no-eng/baseline_y14_results_exp_time_generalisation.json` | `lr-no-eng/baseline_y14_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_14 | LR+Naive | balanced | 5000000 | 2000000 | 2000000 | 2026-02-12T17:07:05.549595 | `lr-no-eng/baseline_y14_lr_balanced_results_exp_time_generalisation.json` | `lr-no-eng/baseline_y14_lr_balanced_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_14 | RF | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-13T03:21:31.041163 | `rf-no-eng/baseline_rf_y_14_results_exp_time_generalisation.json` | `rf-no-eng/baseline_rf_y_14_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_14 | RF | balanced | 5000000 | 2000000 | 2000000 | 2026-02-13T03:27:09.819317 | `rf-no-eng/baseline_rf_y_14_balanced_results_exp_time_generalisation.json` | `rf-no-eng/baseline_rf_y_14_balanced_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_30 | LR+Naive | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-12T04:34:24.939780 | `lr-no-eng/baseline_y30_results_exp_time_generalisation.json` | `lr-no-eng/baseline_y30_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_30 | LR+Naive | balanced | 5000000 | 2000000 | 2000000 | 2026-02-12T15:24:00.772012 | `lr-no-eng/baseline_y30_lr_balanced_results_exp_time_generalisation.json` | `lr-no-eng/baseline_y30_lr_balanced_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_30 | RF | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-13T02:42:11.295079 | `rf-no-eng/baseline_rf_y_30_results_exp_time_generalisation.json` | `rf-no-eng/baseline_rf_y_30_results_exp_time_generalisation.md` |
| exp_time_generalisation | y_30 | RF | balanced | 5000000 | 2000000 | 2000000 | 2026-02-13T02:48:38.084053 | `rf-no-eng/baseline_rf_y_30_balanced_results_exp_time_generalisation.json` | `rf-no-eng/baseline_rf_y_30_balanced_results_exp_time_generalisation.md` |
| exp_time_generalisation__train_eval | y_7 | LR+Naive | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-12T16:14:18.560788 | `lr-no-eng/baseline_y7_results_exp_time_generalisation__train_eval.json` | `lr-no-eng/baseline_y7_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_7 | LR+Naive | balanced | 5000000 | 2000000 | 2000000 | 2026-02-12T16:49:52.754905 | `lr-no-eng/baseline_y7_lr_balanced_results_exp_time_generalisation__train_eval.json` | `lr-no-eng/baseline_y7_lr_balanced_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_7 | RF | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-13T04:02:43.852891 | `rf-no-eng/baseline_rf_y_7_results_exp_time_generalisation__train_eval.json` | `rf-no-eng/baseline_rf_y_7_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_7 | RF | balanced | 5000000 | 2000000 | 2000000 | 2026-02-13T04:10:50.817691 | `rf-no-eng/baseline_rf_y_7_balanced_results_exp_time_generalisation__train_eval.json` | `rf-no-eng/baseline_rf_y_7_balanced_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_14 | LR+Naive | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-12T16:02:21.034013 | `lr-no-eng/baseline_y14_results_exp_time_generalisation__train_eval.json` | `lr-no-eng/baseline_y14_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_14 | LR+Naive | balanced | 5000000 | 2000000 | 2000000 | 2026-02-12T17:12:09.925621 | `lr-no-eng/baseline_y14_lr_balanced_results_exp_time_generalisation__train_eval.json` | `lr-no-eng/baseline_y14_lr_balanced_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_14 | RF | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-13T03:33:25.452207 | `rf-no-eng/baseline_rf_y_14_results_exp_time_generalisation__train_eval.json` | `rf-no-eng/baseline_rf_y_14_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_14 | RF | balanced | 5000000 | 2000000 | 2000000 | 2026-02-13T03:39:31.721475 | `rf-no-eng/baseline_rf_y_14_balanced_results_exp_time_generalisation__train_eval.json` | `rf-no-eng/baseline_rf_y_14_balanced_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_30 | LR+Naive | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-12T04:40:55.829190 | `lr-no-eng/baseline_y30_results_exp_time_generalisation__train_eval.json` | `lr-no-eng/baseline_y30_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_30 | LR+Naive | balanced | 5000000 | 2000000 | 2000000 | 2026-02-12T15:28:35.794550 | `lr-no-eng/baseline_y30_lr_balanced_results_exp_time_generalisation__train_eval.json` | `lr-no-eng/baseline_y30_lr_balanced_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_30 | RF | imbalanced | 5000000 | 2000000 | 2000000 | 2026-02-13T02:55:40.894156 | `rf-no-eng/baseline_rf_y_30_results_exp_time_generalisation__train_eval.json` | `rf-no-eng/baseline_rf_y_30_results_exp_time_generalisation__train_eval.md` |
| exp_time_generalisation__train_eval | y_30 | RF | balanced | 5000000 | 2000000 | 2000000 | 2026-02-13T03:02:10.856303 | `rf-no-eng/baseline_rf_y_30_balanced_results_exp_time_generalisation__train_eval.json` | `rf-no-eng/baseline_rf_y_30_balanced_results_exp_time_generalisation__train_eval.md` |

## Engineered / pipeline-related runs (`lr-eng/`)

| Experiment | Description | Timestamp | JSON | MD |
|---|---|---|---|---|
| exp_time_generalisation | LR unbalanced on engineered data (`y_30`) | 2026-02-20T03:00:45.573581 | `lr-eng/lr_unbalanced_engineered_y30_exp_time_generalisation.json` | `lr-eng/lr_unbalanced_engineered_y30_exp_time_generalisation.md` |
| exp_time_generalisation | Tabular summary (`y_30`, full split stats — not the 5M cap table above) | 2026-02-12T03:50:12.775216 | `lr-eng/ml_tabular_y30_summary_exp_time_generalisation.json` | `lr-eng/ml_tabular_y30_summary_exp_time_generalisation.md` |
| exp_time_generalisation__train_eval | Tabular summary (`y_30`) | 2026-02-12T04:03:29.331651 | `lr-eng/ml_tabular_y30_summary_exp_time_generalisation__train_eval.json` | `lr-eng/ml_tabular_y30_summary_exp_time_generalisation__train_eval.md` |

## Notes

- `LR+Naive` means the LR baseline files also include `always_negative` (naive) metrics.
- Non-engineered LR lives in **`lr-no-eng/`**; non-engineered RF in **`rf-no-eng/`**; engineered LR/summary artifacts in **`lr-eng/`**.
- This inventory only updates paths for navigation; numeric results inside the JSON/MD files are unchanged.
- Navigation hub: [`REPORTS_INDEX.md`](./REPORTS_INDEX.md).
