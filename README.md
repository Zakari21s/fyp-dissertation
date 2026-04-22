# SSD Failure Prediction Dissertation Project (FYP-CODE)

End-to-end dissertation codebase for SSD failure prediction using SMART logs, with:

- data auditing and labeling,
- cleaning and QA,
- chronological split design,
- tabular dataset building and feature engineering,
- baseline and engineered model training/evaluation,
- structured report generation under `reports/`.

---

## 1) Project Goal

This project predicts storage device failure risk using SMART telemetry from two datasets:

- `smartlog2018ssd` (year 2018)
- `smartlog2019ssd` (year 2019)

and label file:

- `data_raw/ssd_failure_label.csv`

The pipeline is built to support reproducible experiments across horizons (`y_7`, `y_14`, `y_30`) and split protocols (notably `exp_time_generalisation` and `exp_time_generalisation__train_eval`).

---

## 2) Repository Structure

Top-level folders:

- `configs/` - YAML configuration files (`data_config.yaml`, `feature_set.yaml`)
- `src/` - pipeline, feature engineering, and modeling scripts
- `data_raw/` - raw SMART logs + failure labels
- `data_interim/` - staged intermediate outputs from cleaning/processing
- `data_clean/` - cleaned and labeled parquet outputs
- `data_splits/` - chronological train/val/test split artifacts
- `data_ml/` - tabular ML-ready datasets
- `data_engineered/` - engineered feature datasets
- `logs/` - execution logs
- `reports/` - all markdown/json/csv outputs for analysis and experiments

Reports are organized by domain/model family:

- `reports/index/` (navigation and inventory)
- `reports/cleaning/`, `reports/design/`, `reports/feature_ranking/`, `reports/feature_engineering/`
- `reports/lr-eng/`, `reports/lr-no-eng/`
- `reports/rf-eng/`, `reports/rf-no-eng/`
- `reports/xgb-eng/`, `reports/xgb-no-eng/`
- `reports/metadata/`, `reports/docs/`

---

## 3) Environment Setup

### Prerequisites

- Python environment (virtualenv recommended)
- Dependencies in `requirements.txt`

### Install

```bash
cd "/Users/slimanizakaria/Desktop/FYP-CODE"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main libraries:

- `pandas`, `numpy`, `pyarrow`
- `PyYAML`
- `scikit-learn`
- `xgboost`

---

## 4) Configuration

### `configs/data_config.yaml`

Controls data locations and processing defaults, including:

- dataset roots (`smartlog2018ssd`, `smartlog2019ssd`)
- key columns (`serial_number`, `date`, `model`)
- logging directory
- chunk and missingness settings

### `configs/feature_set.yaml`

Defines engineered feature behavior:

- selected SMART raw/normalized feature lists
- row caps per split (`train: 5,000,000`, `val: 2,000,000`, `test: 2,000,000`)
- temporal rolling settings (`rolling_window_days: 7`)
- dispersion mode (`std` / `var`)
- instability metric config

---

## 5) Pipeline Overview

Typical workflow from raw logs to final reports:

1. **Raw audit + label preparation**
   - `src/audit_raw.py`
   - `src/prepare_failure_labels.py`
   - `src/audit_labels.py`
2. **Build labeled dataset and cleaning**
   - `src/build_labeled_dataset.py`
   - `src/clean_labeled_pipeline.py`
   - `src/analyze_outliers.py`
3. **Split design + tabular dataset**
   - `src/make_time_splits.py`
   - `src/build_tabular_ml_dataset.py`
4. **Feature engineering + ranking**
   - `src/feature_engineering_pipeline.py`
   - `src/feature_ranking_y30.py`
   - `src/check_engineered_sample.py`
5. **Model training/evaluation**
   - Baselines:
     - `src/train_baselines_y30.py`
     - `src/train_baseline_y14.py`
     - `src/train_baseline_y7.py`
     - `src/run_lr_balanced_baseline.py`
     - `src/run_lr_balanced_baseline_y14.py`
     - `src/run_lr_balanced_baseline_y7.py`
     - `src/run_rf_baseline.py`
     - `src/run_rf_baseline_y14.py`
     - `src/run_rf_baseline_y7.py`
   - Engineered:
     - `src/run_lr_unbalanced_engineered.py`
     - `src/run_rf_unbalanced_engineered.py`
     - `src/run_xgb_unbalanced_engineered.py`

---

## 6) Example Run Commands

Run scripts from repository root:

```bash
cd "/Users/slimanizakaria/Desktop/FYP-CODE"
source .venv/bin/activate

python src/audit_raw.py
python src/prepare_failure_labels.py
python src/audit_labels.py

python src/build_labeled_dataset.py
python src/clean_labeled_pipeline.py
python src/analyze_outliers.py

python src/make_time_splits.py
python src/build_tabular_ml_dataset.py

python src/feature_engineering_pipeline.py
python src/feature_ranking_y30.py
python src/check_engineered_sample.py

python src/train_baselines_y30.py
python src/train_baseline_y14.py
python src/train_baseline_y7.py
python src/run_lr_balanced_baseline.py
python src/run_lr_balanced_baseline_y14.py
python src/run_lr_balanced_baseline_y7.py
python src/run_rf_baseline.py
python src/run_rf_baseline_y14.py
python src/run_rf_baseline_y7.py

python src/run_lr_unbalanced_engineered.py
python src/run_rf_unbalanced_engineered.py
python src/run_xgb_unbalanced_engineered.py
```

Notes:

- Most scripts write outputs under `reports/` and logs under `logs/`.
- If you rerun experiments, place new result files in the matching report subfolder (`lr-eng`, `rf-no-eng`, etc.) to preserve organization.

---

## 7) Experiments and Outputs

Primary reporting entry point:

- `reports/index/REPORTS_INDEX.md`

Frozen baseline inventory:

- `reports/index/baseline_inventory.md`

Model output conventions:

- non-engineered LR: `reports/lr-no-eng/`
- non-engineered RF: `reports/rf-no-eng/`
- engineered LR: `reports/lr-eng/`
- engineered RF: `reports/rf-eng/`
- engineered XGB: `reports/xgb-eng/`

Each run typically produces:

- `.json` (structured metrics/artifacts)
- `.md` (human-readable summary)

---

## 8) Reproducibility Notes

- Keep `configs/data_config.yaml` and `configs/feature_set.yaml` versioned.
- Preserve chronological split logic when comparing runs.
- Keep row caps consistent for fair baseline comparison.
- Track generated reports in the organized report subfolders.

---

## 9) Dissertation Context

This repository is the implementation backbone for the dissertation experiments and result tables/figures.  
It is structured to support:

- transparent pipeline stages,
- auditable outputs,
- repeatable model comparisons across horizons and feature sets.

