"""
Train and evaluate Logistic Regression on ENGINEERED data (tabular-equivalent LR + FE columns).

Matches pre-FE `train_baselines_y30` training logic by default:
- Features: `n_*` and `r_*` (same as tabular `build_xy`) plus engineered columns only
  (`log1p_*`, `delta*`, `roll*`, `instab_*`, `age_days`, `model_code`).
- Row caps: same pool rule as `train_baselines_y30.load_split_data_iterative` (keep positives,
  sample negatives, shuffle).
- `class_weight='balanced'`, `solver='liblinear'`, `max_iter=200` — same as tabular baseline LR.

Use `--no_class_weight` to disable balancing for an ablation.

Use `--feature-set` for FE ablations on the same `data_engineered` rows: `baseline` = `n_*`/`r_*`
only; `delta`, `rollmean`, `rollvar`, `instability`, `log1p` = baseline plus that family; `all` =
full engineered features (default; same column set as before this flag existed).

Optional `--train-neg-ratio N` (train only): after the row-capped pool, keep all positives and at
most N negatives per positive; val/test unchanged. Report/log filenames add `_tnr<N>` when set.

Saves: reports/lr_unbalanced_engineered_<label>_<experiment>.json and .md (suffixes: `_fe_<set>`,
`_tnr<N>` when applicable).

Horizons: use ``--label y_7``, ``--label y_14``, or ``--label y_30`` (parquet must contain that
column). Cross-horizon comparison (same FE + TNR as y_30): e.g. ``--feature-set baseline`` and
``--train-neg-ratio 2`` — see epilog on ``python -m src.run_lr_unbalanced_engineered --help``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


FEATURE_SET_CHOICES: Tuple[str, ...] = (
    "all",
    "baseline",
    "delta",
    "rollmean",
    "rollvar",
    "instability",
    "log1p",
)
def _is_excluded_id_or_label(col: str, label_col: str) -> bool:
    """IDs, label column, and other non-predictors."""
    if col == label_col or col in ("disk_id", "smart_day", "model", "ds"):
        return True
    if col.startswith("y_"):
        return True
    return False


def column_matches_feature_set(col: str, label_col: str, feature_set: str) -> bool:
    """
    Decide whether `col` is included in X for the given ablation mode.

    - baseline: only n_* / r_*
    - delta | rollmean | rollvar | instability | log1p: original + that FE family only
    - all: original + every engineered column (log1p, deltas, rolls, instab, age_days, model_code)
    """
    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")

    if _is_excluded_id_or_label(col, label_col):
        return False

    if feature_set == "all":
        if col.startswith("n_") or col.startswith("r_"):
            return True
        if col.startswith("log1p_") or col in ("age_days", "model_code"):
            return True
        if col.startswith("delta1_") or col.startswith("delta7_"):
            return True
        if col.startswith("rollmean7_") or col.startswith("rollstd7_") or col.startswith("rollvar7_"):
            return True
        if col.startswith("instab_"):
            return True
        return False

    if feature_set == "baseline":
        return col.startswith("n_") or col.startswith("r_")

    # Ablation: n_/r_/ + one FE family (exclude age_days, model_code, other families)
    if col.startswith("n_") or col.startswith("r_"):
        return True
    if feature_set == "delta":
        return col.startswith("delta")
    if feature_set == "rollmean":
        return col.startswith("rollmean7_")
    if feature_set == "rollvar":
        return col.startswith("rollvar7_") or col.startswith("rollstd7_")
    if feature_set == "instability":
        return col.startswith("instab_")
    if feature_set == "log1p":
        return col.startswith("log1p_")
    return False


@dataclass
class ModelMetrics:
    """Metrics for a single model on a single split."""
    split_name: str = ""
    accuracy: float = 0.0
    pr_auc: float = 0.0
    roc_auc: float = 0.0
    threshold: float = 0.5
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0


@dataclass
class ModelResults:
    """Results for a single model."""
    model_name: str = ""
    val_metrics: ModelMetrics = field(default_factory=lambda: ModelMetrics(split_name="val"))
    test_metrics: ModelMetrics = field(default_factory=lambda: ModelMetrics(split_name="test"))
    best_threshold: Optional[float] = None


@dataclass
class BaselineResults:
    """Overall results."""
    experiment_name: str = ""
    label_column: str = "y_30"
    train_rows: int = 0
    train_positives: int = 0
    train_negatives: int = 0
    val_rows: int = 0
    val_positives: int = 0
    val_negatives: int = 0
    test_rows: int = 0
    test_positives: int = 0
    test_negatives: int = 0
    feature_count: int = 0
    missingness_before_imputation: float = 0.0
    always_negative: ModelResults = field(default_factory=lambda: ModelResults(model_name="always_negative"))
    logistic_regression: ModelResults = field(default_factory=lambda: ModelResults(model_name="logistic_regression"))
    processing_time_seconds: float = 0.0
    # Keep default aligned with `train_baselines_y30`; disable via `--no_class_weight`.
    lr_class_weight: Optional[str] = "balanced"
    feature_set: str = "all"
    feature_names: List[str] = field(default_factory=list)
    # Optional train-only negative downsampling (`--train-neg-ratio`).
    train_neg_ratio: Optional[int] = None
    train_pool_rows: Optional[int] = None
    train_pool_positives: Optional[int] = None
    train_pool_negatives: Optional[int] = None


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(
    log_dir: Path,
    log_level: str,
    experiment_name: str,
    label_slug: str,
    run_suffix: str = "",
) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"lr_unbalanced_engineered_{label_slug}_{experiment_name}{run_suffix}.log"
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return logging.getLogger(__name__)


def find_partitions(input_dir: Path) -> List[Path]:
    partitions = []
    for year_dir in sorted(input_dir.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            if month_dir.is_dir():
                partitions.append(month_dir)
    return partitions


def load_split_streaming(
    split_dir: Path,
    max_rows: Optional[int],
    seed: int,
    label_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load data from data_engineered: same row-pool rule as train_baselines_y30.load_split_data_iterative.

    Whole parquet files are read in partition order until cumulative row count >= max_rows (or all
    files if max_rows is None). If the combined frame has more than max_rows rows, keep all
    positives, randomly sample negatives to fill max_rows, then shuffle — matching pre-FE tabular
    baseline construction for comparable label counts.
    """
    if not split_dir.exists():
        logger.warning("Split directory does not exist: %s", split_dir)
        return pd.DataFrame()

    partitions = find_partitions(split_dir)
    logger.info("Found %d partitions in %s", len(partitions), split_dir)

    chunks: List[pd.DataFrame] = []
    total_rows = 0

    for partition_dir in partitions:
        parquet_files = sorted(partition_dir.glob("*.parquet"))
        for path in parquet_files:
            try:
                df = pd.read_parquet(path)
                n = len(df)
                if n == 0:
                    continue
                chunks.append(df)
                total_rows += n
                if max_rows is not None and total_rows >= max_rows:
                    logger.info("Reached max_rows pool threshold (%s), stopping load", f"{max_rows:,}")
                    break
            except Exception as e:
                logger.error("Error reading %s: %s", path, e, exc_info=True)
        if max_rows is not None and total_rows >= max_rows:
            break

    if not chunks:
        return pd.DataFrame()

    df_combined = pd.concat(chunks, ignore_index=True)

    if max_rows is not None and len(df_combined) > max_rows:
        logger.info(
            "Downsampling pool of %s rows to %s (keep all positives, sample negatives, shuffle)",
            f"{len(df_combined):,}",
            f"{max_rows:,}",
        )
        if label_col not in df_combined.columns:
            logger.warning("Column '%s' not found, cannot preserve positives; random sample", label_col)
            df_combined = df_combined.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        else:
            positives = df_combined[df_combined[label_col] == 1].copy()
            negatives = df_combined[df_combined[label_col] == 0].copy()
            n_positives = len(positives)
            n_negatives_needed = max_rows - n_positives

            if n_negatives_needed < 0:
                logger.warning(
                    "More positives (%s) than max_rows (%s); subsampling positives only",
                    f"{n_positives:,}",
                    f"{max_rows:,}",
                )
                df_combined = positives.sample(n=max_rows, random_state=seed).reset_index(drop=True)
            else:
                if len(negatives) > n_negatives_needed:
                    negatives_sampled = negatives.sample(
                        n=n_negatives_needed, random_state=seed
                    ).reset_index(drop=True)
                else:
                    negatives_sampled = negatives.copy()
                df_combined = pd.concat([positives, negatives_sampled], ignore_index=True)
                df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)

        logger.info("After downsampling: %s rows", f"{len(df_combined):,}")

    logger.info("Loaded %s rows from %s", f"{len(df_combined):,}", split_dir)
    return df_combined


def downsample_train_negatives(
    df: pd.DataFrame,
    label_col: str,
    negatives_per_positive: int,
    seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Keep all positive rows; retain at most ``n_pos * negatives_per_positive`` negatives
    (sample without replacement). If fewer negatives exist than the cap, keep all negatives.
    Reproducible via ``seed`` for negative sampling and shuffle.
    """
    if negatives_per_positive < 1:
        raise ValueError(f"negatives_per_positive must be >= 1, got {negatives_per_positive}")
    if label_col not in df.columns:
        logger.warning("downsample_train_negatives: label %s missing; returning df unchanged", label_col)
        return df

    positives = df[df[label_col] == 1].copy()
    negatives = df[df[label_col] == 0].copy()
    n_pos = len(positives)
    n_neg_avail = len(negatives)
    target_neg = min(n_neg_avail, n_pos * negatives_per_positive)

    if n_pos == 0:
        logger.warning("Train neg downsampling: zero positives in train; returning df unchanged")
        return df

    if target_neg >= n_neg_avail:
        logger.info(
            "Train negative downsampling: target %s negatives (1:%s × %s pos) >= available %s; keeping all train rows",
            f"{n_pos * negatives_per_positive:,}",
            negatives_per_positive,
            f"{n_pos:,}",
            f"{n_neg_avail:,}",
        )
        return df

    neg_sampled = negatives.sample(n=target_neg, random_state=seed)
    out = pd.concat([positives, neg_sampled], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        "Train negative downsampling: kept all %s positives; sampled %s of %s negatives (1:%s requested)",
        f"{n_pos:,}",
        f"{target_neg:,}",
        f"{n_neg_avail:,}",
        negatives_per_positive,
    )
    return out


def build_xy(
    df: pd.DataFrame,
    label_col: str,
    logger: logging.Logger,
    feature_set: str = "all",
    log_feature_names: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build X from columns selected by ``feature_set``; y from ``label_col``."""
    feature_cols = [c for c in df.columns if column_matches_feature_set(c, label_col, feature_set)]
    if not feature_cols:
        logger.warning("No feature columns found (feature_set=%s)", feature_set)
        return pd.DataFrame(), pd.Series(dtype=float)
    feature_cols = sorted(feature_cols)
    X = df[feature_cols].copy()
    if label_col not in df.columns:
        logger.error("Label column '%s' not found", label_col)
        return pd.DataFrame(), pd.Series(dtype=float)
    y = df[label_col].copy()
    logger.info(
        "Built X with %d features (feature_set=%s), y with %d samples",
        len(feature_cols),
        feature_set,
        len(y),
    )
    if log_feature_names:
        for name in feature_cols:
            logger.info("  feature: %s", name)
    return X, y


def fit_preprocess(train_X: pd.DataFrame, logger: logging.Logger) -> Tuple[SimpleImputer, StandardScaler]:
    logger.info("Fitting imputer and scaler on training data...")
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_X)
    train_X_imputed = pd.DataFrame(
        imputer.transform(train_X),
        columns=train_X.columns,
        index=train_X.index,
    )
    scaler = StandardScaler()
    scaler.fit(train_X_imputed)
    logger.info("Fitted imputer and scaler")
    return imputer, scaler


def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    threshold: float = 0.5,
    logger: Optional[logging.Logger] = None,
) -> ModelMetrics:
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index,
    )
    X_scaled = pd.DataFrame(
        scaler.transform(X_imputed),
        columns=X.columns,
        index=X.index,
    )
    if isinstance(model, str) and model == "always_negative":
        y_pred_proba = np.zeros(len(y))
    else:
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = ModelMetrics(threshold=threshold)
    try:
        metrics.accuracy = float(accuracy_score(y, y_pred))
    except Exception:
        metrics.accuracy = 0.0
    try:
        metrics.pr_auc = float(average_precision_score(y, y_pred_proba))
    except Exception:
        metrics.pr_auc = 0.0
    try:
        metrics.roc_auc = float(roc_auc_score(y, y_pred_proba))
    except Exception:
        metrics.roc_auc = 0.0
    cm = confusion_matrix(y, y_pred)
    if cm.size == 4:
        metrics.tn, metrics.fp, metrics.fn, metrics.tp = map(int, cm.ravel())
    else:
        if len(np.unique(y_pred)) == 1:
            if int(y_pred.iloc[0]) == 0:
                metrics.tn = int((y == 0).sum())
                metrics.fn = int((y == 1).sum())
            else:
                metrics.fp = int((y == 0).sum())
                metrics.tp = int((y == 1).sum())
    try:
        metrics.precision = float(precision_score(y, y_pred, zero_division=0))
        metrics.recall = float(recall_score(y, y_pred, zero_division=0))
        metrics.f1 = float(f1_score(y, y_pred, zero_division=0))
    except Exception:
        metrics.precision = 0.0
        metrics.recall = 0.0
        metrics.f1 = 0.0
    return metrics


def find_best_threshold(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    threshold_scan_step: float = 0.01,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float]:
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index,
    )
    X_scaled = pd.DataFrame(
        scaler.transform(X_imputed),
        columns=X.columns,
        index=X.index,
    )
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    best_threshold = 0.5
    best_f1 = 0.0
    for t in np.arange(0.01, 1.0, threshold_scan_step):
        y_pred = (y_pred_proba >= t).astype(int)
        try:
            f1 = f1_score(y, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        except Exception:
            continue
    if logger:
        logger.info("Best threshold on validation: %.3f (F1=%.4f)", best_threshold, best_f1)
    return best_threshold, best_f1


def train_baselines(
    experiment_name: str,
    input_base_dir: Path,
    label_col: str,
    logger: logging.Logger,
    max_train_rows: Optional[int] = 5_000_000,
    max_val_rows: Optional[int] = 2_000_000,
    max_test_rows: Optional[int] = 2_000_000,
    seed: int = 42,
    threshold_scan_step: float = 0.01,
    class_weight: Optional[str] = "balanced",
    feature_set: str = "all",
    log_feature_names: bool = False,
    train_neg_ratio: Optional[int] = None,
) -> BaselineResults:
    start = datetime.now()
    results = BaselineResults(
        experiment_name=experiment_name,
        label_column=label_col,
        lr_class_weight=class_weight,
        feature_set=feature_set,
        train_neg_ratio=train_neg_ratio,
    )
    logger.info("=" * 60)
    logger.info(
        "LR on engineered data — feature_set=%s; class_weight=%s; train_neg_ratio=%s; "
        "row pools: keep positives + sample negatives to cap (tabular rule)",
        feature_set,
        repr(class_weight),
        repr(train_neg_ratio),
    )
    logger.info("=" * 60)
    logger.info("Experiment: %s", experiment_name)
    logger.info("Label: %s", label_col)
    logger.info("Using max_train_rows = %s", f"{max_train_rows:,}" if max_train_rows is not None else "None (no cap)")
    logger.info("Seed: %s", seed)
    logger.info("Input base: %s", input_base_dir)

    input_experiment_dir = input_base_dir / experiment_name
    if not input_experiment_dir.exists():
        raise FileNotFoundError(f"Input experiment directory not found: {input_experiment_dir}")

    np.random.seed(seed)

    # Train
    logger.info("Loading training data (streaming, max_train_rows=%s)...", f"{max_train_rows:,}" if max_train_rows is not None else "None")
    train_dir = input_experiment_dir / "train"
    train_df = load_split_streaming(train_dir, max_train_rows, seed, label_col, logger)
    if len(train_df) == 0:
        raise ValueError("No training data loaded")

    if train_neg_ratio is not None:
        if label_col not in train_df.columns:
            raise ValueError(f"Label column '{label_col}' not in training data")
        results.train_pool_rows = len(train_df)
        results.train_pool_positives = int((train_df[label_col] == 1).sum())
        results.train_pool_negatives = int((train_df[label_col] == 0).sum())
        logger.info(
            "Train pool before negative downsampling: %s rows (%s pos, %s neg)",
            f"{results.train_pool_rows:,}",
            f"{results.train_pool_positives:,}",
            f"{results.train_pool_negatives:,}",
        )
        train_df = downsample_train_negatives(train_df, label_col, train_neg_ratio, seed, logger)
        logger.info(
            "Train negative downsampling: enabled | requested ratio 1:%s | after sampling: %s rows (%s pos, %s neg)",
            train_neg_ratio,
            f"{len(train_df):,}",
            f"{int((train_df[label_col] == 1).sum()):,}",
            f"{int((train_df[label_col] == 0).sum()):,}",
        )
    else:
        logger.info("Train negative downsampling: disabled (use --train-neg-ratio N to enable)")

    train_X, train_y = build_xy(
        train_df, label_col, logger, feature_set=feature_set, log_feature_names=log_feature_names
    )
    if len(train_X) == 0:
        raise ValueError("No features in training data")

    results.train_rows = len(train_df)
    results.train_positives = int((train_y == 1).sum())
    results.train_negatives = int((train_y == 0).sum())
    results.feature_count = len(train_X.columns)
    results.feature_names = list(train_X.columns)
    missing_count = train_X.isna().sum().sum()
    total_cells = len(train_X) * len(train_X.columns)
    results.missingness_before_imputation = (missing_count / total_cells * 100) if total_cells > 0 else 0.0
    logger.info("Train: %s rows (%s pos, %s neg), %s features, missingness %.2f%%",
                f"{results.train_rows:,}", f"{results.train_positives:,}", f"{results.train_negatives:,}",
                results.feature_count, results.missingness_before_imputation)

    imputer, scaler = fit_preprocess(train_X, logger)
    logger.info("Training Logistic Regression (solver=liblinear, max_iter=200, class_weight=%s)...", repr(class_weight))
    train_X_imputed = pd.DataFrame(
        imputer.transform(train_X),
        columns=train_X.columns,
        index=train_X.index,
    )
    train_X_scaled = pd.DataFrame(
        scaler.transform(train_X_imputed),
        columns=train_X.columns,
        index=train_X.index,
    )
    lr_model = LogisticRegression(
        class_weight=class_weight,
        solver="liblinear",
        max_iter=200,
        random_state=seed,
    )
    logger.info(
        "Fitting LR on %s × %d matrix (can take many minutes on large train sets)",
        f"{len(train_X_scaled):,}",
        train_X_scaled.shape[1],
    )
    lr_model.fit(train_X_scaled, train_y)
    logger.info("LR trained")

    # Val
    logger.info("Loading validation data (max_rows=%s)...", max_val_rows)
    val_dir = input_experiment_dir / "val"
    val_df = load_split_streaming(val_dir, max_val_rows, seed, label_col, logger)
    if len(val_df) > 0:
        val_X, val_y = build_xy(
            val_df, label_col, logger, feature_set=feature_set, log_feature_names=False
        )
        results.val_rows = len(val_df)
        results.val_positives = int((val_y == 1).sum())
        results.val_negatives = int((val_y == 0).sum())
        logger.info("Val: %s rows (%s pos, %s neg)", f"{results.val_rows:,}", results.val_positives, results.val_negatives)

        results.always_negative.val_metrics = evaluate(
            "always_negative", val_X, val_y, imputer, scaler, threshold=0.5, logger=logger
        )
        results.always_negative.val_metrics.split_name = "val"

        results.logistic_regression.val_metrics = evaluate(
            lr_model, val_X, val_y, imputer, scaler, threshold=0.5, logger=logger
        )
        results.logistic_regression.val_metrics.split_name = "val"

        best_threshold, _ = find_best_threshold(
            lr_model, val_X, val_y, imputer, scaler, threshold_scan_step, logger
        )
        results.logistic_regression.best_threshold = best_threshold
        results.logistic_regression.val_metrics = evaluate(
            lr_model, val_X, val_y, imputer, scaler, threshold=best_threshold, logger=logger
        )
        results.logistic_regression.val_metrics.split_name = "val"
        results.logistic_regression.val_metrics.threshold = best_threshold

    # Test
    logger.info("Loading test data (max_rows=%s)...", max_test_rows)
    test_dir = input_experiment_dir / "test"
    test_df = load_split_streaming(test_dir, max_test_rows, seed, label_col, logger)
    if len(test_df) > 0:
        test_X, test_y = build_xy(
            test_df, label_col, logger, feature_set=feature_set, log_feature_names=False
        )
        results.test_rows = len(test_df)
        results.test_positives = int((test_y == 1).sum())
        results.test_negatives = int((test_y == 0).sum())
        logger.info("Test: %s rows (%s pos, %s neg)", f"{results.test_rows:,}", results.test_positives, results.test_negatives)

        results.always_negative.test_metrics = evaluate(
            "always_negative", test_X, test_y, imputer, scaler, threshold=0.5, logger=logger
        )
        results.always_negative.test_metrics.split_name = "test"

        test_threshold = results.logistic_regression.best_threshold if results.logistic_regression.best_threshold is not None else 0.5
        logger.info("Evaluating LR on test with threshold %.3f", test_threshold)
        results.logistic_regression.test_metrics = evaluate(
            lr_model, test_X, test_y, imputer, scaler, threshold=test_threshold, logger=logger
        )
        results.logistic_regression.test_metrics.split_name = "test"
        results.logistic_regression.test_metrics.threshold = test_threshold

    results.processing_time_seconds = (datetime.now() - start).total_seconds()
    logger.info("Processing time: %.2fs", results.processing_time_seconds)
    return results


def _report_file_stem(
    label_slug: str,
    experiment_name: str,
    feature_set: str,
    train_neg_ratio: Optional[int] = None,
) -> str:
    """`all` + no downsampling keeps the original basename; otherwise add `_fe_*` / `_tnr_*` suffixes."""
    parts: List[str] = []
    if feature_set != "all":
        parts.append(f"_fe_{feature_set}")
    if train_neg_ratio is not None:
        parts.append(f"_tnr{train_neg_ratio}")
    return f"lr_unbalanced_engineered_{label_slug}_{experiment_name}{''.join(parts)}"


def write_reports(
    results: BaselineResults, experiment_name: str, label_slug: str, logger: logging.Logger
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = _report_file_stem(
        label_slug, experiment_name, results.feature_set, results.train_neg_ratio
    )
    base = reports_dir / stem

    train_downsample_block: Dict[str, Any] = {
        "enabled": results.train_neg_ratio is not None,
        "negatives_per_positive_requested": results.train_neg_ratio,
    }
    if results.train_neg_ratio is not None and results.train_pool_rows is not None:
        train_downsample_block["train_pool_before_downsample"] = {
            "rows": results.train_pool_rows,
            "positives": results.train_pool_positives,
            "negatives": results.train_pool_negatives,
        }
        train_downsample_block["note"] = (
            "Train rows after downsampling are in data_summary.train (used for LR fit). "
            "Val/test unchanged."
        )
    report_json = {
        "experiment_name": results.experiment_name,
        "timestamp": datetime.now().isoformat(),
        "label_column": results.label_column,
        "data_source": "data_engineered",
        "class_weight": results.lr_class_weight,
        "feature_set": results.feature_set,
        "feature_policy": "column_matches_feature_set",
        "feature_names": results.feature_names,
        "row_cap_policy": "keep_all_positives_sample_negatives_shuffle",
        "train_negative_downsampling": train_downsample_block,
        "data_summary": {
            "train": {"rows": results.train_rows, "positives": results.train_positives, "negatives": results.train_negatives},
            "val": {"rows": results.val_rows, "positives": results.val_positives, "negatives": results.val_negatives},
            "test": {"rows": results.test_rows, "positives": results.test_positives, "negatives": results.test_negatives},
        },
        "feature_info": {
            "feature_count": results.feature_count,
            "missingness_before_imputation_pct": results.missingness_before_imputation,
        },
        "models": {
            "always_negative": {
                "val": asdict(results.always_negative.val_metrics),
                "test": asdict(results.always_negative.test_metrics),
            },
            "logistic_regression": {
                "class_weight": results.lr_class_weight,
                "best_threshold": results.logistic_regression.best_threshold,
                "val": asdict(results.logistic_regression.val_metrics),
                "test": asdict(results.logistic_regression.test_metrics),
            },
        },
        "processing_time_seconds": results.processing_time_seconds,
    }
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(report_json, f, indent=2, default=str)
    logger.info("Wrote %s", base.with_suffix(".json"))

    def _metrics_line(m: ModelMetrics, prefix: str = ""):
        return (
            f"{prefix}- Accuracy: {m.accuracy:.4f}\n"
            f"{prefix}- Precision: {m.precision:.4f}\n"
            f"{prefix}- Recall: {m.recall:.4f}\n"
            f"{prefix}- F1: {m.f1:.4f}\n"
            f"{prefix}- ROC-AUC: {m.roc_auc:.4f}\n"
            f"{prefix}- PR-AUC: {m.pr_auc:.4f}\n"
            f"{prefix}- Threshold: {m.threshold:.3f}\n"
            f"{prefix}- TN/FP/FN/TP: {m.tn}/{m.fp}/{m.fn}/{m.tp}\n"
        )

    cw = results.lr_class_weight or "none"
    md = [
        f"# LR Engineered (tabular-equivalent + FE) – {experiment_name}",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"- **Label:** `{results.label_column}`",
        f"- **Data source:** data_engineered",
        f"- **Feature set mode:** `{results.feature_set}` (see `column_matches_feature_set` in script)",
        f"- **Class weight:** `{cw}` (matches pre-FE `train_baselines_y30` when `balanced`)",
        "- **Row cap policy:** load parquet pool in hive order until cumulative rows ≥ cap; then keep all positives, sample negatives to cap, shuffle (same as pre-FE `train_baselines_y30` loaders).",
        "",
        "## Train negative downsampling",
        "",
    ]
    if results.train_neg_ratio is not None and results.train_pool_rows is not None:
        md.extend(
            [
                f"- **Status:** enabled",
                f"- **Requested ratio:** 1:{results.train_neg_ratio} (negatives per positive)",
                f"- **Original train pool (after row cap, before neg downsampling):** {results.train_pool_rows:,} rows — {results.train_pool_positives:,} pos / {results.train_pool_negatives:,} neg",
                f"- **Sampled train (used for LR fit):** {results.train_rows:,} rows — {results.train_positives:,} pos / {results.train_negatives:,} neg",
                "",
            ]
        )
    else:
        md.extend(
            [
                "- **Status:** disabled (entire capped train pool used for fitting)",
                "",
            ]
        )
    md.extend(
        [
        "## Data summary",
        "",
        "| Split | Rows | Positives | Negatives |",
        "|-------|------|-----------|-----------|",
        f"| Train (used for fit) | {results.train_rows:,} | {results.train_positives:,} | {results.train_negatives:,} |",
        f"| Val   | {results.val_rows:,} | {results.val_positives:,} | {results.val_negatives:,} |",
        f"| Test  | {results.test_rows:,} | {results.test_positives:,} | {results.test_negatives:,} |",
        "",
        f"**Features:** {results.feature_count}",
        f"**Missingness (before imputation):** {results.missingness_before_imputation:.2f}%",
        "",
        "## Always-negative",
        "",
        "**Val:**",
        _metrics_line(results.always_negative.val_metrics, "  "),
        "",
        "**Test:**",
        _metrics_line(results.always_negative.test_metrics, "  "),
        "",
        "## Logistic Regression",
        "",
        f"**Best threshold (from val):** {results.logistic_regression.best_threshold}",
        "",
        "**Val:**",
        _metrics_line(results.logistic_regression.val_metrics, "  "),
        "",
        "**Test:**",
        _metrics_line(results.logistic_regression.test_metrics, "  "),
        "",
        f"**Processing time:** {results.processing_time_seconds:.2f}s",
        "",
        "## Feature names (reproducibility)",
        "",
        "Sorted column names used in X (same order as training matrix):",
        "",
        "```",
        *results.feature_names,
        "```",
        ]
    )
    with open(base.with_suffix(".md"), "w") as f:
        f.write("\n".join(md))
    logger.info("Wrote %s", base.with_suffix(".md"))


def _epilog_six_horizon_runs() -> str:
    """CLI examples for y_7 and y_14 runs with TNR=2."""
    base = (
        "python -m src.run_lr_unbalanced_engineered "
        "--experiment exp_time_generalisation --train-neg-ratio 2"
    )
    lines = [
        "Cross-horizon LR (engineered data): y_7 and y_14, feature sets baseline / rollmean / rollvar, TNR=2.",
        "Requires y_7, y_14, y_30 columns in data_engineered parquet.",
        "",
        "y_7:",
        f"  {base} --label y_7 --feature-set baseline",
        f"  {base} --label y_7 --feature-set rollmean",
        f"  {base} --label y_7 --feature-set rollvar",
        "",
        "y_14:",
        f"  {base} --label y_14 --feature-set baseline",
        f"  {base} --label y_14 --feature-set rollmean",
        f"  {base} --label y_14 --feature-set rollvar",
        "",
        "Reports: reports/lr_unbalanced_engineered_<label>_exp_time_generalisation_fe_<set>_tnr2.{md,json}",
        "Default --label y_30 and --feature-set all are unchanged for backward compatibility.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train LR on data_engineered. By default matches pre-FE train_baselines_y30: "
            "class_weight=balanced, liblinear, n_*/r_* + engineered features only, same row-cap sampling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog_six_horizon_runs(),
    )
    parser.add_argument("--experiment", type=str, default="exp_time_generalisation", help="Experiment name")
    parser.add_argument(
        "--label",
        type=str,
        default="y_30",
        metavar="COL",
        help=(
            "Target / horizon label column. Use y_7, y_14, or y_30 for failure horizons; "
            "must exist in engineered parquet. Default y_30 (unchanged)."
        ),
    )
    parser.add_argument(
        "--no_class_weight",
        action="store_true",
        help="Disable class_weight (ablation). Default is balanced to match train_baselines_y30.",
    )
    parser.add_argument("--max_train_rows", type=int, default=5_000_000, help="Cap train rows (memory-safe)")
    parser.add_argument("--max_val_rows", type=int, default=2_000_000, help="Cap val rows")
    parser.add_argument("--max_test_rows", type=int, default=2_000_000, help="Cap test rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold_scan_step", type=float, default=0.01)
    parser.add_argument(
        "--feature-set",
        type=str,
        default="all",
        choices=list(FEATURE_SET_CHOICES),
        metavar="SET",
        help=(
            "Which columns to include: "
            "`baseline` = n_/r_ only; "
            "`delta` | `rollmean` | `rollvar` | `instability` | `log1p` = baseline + that FE family; "
            "`all` = full engineered table (default, same as before)."
        ),
    )
    parser.add_argument(
        "--log-feature-names",
        action="store_true",
        help="Log each selected feature column name at INFO when building the training matrix.",
    )
    parser.add_argument(
        "--train-neg-ratio",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Train split only: keep all positives, then sample at most N negatives per positive "
            "(after the usual row-cap pool). Reproducible with --seed. Val/test unchanged. "
            "Omit for legacy behaviour (use full capped train pool)."
        ),
    )
    args = parser.parse_args()

    if args.train_neg_ratio is not None and args.train_neg_ratio < 1:
        print("Error: --train-neg-ratio must be >= 1 when provided", file=sys.stderr)
        sys.exit(2)

    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "configs" / "data_config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
    config = load_config(config_path)
    log_dir = base_dir / config["logging"]["log_dir"]
    label_slug = args.label.replace(".", "_")
    run_suffix_parts: List[str] = []
    if args.feature_set != "all":
        run_suffix_parts.append(f"_fe_{args.feature_set}")
    if args.train_neg_ratio is not None:
        run_suffix_parts.append(f"_tnr{args.train_neg_ratio}")
    run_suffix = "".join(run_suffix_parts)
    logger = setup_logging(
        log_dir, config["logging"]["level"], args.experiment, label_slug, run_suffix=run_suffix
    )

    # Engineered features live under `data_engineered`.
    input_base_dir = base_dir / "data_engineered"

    try:
        results = train_baselines(
            args.experiment,
            input_base_dir,
            args.label,
            logger,
            max_train_rows=args.max_train_rows,
            max_val_rows=args.max_val_rows,
            max_test_rows=args.max_test_rows,
            seed=args.seed,
            threshold_scan_step=args.threshold_scan_step,
            class_weight=None if args.no_class_weight else "balanced",
            feature_set=args.feature_set,
            log_feature_names=args.log_feature_names,
            train_neg_ratio=args.train_neg_ratio,
        )
        write_reports(results, args.experiment, label_slug, logger)
        logger.info(
            "SUCCESS: LR engineered (feature_set=%s, train_neg_ratio=%s) complete",
            args.feature_set,
            repr(args.train_neg_ratio),
        )
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
