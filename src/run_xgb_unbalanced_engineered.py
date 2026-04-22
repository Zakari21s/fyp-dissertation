"""
XGBoost on data_engineered — mirrors ``run_lr_unbalanced_engineered`` / ``run_rf_unbalanced_engineered``.

Same loaders, feature-set policy, row caps, train negative downsampling, median imputer + StandardScaler,
threshold scan on validation (max F1), then test.

Imbalance: ``--imbalance-mode none`` (no scale_pos_weight) or ``auto`` (scale_pos_weight = neg/pos on
final train matrix). Reports/logs: ``xgb_unbalanced_engineered_..._imnone`` / ``_imauto``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.run_lr_unbalanced_engineered import (
    FEATURE_SET_CHOICES,
    ModelMetrics,
    ModelResults,
    build_xy,
    downsample_train_negatives,
    evaluate,
    find_best_threshold,
    fit_preprocess,
    load_config,
    load_split_streaming,
)


@dataclass
class XGBExperimentResults:
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
    xgboost_model: ModelResults = field(default_factory=lambda: ModelResults(model_name="xgboost"))
    processing_time_seconds: float = 0.0
    imbalance_mode: str = "none"  # "none" | "auto"
    scale_pos_weight_used: Optional[float] = None
    feature_set: str = "all"
    feature_names: List[str] = field(default_factory=list)
    train_neg_ratio: Optional[int] = None
    train_pool_rows: Optional[int] = None
    train_pool_positives: Optional[int] = None
    train_pool_negatives: Optional[int] = None
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8


def setup_logging(
    log_dir: Path,
    log_level: str,
    experiment_name: str,
    label_slug: str,
    run_suffix: str = "",
) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"xgb_unbalanced_engineered_{label_slug}_{experiment_name}{run_suffix}.log"
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


def _report_file_stem(
    label_slug: str,
    experiment_name: str,
    feature_set: str,
    train_neg_ratio: Optional[int],
    imbalance_slug: str,
) -> str:
    parts: List[str] = []
    if feature_set != "all":
        parts.append(f"_fe_{feature_set}")
    if train_neg_ratio is not None:
        parts.append(f"_tnr{train_neg_ratio}")
    parts.append(f"_{imbalance_slug}")
    return f"xgb_unbalanced_engineered_{label_slug}_{experiment_name}{''.join(parts)}"


def _import_xgb_classifier():
    """Lazy import so ``--help`` works without xgboost installed."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError as e:
        sys.exit(
            "xgboost is not installed. Install with:\n"
            "  pip install xgboost\n"
            "or:\n"
            "  pip install -r requirements.txt\n"
            f"Original error: {e}"
        )


def _compute_scale_pos_weight(train_y: pd.Series, logger: logging.Logger) -> float:
    n_pos = int((train_y == 1).sum())
    n_neg = int((train_y == 0).sum())
    if n_pos <= 0:
        logger.warning("scale_pos_weight: zero positives in train; using 1.0")
        return 1.0
    spw = float(n_neg) / float(n_pos)
    logger.info(
        "imbalance-mode auto: scale_pos_weight = neg/pos = %s / %s = %.6f",
        f"{n_neg:,}",
        f"{n_pos:,}",
        spw,
    )
    return spw


def train_xgb_baselines(
    experiment_name: str,
    input_base_dir: Path,
    label_col: str,
    logger: logging.Logger,
    max_train_rows: Optional[int] = 5_000_000,
    max_val_rows: Optional[int] = 2_000_000,
    max_test_rows: Optional[int] = 2_000_000,
    seed: int = 42,
    threshold_scan_step: float = 0.01,
    imbalance_mode: str = "none",
    feature_set: str = "all",
    log_feature_names: bool = False,
    train_neg_ratio: Optional[int] = None,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> XGBExperimentResults:
    start = datetime.now()
    results = XGBExperimentResults(
        experiment_name=experiment_name,
        label_column=label_col,
        imbalance_mode=imbalance_mode,
        feature_set=feature_set,
        train_neg_ratio=train_neg_ratio,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )
    logger.info("=" * 60)
    logger.info(
        "XGBoost on engineered data — feature_set=%s; imbalance_mode=%s; train_neg_ratio=%s",
        feature_set,
        imbalance_mode,
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

    logger.info(
        "Loading training data (streaming, max_train_rows=%s)...",
        f"{max_train_rows:,}" if max_train_rows is not None else "None",
    )
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
            "Train after neg downsampling: %s rows (%s pos, %s neg)",
            f"{len(train_df):,}",
            f"{int((train_df[label_col] == 1).sum()):,}",
            f"{int((train_df[label_col] == 0).sum()):,}",
        )
    else:
        logger.info("Train negative downsampling: disabled")

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
    logger.info(
        "Train: %s rows (%s pos, %s neg), %s features, missingness %.2f%%",
        f"{results.train_rows:,}",
        f"{results.train_positives:,}",
        f"{results.train_negatives:,}",
        results.feature_count,
        results.missingness_before_imputation,
    )

    imputer, scaler = fit_preprocess(train_X, logger)
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

    scale_pos_weight: Optional[float] = None
    if imbalance_mode == "auto":
        scale_pos_weight = _compute_scale_pos_weight(train_y, logger)
        results.scale_pos_weight_used = scale_pos_weight
    elif imbalance_mode != "none":
        raise ValueError(f"Unknown imbalance_mode: {imbalance_mode!r}")
    else:
        logger.info("imbalance-mode none: not setting scale_pos_weight (XGBoost default 1.0)")

    xgb_kwargs: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": -1,
    }
    if scale_pos_weight is not None:
        xgb_kwargs["scale_pos_weight"] = scale_pos_weight

    XGBClassifier = _import_xgb_classifier()
    logger.info(
        "Fitting XGBClassifier (same imputer+StandardScaler as LR/RF) with kwargs: %s",
        {k: v for k, v in xgb_kwargs.items() if k != "random_state"},
    )
    xgb_model = XGBClassifier(**xgb_kwargs)
    logger.info(
        "Fitting XGB on %s × %d matrix",
        f"{len(train_X_scaled):,}",
        train_X_scaled.shape[1],
    )
    xgb_model.fit(train_X_scaled, train_y)
    logger.info("XGBoost trained")

    val_dir = input_experiment_dir / "val"
    val_df = load_split_streaming(val_dir, max_val_rows, seed, label_col, logger)
    if len(val_df) > 0:
        val_X, val_y = build_xy(val_df, label_col, logger, feature_set=feature_set, log_feature_names=False)
        results.val_rows = len(val_df)
        results.val_positives = int((val_y == 1).sum())
        results.val_negatives = int((val_y == 0).sum())
        logger.info("Val: %s rows", f"{results.val_rows:,}")

        results.always_negative.val_metrics = evaluate(
            "always_negative", val_X, val_y, imputer, scaler, threshold=0.5, logger=logger
        )
        results.always_negative.val_metrics.split_name = "val"

        results.xgboost_model.val_metrics = evaluate(
            xgb_model, val_X, val_y, imputer, scaler, threshold=0.5, logger=logger
        )
        results.xgboost_model.val_metrics.split_name = "val"

        best_threshold, _ = find_best_threshold(
            xgb_model, val_X, val_y, imputer, scaler, threshold_scan_step, logger
        )
        results.xgboost_model.best_threshold = best_threshold
        results.xgboost_model.val_metrics = evaluate(
            xgb_model, val_X, val_y, imputer, scaler, threshold=best_threshold, logger=logger
        )
        results.xgboost_model.val_metrics.split_name = "val"
        results.xgboost_model.val_metrics.threshold = best_threshold

    test_dir = input_experiment_dir / "test"
    test_df = load_split_streaming(test_dir, max_test_rows, seed, label_col, logger)
    if len(test_df) > 0:
        test_X, test_y = build_xy(test_df, label_col, logger, feature_set=feature_set, log_feature_names=False)
        results.test_rows = len(test_df)
        results.test_positives = int((test_y == 1).sum())
        results.test_negatives = int((test_y == 0).sum())
        logger.info("Test: %s rows", f"{results.test_rows:,}")

        results.always_negative.test_metrics = evaluate(
            "always_negative", test_X, test_y, imputer, scaler, threshold=0.5, logger=logger
        )
        results.always_negative.test_metrics.split_name = "test"

        test_threshold = (
            results.xgboost_model.best_threshold
            if results.xgboost_model.best_threshold is not None
            else 0.5
        )
        logger.info("Evaluating XGB on test with threshold %.3f", test_threshold)
        results.xgboost_model.test_metrics = evaluate(
            xgb_model, test_X, test_y, imputer, scaler, threshold=test_threshold, logger=logger
        )
        results.xgboost_model.test_metrics.split_name = "test"
        results.xgboost_model.test_metrics.threshold = test_threshold

    results.processing_time_seconds = (datetime.now() - start).total_seconds()
    logger.info("Processing time: %.2fs", results.processing_time_seconds)
    return results


def write_reports(
    results: XGBExperimentResults,
    experiment_name: str,
    label_slug: str,
    imbalance_slug: str,
    logger: logging.Logger,
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = _report_file_stem(
        label_slug,
        experiment_name,
        results.feature_set,
        results.train_neg_ratio,
        imbalance_slug,
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
            "Train rows after downsampling are in data_summary.train (used for XGB fit). Val/test unchanged."
        )

    report_json = {
        "experiment_name": results.experiment_name,
        "timestamp": datetime.now().isoformat(),
        "model": "xgboost",
        "label_column": results.label_column,
        "data_source": "data_engineered",
        "imbalance_mode": results.imbalance_mode,
        "scale_pos_weight": results.scale_pos_weight_used,
        "xgboost_hyperparameters": {
            "n_estimators": results.n_estimators,
            "max_depth": results.max_depth,
            "learning_rate": results.learning_rate,
            "subsample": results.subsample,
            "colsample_bytree": results.colsample_bytree,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
        },
        "feature_set": results.feature_set,
        "feature_policy": "column_matches_feature_set (shared with LR/RF runners)",
        "feature_names": results.feature_names,
        "row_cap_policy": "keep_all_positives_sample_negatives_shuffle",
        "preprocessing": (
            "SimpleImputer(strategy='median') + StandardScaler — same as LR and RF engineered runners "
            "for fair comparison (explicit in logs/reports)."
        ),
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
            "xgboost": {
                "imbalance_mode": results.imbalance_mode,
                "scale_pos_weight": results.scale_pos_weight_used,
                "best_threshold": results.xgboost_model.best_threshold,
                "val": asdict(results.xgboost_model.val_metrics),
                "test": asdict(results.xgboost_model.test_metrics),
            },
        },
        "processing_time_seconds": results.processing_time_seconds,
    }
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(report_json, f, indent=2, default=str)
    logger.info("Wrote %s", base.with_suffix(".json"))

    def _metrics_line(m: ModelMetrics, prefix: str = "") -> str:
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

    spw_line = (
        f"- **scale_pos_weight (used):** `{results.scale_pos_weight_used}`\n"
        if results.scale_pos_weight_used is not None
        else "- **scale_pos_weight:** not set (default 1.0)\n"
    )
    md = [
        f"# XGBoost (engineered data) – {experiment_name}",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"- **Label:** `{results.label_column}`",
        f"- **Data source:** data_engineered",
        f"- **Feature set mode:** `{results.feature_set}`",
        f"- **Imbalance mode:** `{results.imbalance_mode}` (`none` = no scale_pos_weight; `auto` = neg/pos on final train)",
        spw_line.rstrip(),
        f"- **XGB:** n_estimators={results.n_estimators}, max_depth={results.max_depth}, "
        f"learning_rate={results.learning_rate}, subsample={results.subsample}, colsample_bytree={results.colsample_bytree}",
        "- **Preprocessing:** median imputation + **StandardScaler** (same as LR/RF engineered runners).",
        "- **Row cap policy:** same as LR/RF.",
        "",
        "## Train negative downsampling",
        "",
    ]
    if results.train_neg_ratio is not None and results.train_pool_rows is not None:
        md.extend(
            [
                "- **Status:** enabled",
                f"- **Requested ratio:** 1:{results.train_neg_ratio}",
                f"- **Train pool before neg downsampling:** {results.train_pool_rows:,} rows — {results.train_pool_positives:,} pos / {results.train_pool_negatives:,} neg",
                f"- **Train used for fit:** {results.train_rows:,} rows — {results.train_positives:,} pos / {results.train_negatives:,} neg",
                "",
            ]
        )
    else:
        md.extend(["- **Status:** disabled", ""])
    md.extend(
        [
            "## Data summary",
            "",
            "| Split | Rows | Positives | Negatives |",
            "|-------|------|-----------|-----------|",
            f"| Train (fit) | {results.train_rows:,} | {results.train_positives:,} | {results.train_negatives:,} |",
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
            "## XGBoost",
            "",
            f"**Best threshold (from val):** {results.xgboost_model.best_threshold}",
            "",
            "**Val:**",
            _metrics_line(results.xgboost_model.val_metrics, "  "),
            "",
            "**Test:**",
            _metrics_line(results.xgboost_model.test_metrics, "  "),
            "",
            f"**Processing time:** {results.processing_time_seconds:.2f}s",
            "",
            "## Feature names",
            "",
            "```",
            *results.feature_names,
            "```",
        ]
    )
    with open(base.with_suffix(".md"), "w") as f:
        f.write("\n".join(md))
    logger.info("Wrote %s", base.with_suffix(".md"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost on data_engineered (same workflow as LR/RF engineered runners).",
    )
    parser.add_argument("--experiment", type=str, default="exp_time_generalisation")
    parser.add_argument(
        "--label",
        type=str,
        default="y_30",
        metavar="COL",
        help="Target column (y_7, y_14, y_30, …)",
    )
    parser.add_argument(
        "--imbalance-mode",
        type=str,
        choices=["none", "auto"],
        default="none",
        dest="imbalance_mode",
        help="none: no scale_pos_weight; auto: scale_pos_weight = neg/pos on final train rows",
    )
    parser.add_argument("--max_train_rows", type=int, default=5_000_000)
    parser.add_argument("--max_val_rows", type=int, default=2_000_000)
    parser.add_argument("--max_test_rows", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold_scan_step", type=float, default=0.01)
    parser.add_argument(
        "--feature-set",
        type=str,
        default="all",
        choices=list(FEATURE_SET_CHOICES),
        metavar="SET",
    )
    parser.add_argument("--log-feature-names", action="store_true")
    parser.add_argument("--train-neg-ratio", type=int, default=None, metavar="N")
    parser.add_argument("--n-estimators", type=int, default=300, dest="n_estimators")
    parser.add_argument("--max-depth", type=int, default=6, dest="max_depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, dest="learning_rate")
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8, dest="colsample_bytree")
    args = parser.parse_args()

    if args.train_neg_ratio is not None and args.train_neg_ratio < 1:
        print("Error: --train-neg-ratio must be >= 1 when provided", file=sys.stderr)
        sys.exit(2)

    imbalance_slug = "imnone" if args.imbalance_mode == "none" else "imauto"
    label_slug = args.label.replace(".", "_")
    run_suffix_parts: List[str] = []
    if args.feature_set != "all":
        run_suffix_parts.append(f"_fe_{args.feature_set}")
    if args.train_neg_ratio is not None:
        run_suffix_parts.append(f"_tnr{args.train_neg_ratio}")
    run_suffix_parts.append(f"_{imbalance_slug}")
    run_suffix = "".join(run_suffix_parts)

    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "configs" / "data_config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
    config = load_config(config_path)
    log_dir = base_dir / config["logging"]["log_dir"]
    logger = setup_logging(
        log_dir, config["logging"]["level"], args.experiment, label_slug, run_suffix=run_suffix
    )

    input_base_dir = base_dir / "data_engineered"

    try:
        results = train_xgb_baselines(
            args.experiment,
            input_base_dir,
            args.label,
            logger,
            max_train_rows=args.max_train_rows,
            max_val_rows=args.max_val_rows,
            max_test_rows=args.max_test_rows,
            seed=args.seed,
            threshold_scan_step=args.threshold_scan_step,
            imbalance_mode=args.imbalance_mode,
            feature_set=args.feature_set,
            log_feature_names=args.log_feature_names,
            train_neg_ratio=args.train_neg_ratio,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
        )
        write_reports(results, args.experiment, label_slug, imbalance_slug, logger)
        logger.info("SUCCESS: XGB engineered (feature_set=%s) complete", args.feature_set)
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
