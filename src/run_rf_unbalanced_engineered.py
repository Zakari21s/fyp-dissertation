"""
Random Forest on data_engineered — mirrors ``run_lr_unbalanced_engineered`` workflow.

Same loaders, feature-set policy, row caps, train negative downsampling, imputer+scaler,
threshold scan on validation (max F1), then test. Reports/logs use ``rf_unbalanced_engineered_`` prefix.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
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
class RFExperimentResults:
    """Parallel to LR ``BaselineResults`` but for Random Forest."""
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
    random_forest: ModelResults = field(default_factory=lambda: ModelResults(model_name="random_forest"))
    processing_time_seconds: float = 0.0
    rf_class_weight: Optional[str] = None  # None or "balanced"
    feature_set: str = "all"
    feature_names: List[str] = field(default_factory=list)
    train_neg_ratio: Optional[int] = None
    train_pool_rows: Optional[int] = None
    train_pool_positives: Optional[int] = None
    train_pool_negatives: Optional[int] = None
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    n_jobs: int = -1


def setup_logging(
    log_dir: Path,
    log_level: str,
    experiment_name: str,
    label_slug: str,
    run_suffix: str = "",
) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"rf_unbalanced_engineered_{label_slug}_{experiment_name}{run_suffix}.log"
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
    train_neg_ratio: Optional[int] = None,
    class_weight_slug: str = "cwnone",
) -> str:
    parts: List[str] = []
    if feature_set != "all":
        parts.append(f"_fe_{feature_set}")
    if train_neg_ratio is not None:
        parts.append(f"_tnr{train_neg_ratio}")
    parts.append(f"_{class_weight_slug}")
    return f"rf_unbalanced_engineered_{label_slug}_{experiment_name}{''.join(parts)}"


def train_rf_baselines(
    experiment_name: str,
    input_base_dir: Path,
    label_col: str,
    logger: logging.Logger,
    max_train_rows: Optional[int] = 5_000_000,
    max_val_rows: Optional[int] = 2_000_000,
    max_test_rows: Optional[int] = 2_000_000,
    seed: int = 42,
    threshold_scan_step: float = 0.01,
    class_weight: Optional[str] = None,
    feature_set: str = "all",
    log_feature_names: bool = False,
    train_neg_ratio: Optional[int] = None,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    n_jobs: int = -1,
) -> RFExperimentResults:
    start = datetime.now()
    results = RFExperimentResults(
        experiment_name=experiment_name,
        label_column=label_col,
        rf_class_weight=class_weight,
        feature_set=feature_set,
        train_neg_ratio=train_neg_ratio,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
    )
    logger.info("=" * 60)
    logger.info(
        "Random Forest on engineered data — feature_set=%s; class_weight=%s; train_neg_ratio=%s; "
        "n_estimators=%s max_depth=%s min_samples_leaf=%s n_jobs=%s",
        feature_set,
        repr(class_weight),
        repr(train_neg_ratio),
        n_estimators,
        repr(max_depth),
        min_samples_leaf,
        n_jobs,
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

    logger.info(
        "Fitting RandomForestClassifier (same imputer+scaler as LR pipeline for comparability)..."
    )
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=seed,
        class_weight=class_weight,
    )
    logger.info(
        "Fitting RF on %s × %d matrix (can take a long time on large train)",
        f"{len(train_X_scaled):,}",
        train_X_scaled.shape[1],
    )
    rf_model.fit(train_X_scaled, train_y)
    logger.info("Random Forest trained")

    logger.info("Loading validation data (max_rows=%s)...", max_val_rows)
    val_dir = input_experiment_dir / "val"
    val_df = load_split_streaming(val_dir, max_val_rows, seed, label_col, logger)
    if len(val_df) > 0:
        val_X, val_y = build_xy(val_df, label_col, logger, feature_set=feature_set, log_feature_names=False)
        results.val_rows = len(val_df)
        results.val_positives = int((val_y == 1).sum())
        results.val_negatives = int((val_y == 0).sum())
        logger.info("Val: %s rows (%s pos, %s neg)", f"{results.val_rows:,}", results.val_positives, results.val_negatives)

        results.always_negative.val_metrics = evaluate(
            "always_negative", val_X, val_y, imputer, scaler, threshold=0.5, logger=logger
        )
        results.always_negative.val_metrics.split_name = "val"

        results.random_forest.val_metrics = evaluate(
            rf_model, val_X, val_y, imputer, scaler, threshold=0.5, logger=logger
        )
        results.random_forest.val_metrics.split_name = "val"

        best_threshold, _ = find_best_threshold(
            rf_model, val_X, val_y, imputer, scaler, threshold_scan_step, logger
        )
        results.random_forest.best_threshold = best_threshold
        results.random_forest.val_metrics = evaluate(
            rf_model, val_X, val_y, imputer, scaler, threshold=best_threshold, logger=logger
        )
        results.random_forest.val_metrics.split_name = "val"
        results.random_forest.val_metrics.threshold = best_threshold

    logger.info("Loading test data (max_rows=%s)...", max_test_rows)
    test_dir = input_experiment_dir / "test"
    test_df = load_split_streaming(test_dir, max_test_rows, seed, label_col, logger)
    if len(test_df) > 0:
        test_X, test_y = build_xy(test_df, label_col, logger, feature_set=feature_set, log_feature_names=False)
        results.test_rows = len(test_df)
        results.test_positives = int((test_y == 1).sum())
        results.test_negatives = int((test_y == 0).sum())
        logger.info("Test: %s rows (%s pos, %s neg)", f"{results.test_rows:,}", results.test_positives, results.test_negatives)

        results.always_negative.test_metrics = evaluate(
            "always_negative", test_X, test_y, imputer, scaler, threshold=0.5, logger=logger
        )
        results.always_negative.test_metrics.split_name = "test"

        test_threshold = results.random_forest.best_threshold if results.random_forest.best_threshold is not None else 0.5
        logger.info("Evaluating RF on test with threshold %.3f", test_threshold)
        results.random_forest.test_metrics = evaluate(
            rf_model, test_X, test_y, imputer, scaler, threshold=test_threshold, logger=logger
        )
        results.random_forest.test_metrics.split_name = "test"
        results.random_forest.test_metrics.threshold = test_threshold

    results.processing_time_seconds = (datetime.now() - start).total_seconds()
    logger.info("Processing time: %.2fs", results.processing_time_seconds)
    return results


def write_reports(
    results: RFExperimentResults,
    experiment_name: str,
    label_slug: str,
    logger: logging.Logger,
    class_weight_slug: str = "cwnone",
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = _report_file_stem(
        label_slug,
        experiment_name,
        results.feature_set,
        results.train_neg_ratio,
        class_weight_slug=class_weight_slug,
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
            "Train rows after downsampling are in data_summary.train (used for RF fit). "
            "Val/test unchanged."
        )

    cw_json: Optional[str] = results.rf_class_weight
    report_json = {
        "experiment_name": results.experiment_name,
        "timestamp": datetime.now().isoformat(),
        "model": "random_forest",
        "label_column": results.label_column,
        "data_source": "data_engineered",
        "class_weight": cw_json,
        "random_forest_hyperparameters": {
            "n_estimators": results.n_estimators,
            "max_depth": results.max_depth,
            "min_samples_leaf": results.min_samples_leaf,
            "n_jobs": results.n_jobs,
        },
        "feature_set": results.feature_set,
        "feature_policy": "column_matches_feature_set (shared with LR runner)",
        "feature_names": results.feature_names,
        "row_cap_policy": "keep_all_positives_sample_negatives_shuffle",
        "preprocessing": "SimpleImputer(median) + StandardScaler (same as LR engineered runner)",
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
            "random_forest": {
                "class_weight": cw_json,
                "best_threshold": results.random_forest.best_threshold,
                "val": asdict(results.random_forest.val_metrics),
                "test": asdict(results.random_forest.test_metrics),
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

    cw = results.rf_class_weight or "none"
    md = [
        f"# Random Forest (engineered data) – {experiment_name}",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"- **Label:** `{results.label_column}`",
        f"- **Data source:** data_engineered",
        f"- **Feature set mode:** `{results.feature_set}` (same policy as LR: `column_matches_feature_set`)",
        f"- **Class weight:** `{cw}`",
        f"- **RF:** n_estimators={results.n_estimators}, max_depth={results.max_depth}, "
        f"min_samples_leaf={results.min_samples_leaf}, n_jobs={results.n_jobs}",
        "- **Preprocessing:** median imputation + StandardScaler (same as LR engineered runner for fair comparison).",
        "- **Row cap policy:** same as LR (hive-order pool, then keep positives + sample negatives to cap, shuffle).",
        "",
        "## Train negative downsampling",
        "",
    ]
    if results.train_neg_ratio is not None and results.train_pool_rows is not None:
        md.extend(
            [
                "- **Status:** enabled",
                f"- **Requested ratio:** 1:{results.train_neg_ratio} (negatives per positive)",
                f"- **Original train pool (after row cap, before neg downsampling):** {results.train_pool_rows:,} rows — {results.train_pool_positives:,} pos / {results.train_pool_negatives:,} neg",
                f"- **Sampled train (used for RF fit):** {results.train_rows:,} rows — {results.train_positives:,} pos / {results.train_negatives:,} neg",
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
            "## Random Forest",
            "",
            f"**Best threshold (from val, same scan as LR):** {results.random_forest.best_threshold}",
            "",
            "**Val:**",
            _metrics_line(results.random_forest.val_metrics, "  "),
            "",
            "**Test:**",
            _metrics_line(results.random_forest.test_metrics, "  "),
            "",
            f"**Processing time:** {results.processing_time_seconds:.2f}s",
            "",
            "## Feature names (reproducibility)",
            "",
            "Sorted column names used in X:",
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
        description="Train Random Forest on data_engineered (same workflow as run_lr_unbalanced_engineered).",
    )
    parser.add_argument("--experiment", type=str, default="exp_time_generalisation")
    parser.add_argument(
        "--label",
        type=str,
        default="y_30",
        metavar="COL",
        help="Target column (y_7, y_14, y_30, …). Default: y_30",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        choices=["none", "balanced"],
        default="none",
        help="RandomForest class_weight: none → None, balanced → 'balanced'. Default: none",
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
    parser.add_argument("--n-estimators", type=int, default=200, dest="n_estimators")
    parser.add_argument("--max-depth", type=int, default=None, dest="max_depth")
    parser.add_argument("--min-samples-leaf", type=int, default=1, dest="min_samples_leaf")
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    args = parser.parse_args()

    if args.train_neg_ratio is not None and args.train_neg_ratio < 1:
        print("Error: --train-neg-ratio must be >= 1 when provided", file=sys.stderr)
        sys.exit(2)

    cw: Optional[str] = None if args.class_weight == "none" else "balanced"

    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "configs" / "data_config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
    config = load_config(config_path)
    log_dir = base_dir / config["logging"]["log_dir"]
    label_slug = args.label.replace(".", "_")
    cw_slug = "cwnone" if args.class_weight == "none" else "cwbalanced"
    run_suffix_parts: List[str] = []
    if args.feature_set != "all":
        run_suffix_parts.append(f"_fe_{args.feature_set}")
    if args.train_neg_ratio is not None:
        run_suffix_parts.append(f"_tnr{args.train_neg_ratio}")
    run_suffix_parts.append(f"_{cw_slug}")
    run_suffix = "".join(run_suffix_parts)
    logger = setup_logging(
        log_dir, config["logging"]["level"], args.experiment, label_slug, run_suffix=run_suffix
    )

    input_base_dir = base_dir / "data_engineered"

    try:
        results = train_rf_baselines(
            args.experiment,
            input_base_dir,
            args.label,
            logger,
            max_train_rows=args.max_train_rows,
            max_val_rows=args.max_val_rows,
            max_test_rows=args.max_test_rows,
            seed=args.seed,
            threshold_scan_step=args.threshold_scan_step,
            class_weight=cw,
            feature_set=args.feature_set,
            log_feature_names=args.log_feature_names,
            train_neg_ratio=args.train_neg_ratio,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=args.n_jobs,
        )
        write_reports(results, args.experiment, label_slug, logger, class_weight_slug=cw_slug)
        logger.info("SUCCESS: RF engineered (feature_set=%s) complete", args.feature_set)
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
