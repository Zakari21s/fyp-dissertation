"""
Train and evaluate Random Forest baseline for disk failure prediction using y_14.

This script trains a Random Forest model on data_splits datasets and evaluates it on validation and test sets.
It supports both balanced (class_weight="balanced") and unbalanced modes.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

LABEL_COLUMN = "y_14"


@dataclass
class ModelMetrics:
    """Metrics for a single model on a single split."""
    split_name: str = ""
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
    val_metrics: ModelMetrics = field(default_factory=lambda: ModelMetrics(split_name='val'))
    test_metrics: ModelMetrics = field(default_factory=lambda: ModelMetrics(split_name='test'))
    best_threshold: Optional[float] = None


@dataclass
class BaselineResults:
    """Overall results for baseline training."""
    experiment_name: str = ""
    label_column: str = LABEL_COLUMN
    class_weight: str = "none"
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
    random_forest: ModelResults = field(default_factory=lambda: ModelResults(model_name='random_forest'))
    processing_time_seconds: float = 0.0


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path, log_level: str, experiment_name: str, balanced: bool) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    balanced_suffix = "_balanced" if balanced else ""
    log_file = log_dir / f"rf_baseline_{LABEL_COLUMN}{balanced_suffix}_{experiment_name}.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    root_logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(stream_handler)
    
    return logging.getLogger(__name__)


def find_partitions(input_dir: Path) -> List[Path]:
    """Find all partition directories (year=YYYY/month=MM) in input directory."""
    partitions = []
    for year_dir in sorted(input_dir.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            if month_dir.is_dir():
                partitions.append(month_dir)
    return partitions


def load_split_data_iterative(
    split_dir: Path,
    max_rows: Optional[int],
    seed: int,
    label_col: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load data from a split directory iteratively (memory-safe).
    
    Args:
        split_dir: Directory containing partitioned parquet files
        max_rows: Maximum number of rows to load (None = all)
        seed: Random seed for sampling
        label_col: Label column name (e.g., 'y_14')
        logger: Logger instance
        
    Returns:
        Combined dataframe
    """
    if not split_dir.exists():
        logger.warning(f"Split directory does not exist: {split_dir}")
        return pd.DataFrame()
    
    partitions = find_partitions(split_dir)
    logger.info(f"Found {len(partitions)} partitions in {split_dir}")
    
    all_dataframes: List[pd.DataFrame] = []
    total_rows = 0
    
    for partition_dir in partitions:
        parquet_files = sorted(partition_dir.glob("*.parquet"))
        
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                all_dataframes.append(df)
                total_rows += len(df)
                
                if max_rows is not None and total_rows >= max_rows:
                    logger.info(f"Reached max_rows limit ({max_rows:,}), stopping early")
                    break
                    
            except Exception as e:
                logger.error(f"Error reading {parquet_file}: {e}", exc_info=True)
                continue
        
        if max_rows is not None and total_rows >= max_rows:
            break
    
    if len(all_dataframes) == 0:
        return pd.DataFrame()
    
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    # If capped, keep all positives and downsample negatives.
    if max_rows is not None and len(df_combined) > max_rows:
        logger.info(f"Sampling from {len(df_combined):,} rows to {max_rows:,} rows")
        
        if label_col not in df_combined.columns:
            logger.warning(f"Column '{label_col}' not found, cannot perform smart sampling")
            df_combined = df_combined.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        else:
            positives = df_combined[df_combined[label_col] == 1].copy()
            negatives = df_combined[df_combined[label_col] == 0].copy()
            
            n_positives = len(positives)
            n_negatives_needed = max_rows - n_positives
            
            if n_negatives_needed < 0:
                logger.warning(f"More positives ({n_positives:,}) than max_rows ({max_rows:,}), keeping all positives")
                df_combined = positives.sample(n=max_rows, random_state=seed).reset_index(drop=True)
            else:
                if len(negatives) > n_negatives_needed:
                    negatives_sampled = negatives.sample(n=n_negatives_needed, random_state=seed).reset_index(drop=True)
                else:
                    negatives_sampled = negatives.copy()
                
                df_combined = pd.concat([positives, negatives_sampled], ignore_index=True)
                df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        logger.info(f"After sampling: {len(df_combined):,} rows")
    
    return df_combined


def build_xy(df: pd.DataFrame, label_col: str, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target vector y from dataframe.
    
    Args:
        df: Input dataframe
        label_col: Label column name (e.g., 'y_14')
        logger: Logger instance
        
    Returns:
        Tuple of (X, y) where X contains only n_* and r_* columns
    """
    feature_cols = [col for col in df.columns if col.startswith('n_') or col.startswith('r_')]
    
    if len(feature_cols) == 0:
        logger.warning("No feature columns found (n_* or r_*)")
        return pd.DataFrame(), pd.Series()
    
    X = df[feature_cols].copy()
    
    if label_col not in df.columns:
        logger.error(f"Column '{label_col}' not found")
        return pd.DataFrame(), pd.Series()
    
    y = df[label_col].copy()
    
    logger.info(f"Built X with {len(feature_cols)} features, y with {len(y)} samples")
    
    return X, y


def fit_preprocess(train_X: pd.DataFrame, logger: logging.Logger) -> Tuple[SimpleImputer, StandardScaler]:
    """
    Fit preprocessing transformers on training data.
    
    Args:
        train_X: Training feature matrix
        logger: Logger instance
        
    Returns:
        Tuple of (fitted imputer, fitted scaler)
    """
    logger.info("Fitting preprocessing transformers on training data...")
    
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train_X)
    logger.info(f"Fitted imputer: {train_X.isna().sum().sum()} missing values in training data")
    
    # Keep scaling for consistency with the LR baseline pipeline.
    scaler = StandardScaler()
    train_X_imputed = pd.DataFrame(
        imputer.transform(train_X),
        columns=train_X.columns,
        index=train_X.index
    )
    scaler.fit(train_X_imputed)
    logger.info("Fitted scaler")
    
    return imputer, scaler


def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    threshold: float = 0.5,
    logger: Optional[logging.Logger] = None
) -> ModelMetrics:
    """
    Evaluate a model on data.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        imputer: Fitted imputer
        scaler: Fitted scaler
        threshold: Classification threshold
        logger: Optional logger instance
        
    Returns:
        ModelMetrics object
    """
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index
    )
    X_scaled = pd.DataFrame(
        scaler.transform(X_imputed),
        columns=X.columns,
        index=X.index
    )
    
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = ModelMetrics()
    metrics.threshold = threshold
    
    try:
        metrics.pr_auc = average_precision_score(y, y_pred_proba)
    except:
        metrics.pr_auc = 0.0
    
    try:
        metrics.roc_auc = roc_auc_score(y, y_pred_proba)
    except:
        metrics.roc_auc = 0.0
    
    cm = confusion_matrix(y, y_pred)
    if cm.size == 4:
        metrics.tn, metrics.fp, metrics.fn, metrics.tp = cm.ravel()
    else:
        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 0:
                metrics.tn = int((y == 0).sum())
                metrics.fn = int((y == 1).sum())
            else:
                metrics.fp = int((y == 0).sum())
                metrics.tp = int((y == 1).sum())
    
    try:
        metrics.precision = precision_score(y, y_pred, zero_division=0)
        metrics.recall = recall_score(y, y_pred, zero_division=0)
        metrics.f1 = f1_score(y, y_pred, zero_division=0)
    except:
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
    logger: Optional[logging.Logger] = None
) -> Tuple[float, float]:
    """
    Find best threshold on validation set by maximizing F1.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        imputer: Fitted imputer
        scaler: Fitted scaler
        threshold_scan_step: Step size for threshold scanning
        logger: Optional logger instance
        
    Returns:
        Tuple of (best_threshold, best_f1)
    """
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index
    )
    X_scaled = pd.DataFrame(
        scaler.transform(X_imputed),
        columns=X.columns,
        index=X.index
    )
    
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    thresholds = np.arange(0.01, 1.0, threshold_scan_step)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        try:
            f1 = f1_score(y, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        except:
            continue
    
    if logger:
        logger.info(f"Best threshold on validation: {best_threshold:.3f} (F1={best_f1:.4f})")
    
    return best_threshold, best_f1


def train_baselines(
    experiment_name: str,
    input_base_dir: Path,
    logger: logging.Logger,
    balanced: bool = False,
    max_train_rows: Optional[int] = 5_000_000,
    max_val_rows: Optional[int] = 2_000_000,
    max_test_rows: Optional[int] = 2_000_000,
    seed: int = 42,
    threshold_scan_step: float = 0.01
) -> BaselineResults:
    """
    Train and evaluate Random Forest baseline models.
    
    Args:
        experiment_name: Name of the experiment
        input_base_dir: Base input directory (data_splits/)
        logger: Logger instance
        balanced: Whether to use class_weight="balanced"
        max_train_rows: Maximum rows to load from training set
        max_val_rows: Maximum rows to load from validation set
        max_test_rows: Maximum rows to load from test set
        seed: Random seed for reproducibility
        threshold_scan_step: Step size for threshold scanning
        
    Returns:
        BaselineResults object
    """
    start_time = datetime.now()
    class_weight_str = "balanced" if balanced else "none"
    results = BaselineResults(
        experiment_name=experiment_name,
        label_column=LABEL_COLUMN,
        class_weight=class_weight_str
    )
    
    logger.info("=" * 60)
    logger.info("TRAINING RANDOM FOREST BASELINE")
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Label column: {LABEL_COLUMN}")
    logger.info(f"Class weight: {class_weight_str}")
    logger.info(f"Random seed: {seed}")
    logger.info("")
    
    input_experiment_dir = input_base_dir / experiment_name
    
    if not input_experiment_dir.exists():
        raise FileNotFoundError(f"Input experiment directory not found: {input_experiment_dir}")
    
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info("Loading training data...")
    train_dir = input_experiment_dir / 'train'
    train_df = load_split_data_iterative(train_dir, max_train_rows, seed, LABEL_COLUMN, logger)
    
    if len(train_df) == 0:
        raise ValueError("No training data loaded")
    
    train_X, train_y = build_xy(train_df, LABEL_COLUMN, logger)
    
    if len(train_X) == 0:
        raise ValueError("No features found in training data")
    
    results.train_rows = len(train_df)
    results.train_positives = int((train_y == 1).sum())
    results.train_negatives = int((train_y == 0).sum())
    results.feature_count = len(train_X.columns)
    
    missing_count = train_X.isna().sum().sum()
    total_cells = len(train_X) * len(train_X.columns)
    results.missingness_before_imputation = (missing_count / total_cells * 100) if total_cells > 0 else 0.0
    
    logger.info(f"Training data: {results.train_rows:,} rows ({results.train_positives:,} positive, {results.train_negatives:,} negative)")
    logger.info(f"Features: {results.feature_count}")
    logger.info(f"Missingness: {results.missingness_before_imputation:.2f}%")
    logger.info("")
    
    imputer, scaler = fit_preprocess(train_X, logger)
    
    train_X_imputed = pd.DataFrame(
        imputer.transform(train_X),
        columns=train_X.columns,
        index=train_X.index
    )
    train_X_scaled = pd.DataFrame(
        scaler.transform(train_X_imputed),
        columns=train_X.columns,
        index=train_X.index
    )
    
    logger.info("Training Random Forest...")
    rf_params = {
        'n_estimators': 200,
        'random_state': seed,
        'n_jobs': -1,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    
    if balanced:
        rf_params['class_weight'] = 'balanced'
        logger.info("Using class_weight='balanced'")
    
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(train_X_scaled, train_y)
    logger.info("Random Forest trained")
    logger.info("")
    
    logger.info("Loading validation data...")
    val_dir = input_experiment_dir / 'val'
    val_df = load_split_data_iterative(val_dir, max_val_rows, seed, LABEL_COLUMN, logger)
    
    if len(val_df) == 0:
        logger.warning("No validation data loaded")
    else:
        val_X, val_y = build_xy(val_df, LABEL_COLUMN, logger)
        results.val_rows = len(val_df)
        results.val_positives = int((val_y == 1).sum())
        results.val_negatives = int((val_y == 0).sum())
        
        logger.info(f"Validation data: {results.val_rows:,} rows ({results.val_positives:,} positive, {results.val_negatives:,} negative)")
        logger.info("")
        
        logger.info("Finding best threshold on validation...")
        best_threshold, best_f1 = find_best_threshold(
            rf_model,
            val_X, val_y,
            imputer, scaler,
            threshold_scan_step,
            logger
        )
        results.random_forest.best_threshold = best_threshold
        
        logger.info(f"Evaluating Random Forest on validation with threshold={best_threshold:.3f}...")
        results.random_forest.val_metrics = evaluate(
            rf_model,
            val_X, val_y,
            imputer, scaler,
            threshold=best_threshold,
            logger=logger
        )
        results.random_forest.val_metrics.split_name = 'val'
        results.random_forest.val_metrics.threshold = best_threshold
    
    logger.info("Loading test data...")
    test_dir = input_experiment_dir / 'test'
    test_df = load_split_data_iterative(test_dir, max_test_rows, seed, LABEL_COLUMN, logger)
    
    if len(test_df) == 0:
        logger.warning("No test data loaded")
    else:
        test_X, test_y = build_xy(test_df, LABEL_COLUMN, logger)
        results.test_rows = len(test_df)
        results.test_positives = int((test_y == 1).sum())
        results.test_negatives = int((test_y == 0).sum())
        
        logger.info(f"Test data: {results.test_rows:,} rows ({results.test_positives:,} positive, {results.test_negatives:,} negative)")
        logger.info("")
        
        # Reuse the threshold tuned on validation.
        test_threshold = results.random_forest.best_threshold if results.random_forest.best_threshold is not None else 0.5
        logger.info(f"Evaluating Random Forest on test (threshold={test_threshold:.3f})...")
        results.random_forest.test_metrics = evaluate(
            rf_model,
            test_X, test_y,
            imputer, scaler,
            threshold=test_threshold,
            logger=logger
        )
        results.random_forest.test_metrics.split_name = 'test'
        results.random_forest.test_metrics.threshold = test_threshold
    
    results.processing_time_seconds = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processing time: {results.processing_time_seconds:.2f}s")
    
    return results


def write_reports(results: BaselineResults, logger: logging.Logger) -> None:
    """Write JSON and Markdown reports for baseline training."""
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = results.experiment_name
    label_col = results.label_column
    balanced_suffix = "_balanced" if results.class_weight == "balanced" else ""
    
    json_path = reports_dir / f"baseline_rf_{label_col}{balanced_suffix}_results_{experiment_name}.json"
    
    report_json = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'label_column': results.label_column,
        'class_weight': results.class_weight,
        'data_summary': {
            'train': {
                'rows': results.train_rows,
                'positives': results.train_positives,
                'negatives': results.train_negatives
            },
            'val': {
                'rows': results.val_rows,
                'positives': results.val_positives,
                'negatives': results.val_negatives
            },
            'test': {
                'rows': results.test_rows,
                'positives': results.test_positives,
                'negatives': results.test_negatives
            }
        },
        'feature_info': {
            'feature_count': results.feature_count,
            'missingness_before_imputation_pct': results.missingness_before_imputation
        },
        'models': {
            'random_forest': {
                'class_weight': results.class_weight,
                'best_threshold': results.random_forest.best_threshold,
                'val': asdict(results.random_forest.val_metrics),
                'test': asdict(results.random_forest.test_metrics)
            }
        },
        'processing_time_seconds': results.processing_time_seconds
    }
    
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2, default=str)
    logger.info(f"JSON report written to {json_path}")
    
    md_path = reports_dir / f"baseline_rf_{label_col}{balanced_suffix}_results_{experiment_name}.md"
    with open(md_path, 'w') as f:
        f.write(f"# Baseline Model Results (Random Forest) - {experiment_name}\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Experiment Name:** `{experiment_name}`\n")
        f.write(f"- **Target Label:** `{results.label_column}`\n")
        f.write(f"- **Class Weight:** `{results.class_weight}`\n")
        f.write(f"- **Processing Time:** {results.processing_time_seconds:.2f}s\n\n")
        
        f.write("## Data Summary\n\n")
        f.write("| Split | Rows | Positives | Negatives |\n")
        f.write("|-------|------|-----------|----------|\n")
        f.write(f"| Train | {results.train_rows:,} | {results.train_positives:,} | {results.train_negatives:,} |\n")
        f.write(f"| Val | {results.val_rows:,} | {results.val_positives:,} | {results.val_negatives:,} |\n")
        f.write(f"| Test | {results.test_rows:,} | {results.test_positives:,} | {results.test_negatives:,} |\n")
        f.write("\n")
        
        f.write("## Feature Information\n\n")
        f.write(f"- **Feature Count:** {results.feature_count}\n")
        f.write(f"- **Missingness (before imputation):** {results.missingness_before_imputation:.2f}%\n")
        f.write("\n")
        
        f.write("## Model Results\n\n")
        f.write("### Random Forest\n\n")
        f.write(f"- **Class Weight:** `{results.class_weight}`\n")
        if results.random_forest.best_threshold is not None:
            f.write(f"- **Best Threshold (from VAL):** {results.random_forest.best_threshold:.3f}\n\n")
        
        f.write("| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 |\n")
        f.write("|-------|-----------|--------|---------|-----------|--------|-----|\n")
        
        val_metrics = results.random_forest.val_metrics
        f.write(f"| VAL | {val_metrics.threshold:.3f} | {val_metrics.pr_auc:.4f} | {val_metrics.roc_auc:.4f} | {val_metrics.precision:.4f} | {val_metrics.recall:.4f} | {val_metrics.f1:.4f} |\n")
        
        test_metrics = results.random_forest.test_metrics
        f.write(f"| TEST | {test_metrics.threshold:.3f} | {test_metrics.pr_auc:.4f} | {test_metrics.roc_auc:.4f} | {test_metrics.precision:.4f} | {test_metrics.recall:.4f} | {test_metrics.f1:.4f} |\n")
        f.write("\n")
    
    logger.info(f"Markdown report written to {md_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate Random Forest baseline for disk failure prediction using y_14'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Experiment name (e.g., exp_time_generalisation or exp_time_generalisation__train_eval)'
    )
    parser.add_argument(
        '--balanced',
        action='store_true',
        help='Use class_weight="balanced" for Random Forest'
    )
    parser.add_argument(
        '--max_train_rows',
        type=int,
        default=5_000_000,
        help='Maximum rows to load from training set (default: 5,000,000)'
    )
    parser.add_argument(
        '--max_val_rows',
        type=int,
        default=2_000_000,
        help='Maximum rows to load from validation set (default: 2,000,000)'
    )
    parser.add_argument(
        '--max_test_rows',
        type=int,
        default=2_000_000,
        help='Maximum rows to load from test set (default: 2,000,000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--threshold_scan_step',
        type=float,
        default=0.01,
        help='Step size for threshold scanning (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / 'configs' / 'data_config.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    log_dir = Path(config['logging']['log_dir'])
    logger = setup_logging(log_dir, config['logging']['level'], args.experiment, args.balanced)
    
    base_dir = Path(__file__).parent.parent
    
    input_base_dir = base_dir / 'data_splits'
    
    try:
        results = train_baselines(
            args.experiment,
            input_base_dir,
            logger,
            args.balanced,
            args.max_train_rows,
            args.max_val_rows,
            args.max_test_rows,
            args.seed,
            args.threshold_scan_step
        )
        
        write_reports(results, logger)
        
        logger.info("=" * 60)
        logger.info("SUCCESS: Random Forest baseline training complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during baseline training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

