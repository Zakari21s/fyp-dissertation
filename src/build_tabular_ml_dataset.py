"""
Build ML-ready tabular datasets from split datasets.

This script processes split datasets (train/val/test) and creates ML-ready tabular
datasets with y_30 as the target variable. It reads from data_splits/ and writes
to data_ml/ without modifying source data.

Output structure:
- data_ml/<experiment_name>/tabular_y30/<split>/year=YYYY/month=MM/data.parquet
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa


@dataclass
class SplitMLStats:
    """Statistics for a single split in ML dataset."""
    split_name: str = ""
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped_missing_y30: int = 0
    partitions_processed: int = 0
    files_processed: int = 0
    files_failed: int = 0
    label_distribution: Dict[str, int] = field(default_factory=lambda: {'positive': 0, 'negative': 0})
    unique_entities: int = 0
    date_range_min: Optional[str] = None
    date_range_max: Optional[str] = None
    missingness_n_features: float = 0.0
    missingness_r_features: float = 0.0
    n_missing_total: int = 0
    n_cells_total: int = 0
    r_missing_total: int = 0
    r_cells_total: int = 0


@dataclass
class MLDatasetStats:
    """Overall statistics for ML dataset building."""
    experiment_name: str = ""
    train_stats: SplitMLStats = field(default_factory=lambda: SplitMLStats(split_name='train'))
    val_stats: SplitMLStats = field(default_factory=lambda: SplitMLStats(split_name='val'))
    test_stats: SplitMLStats = field(default_factory=lambda: SplitMLStats(split_name='test'))
    processing_time_seconds: float = 0.0


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path, log_level: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"ml_tabular_y30_{experiment_name}.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    
    # Add stream handler
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


def iter_partitions(input_dir: Path, logger: logging.Logger, max_partitions: Optional[int] = None):
    """
    Generator that yields (partition_path, partition_name) tuples.
    
    Args:
        input_dir: Base directory containing partitioned data
        logger: Logger instance
        max_partitions: Maximum number of partitions to process (None = all)
        
    Yields:
        Tuple of (partition_path, partition_name) where partition_name is "year=YYYY/month=MM"
    """
    partitions = find_partitions(input_dir)
    logger.info(f"Found {len(partitions)} partitions in {input_dir}")
    
    if max_partitions is not None and max_partitions > 0:
        partitions = partitions[:max_partitions]
        logger.info(f"Limiting to first {len(partitions)} partitions (--max_partitions={max_partitions})")
    
    for partition_dir in partitions:
        # Extract partition name from path
        year_part = partition_dir.parent.name
        month_part = partition_dir.name
        partition_name = f"{year_part}/{month_part}"
        
        yield partition_dir, partition_name


def iter_parquet_files(partition_dir: Path, logger: logging.Logger, max_files: Optional[int] = None):
    """
    Generator that yields parquet files in a partition.
    
    Args:
        partition_dir: Partition directory
        logger: Logger instance
        max_files: Maximum number of files to process (None = all)
        
    Yields:
        Path to parquet file
    """
    parquet_files = sorted(partition_dir.glob("*.parquet"))
    
    if max_files is not None and max_files > 0:
        parquet_files = parquet_files[:max_files]
        logger.debug(f"Limiting to first {len(parquet_files)} files (--max_files={max_files})")
    
    for parquet_file in parquet_files:
        yield parquet_file


def select_columns(df: pd.DataFrame, keep_extra: bool = False) -> pd.DataFrame:
    """
    Select columns for ML dataset.
    
    Always keeps: disk_id, model, smart_day, y_30
    Keeps: n_* and r_* feature columns
    Optionally keeps: other columns if keep_extra=True
    
    Args:
        df: Input dataframe
        keep_extra: Whether to keep extra columns beyond required ones
        
    Returns:
        Dataframe with selected columns
    """
    required_cols = ['disk_id', 'model', 'smart_day', 'y_30']
    feature_prefixes = ['n_', 'r_']
    
    # Find columns to keep
    cols_to_keep = []
    
    # Required columns
    for col in required_cols:
        if col in df.columns:
            cols_to_keep.append(col)
        else:
            # Log warning but continue
            pass
    
    # Feature columns (n_* and r_*)
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in feature_prefixes):
            cols_to_keep.append(col)
    
    # Extra columns if requested
    if keep_extra:
        for col in df.columns:
            if col not in cols_to_keep:
                cols_to_keep.append(col)
    
    # Select columns
    if cols_to_keep:
        return df[cols_to_keep].copy()
    else:
        return pd.DataFrame()


def compute_stats_update(
    df: pd.DataFrame,
    stats: SplitMLStats,
    partition_name: str,
    logger: logging.Logger
) -> None:
    """
    Update statistics from a processed dataframe.
    
    Args:
        df: Processed dataframe
        stats: SplitMLStats object to update
        partition_name: Partition name for logging
        logger: Logger instance
    """
    if len(df) == 0:
        return
    
    # Label distribution
    if 'y_30' in df.columns:
        positive = int((df['y_30'] == 1).sum())
        negative = int((df['y_30'] == 0).sum())
        stats.label_distribution['positive'] += positive
        stats.label_distribution['negative'] += negative
    
    # Unique entities
    if 'disk_id' in df.columns and 'model' in df.columns:
        valid_mask = df['disk_id'].notna() & df['model'].notna()
        valid_df = df[valid_mask]
        if len(valid_df) > 0:
            # Final unique counts are computed after all partitions are processed.
            entities = set(zip(
                valid_df['disk_id'].astype(str),
                valid_df['model'].astype(str)
            ))
            
    
    # Date range
    if 'smart_day' in df.columns:
        valid_dates = df['smart_day'].dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            
            if stats.date_range_min is None:
                stats.date_range_min = str(min_date)
                stats.date_range_max = str(max_date)
            else:
                # Update range
                current_min = pd.to_datetime(stats.date_range_min)
                current_max = pd.to_datetime(stats.date_range_max)
                if min_date < current_min:
                    stats.date_range_min = str(min_date)
                if max_date > current_max:
                    stats.date_range_max = str(max_date)
    
    # Missingness for feature groups (accumulate totals)
    n_cols = [col for col in df.columns if col.startswith('n_')]
    r_cols = [col for col in df.columns if col.startswith('r_')]
    
    if n_cols:
        n_missing = int(df[n_cols].isna().sum().sum())
        n_cells = len(df) * len(n_cols)
        stats.n_missing_total += n_missing
        stats.n_cells_total += n_cells
        # Update percentage
        stats.missingness_n_features = (stats.n_missing_total / stats.n_cells_total * 100) if stats.n_cells_total > 0 else 0.0
    
    if r_cols:
        r_missing = int(df[r_cols].isna().sum().sum())
        r_cells = len(df) * len(r_cols)
        stats.r_missing_total += r_missing
        stats.r_cells_total += r_cells
        # Update percentage
        stats.missingness_r_features = (stats.r_missing_total / stats.r_cells_total * 100) if stats.r_cells_total > 0 else 0.0


def write_partition_output(
    all_dataframes: List[pd.DataFrame],
    output_partition_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Write combined dataframes to output partition directory.
    
    Args:
        all_dataframes: List of dataframes to combine and write
        output_partition_dir: Output partition directory
        logger: Logger instance
    """
    if len(all_dataframes) == 0:
        return
    
    # Create output directory
    output_partition_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine dataframes
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    # Write as single parquet file
    output_file = output_partition_dir / "data.parquet"
    df_combined.to_parquet(output_file, engine='pyarrow', index=False)
    
    logger.debug(f"  Wrote {len(df_combined):,} rows to {output_file}")


def process_split(
    split_name: str,
    input_dir: Path,
    output_base_dir: Path,
    stats: SplitMLStats,
    logger: logging.Logger,
    max_partitions: Optional[int] = None,
    max_files: Optional[int] = None,
    keep_extra: bool = False
) -> None:
    """
    Process a single split (train/val/test).
    
    Args:
        split_name: Name of split (train/val/test)
        input_dir: Input directory for this split
        output_base_dir: Base output directory
        stats: SplitMLStats object to update
        logger: Logger instance
        max_partitions: Maximum partitions to process
        max_files: Maximum files per partition to process
        keep_extra: Whether to keep extra columns
    """
    logger.info(f"Processing {split_name.upper()} split...")
    
    if not input_dir.exists():
        logger.warning(f"Input directory does not exist: {input_dir}")
        return
    
    output_split_dir = output_base_dir / split_name
    
    partition_dataframes: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    
    for partition_dir, partition_name in iter_partitions(input_dir, logger, max_partitions):
        stats.partitions_processed += 1
        logger.info(f"  Processing partition: {partition_name}")
        
        partition_rows_in = 0
        partition_rows_out = 0
        
        for parquet_file in iter_parquet_files(partition_dir, logger, max_files):
            stats.files_processed += 1
            
            try:
                # Read parquet file
                df = pd.read_parquet(parquet_file)
                partition_rows_in += len(df)
                stats.rows_in += len(df)
                
                # Select columns
                df_selected = select_columns(df, keep_extra)
                
                if len(df_selected) == 0:
                    logger.warning(f"    No columns selected from {parquet_file.name}")
                    continue
                
                # Ensure smart_day is datetime
                if 'smart_day' in df_selected.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df_selected['smart_day']):
                        df_selected['smart_day'] = pd.to_datetime(df_selected['smart_day'], errors='coerce')
                
                # Ensure disk_id is consistent type (prefer int, but keep as-is if needed)
                if 'disk_id' in df_selected.columns:
                    # Keep IDs stable even when some partitions store them as strings.
                    try:
                        df_selected['disk_id'] = df_selected['disk_id'].astype('Int64')
                    except:
                        # Keep as string if conversion fails
                        df_selected['disk_id'] = df_selected['disk_id'].astype(str)
                
                # Drop rows with missing y_30
                if 'y_30' not in df_selected.columns:
                    logger.warning(f"    Column 'y_30' not found in {parquet_file.name}, skipping")
                    stats.files_failed += 1
                    continue
                
                rows_before = len(df_selected)
                df_selected = df_selected[df_selected['y_30'].notna()].copy()
                rows_dropped = rows_before - len(df_selected)
                stats.rows_dropped_missing_y30 += rows_dropped
                
                # Ensure y_30 is binary {0, 1}
                if len(df_selected) > 0:
                    # Check for invalid values
                    invalid_mask = ~df_selected['y_30'].isin([0, 1])
                    if invalid_mask.sum() > 0:
                        logger.warning(f"    Found {invalid_mask.sum()} rows with invalid y_30 values, dropping")
                        df_selected = df_selected[~invalid_mask].copy()
                
                if len(df_selected) == 0:
                    continue
                
                partition_rows_out += len(df_selected)
                stats.rows_out += len(df_selected)
                
                # Store for writing
                partition_dataframes[partition_name].append(df_selected)
                
                # Update stats incrementally
                compute_stats_update(df_selected, stats, partition_name, logger)
                
            except Exception as e:
                logger.error(f"    Error processing {parquet_file}: {e}", exc_info=True)
                stats.files_failed += 1
                continue
        
        # Write partition output
        if partition_name in partition_dataframes:
            output_partition_dir = output_split_dir / partition_name
            write_partition_output(partition_dataframes[partition_name], output_partition_dir, logger)
            logger.info(f"    {partition_name}: {partition_rows_in:,} -> {partition_rows_out:,} rows")
    
    logger.info(f"  {split_name.upper()} complete: {stats.rows_out:,} rows from {stats.partitions_processed} partitions")


def compute_final_entity_count(output_split_dir: Path, logger: logging.Logger) -> int:
    """
    Compute final unique entity count by reading output files.
    
    Args:
        output_split_dir: Output directory for split
        logger: Logger instance
        
    Returns:
        Number of unique entities
    """
    entities: Set[Tuple[str, str]] = set()
    
    if not output_split_dir.exists():
        return 0
    
    for partition_dir, partition_name in iter_partitions(output_split_dir, logger, None):
        for parquet_file in iter_parquet_files(partition_dir, logger, None):
            try:
                df = pd.read_parquet(parquet_file)
                if 'disk_id' in df.columns and 'model' in df.columns:
                    valid_mask = df['disk_id'].notna() & df['model'].notna()
                    valid_df = df[valid_mask]
                    if len(valid_df) > 0:
                        entity_tuples = set(zip(
                            valid_df['disk_id'].astype(str),
                            valid_df['model'].astype(str)
                        ))
                        entities.update(entity_tuples)
            except Exception as e:
                logger.warning(f"Error reading {parquet_file} for entity count: {e}")
                continue
    
    return len(entities)


def build_ml_dataset(
    experiment_name: str,
    input_base_dir: Path,
    output_base_dir: Path,
    logger: logging.Logger,
    max_partitions: Optional[int] = None,
    max_files: Optional[int] = None,
    keep_extra: bool = False
) -> MLDatasetStats:
    """
    Build ML-ready tabular dataset from split datasets.
    
    Args:
        experiment_name: Name of the experiment
        input_base_dir: Base input directory (data_splits/)
        output_base_dir: Base output directory (data_ml/)
        logger: Logger instance
        max_partitions: Maximum partitions to process per split
        max_files: Maximum files per partition to process
        keep_extra: Whether to keep extra columns
        
    Returns:
        MLDatasetStats object with statistics
    """
    start_time = datetime.now()
    stats = MLDatasetStats(experiment_name=experiment_name)
    
    logger.info("=" * 60)
    logger.info("BUILDING ML-READY TABULAR DATASET")
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Input: {input_base_dir / experiment_name}")
    logger.info(f"Output: {output_base_dir / experiment_name / 'tabular_y30'}")
    logger.info("")
    
    input_experiment_dir = input_base_dir / experiment_name
    output_experiment_dir = output_base_dir / experiment_name / 'tabular_y30'
    
    if not input_experiment_dir.exists():
        raise FileNotFoundError(f"Input experiment directory not found: {input_experiment_dir}")
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        input_split_dir = input_experiment_dir / split_name
        split_stats = getattr(stats, f'{split_name}_stats')
        
        process_split(
            split_name,
            input_split_dir,
            output_experiment_dir,
            split_stats,
            logger,
            max_partitions,
            max_files,
            keep_extra
        )
        
        # Compute final entity count from output
        output_split_dir = output_experiment_dir / split_name
        split_stats.unique_entities = compute_final_entity_count(output_split_dir, logger)
        
        logger.info("")
    
    stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 60)
    logger.info("ML DATASET BUILDING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processing time: {stats.processing_time_seconds:.2f}s")
    
    return stats


def write_reports(stats: MLDatasetStats, output_dir: Path, logger: logging.Logger) -> None:
    """Write JSON and Markdown reports for ML dataset building."""
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = stats.experiment_name
    
    # JSON report
    json_path = reports_dir / f"ml_tabular_y30_summary_{experiment_name}.json"
    
    report_json = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'target_label': 'y_30',
        'splits': {
            'train': {
                'rows_in': stats.train_stats.rows_in,
                'rows_out': stats.train_stats.rows_out,
                'rows_dropped_missing_y30': stats.train_stats.rows_dropped_missing_y30,
                'partitions_processed': stats.train_stats.partitions_processed,
                'files_processed': stats.train_stats.files_processed,
                'files_failed': stats.train_stats.files_failed,
                'label_distribution': stats.train_stats.label_distribution,
                'unique_entities': stats.train_stats.unique_entities,
                'date_range_min': stats.train_stats.date_range_min,
                'date_range_max': stats.train_stats.date_range_max,
                'missingness_n_features_pct': stats.train_stats.missingness_n_features,
                'missingness_r_features_pct': stats.train_stats.missingness_r_features
            },
            'val': {
                'rows_in': stats.val_stats.rows_in,
                'rows_out': stats.val_stats.rows_out,
                'rows_dropped_missing_y30': stats.val_stats.rows_dropped_missing_y30,
                'partitions_processed': stats.val_stats.partitions_processed,
                'files_processed': stats.val_stats.files_processed,
                'files_failed': stats.val_stats.files_failed,
                'label_distribution': stats.val_stats.label_distribution,
                'unique_entities': stats.val_stats.unique_entities,
                'date_range_min': stats.val_stats.date_range_min,
                'date_range_max': stats.val_stats.date_range_max,
                'missingness_n_features_pct': stats.val_stats.missingness_n_features,
                'missingness_r_features_pct': stats.val_stats.missingness_r_features
            },
            'test': {
                'rows_in': stats.test_stats.rows_in,
                'rows_out': stats.test_stats.rows_out,
                'rows_dropped_missing_y30': stats.test_stats.rows_dropped_missing_y30,
                'partitions_processed': stats.test_stats.partitions_processed,
                'files_processed': stats.test_stats.files_processed,
                'files_failed': stats.test_stats.files_failed,
                'label_distribution': stats.test_stats.label_distribution,
                'unique_entities': stats.test_stats.unique_entities,
                'date_range_min': stats.test_stats.date_range_min,
                'date_range_max': stats.test_stats.date_range_max,
                'missingness_n_features_pct': stats.test_stats.missingness_n_features,
                'missingness_r_features_pct': stats.test_stats.missingness_r_features
            }
        },
        'processing_time_seconds': stats.processing_time_seconds
    }
    
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2, default=str)
    logger.info(f"JSON report written to {json_path}")
    
    # Markdown report
    md_path = reports_dir / f"ml_tabular_y30_summary_{experiment_name}.md"
    with open(md_path, 'w') as f:
        f.write(f"# ML Tabular Dataset Summary - {experiment_name}\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Experiment Name:** `{experiment_name}`\n")
        f.write(f"- **Target Label:** `y_30`\n")
        f.write(f"- **Processing Time:** {stats.processing_time_seconds:.2f}s\n\n")
        
        f.write("## Split Statistics\n\n")
        
        for split_name, split_stats in [
            ('TRAIN', stats.train_stats),
            ('VAL', stats.val_stats),
            ('TEST', stats.test_stats)
        ]:
            f.write(f"### {split_name}\n\n")
            f.write(f"- **Rows (input):** {split_stats.rows_in:,}\n")
            f.write(f"- **Rows (output):** {split_stats.rows_out:,}\n")
            f.write(f"- **Rows dropped (missing y_30):** {split_stats.rows_dropped_missing_y30:,}\n")
            f.write(f"- **Partitions processed:** {split_stats.partitions_processed}\n")
            f.write(f"- **Files processed:** {split_stats.files_processed}\n")
            f.write(f"- **Files failed:** {split_stats.files_failed}\n")
            f.write(f"- **Unique entities:** {split_stats.unique_entities:,}\n")
            if split_stats.date_range_min and split_stats.date_range_max:
                f.write(f"- **Date Range:** {split_stats.date_range_min} to {split_stats.date_range_max}\n")
            f.write(f"- **Missingness (n_* features):** {split_stats.missingness_n_features:.2f}%\n")
            f.write(f"- **Missingness (r_* features):** {split_stats.missingness_r_features:.2f}%\n")
            f.write("\n")
            
            f.write("**Label Distribution (y_30):**\n\n")
            f.write("| Label | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            positive = split_stats.label_distribution.get('positive', 0)
            negative = split_stats.label_distribution.get('negative', 0)
            total = positive + negative
            if total > 0:
                pos_pct = (positive / total * 100)
                neg_pct = (negative / total * 100)
                f.write(f"| Positive (1) | {positive:,} | {pos_pct:.2f}% |\n")
                f.write(f"| Negative (0) | {negative:,} | {neg_pct:.2f}% |\n")
                f.write(f"| **Total** | **{total:,}** | **100.00%** |\n")
            f.write("\n")
    
    logger.info(f"Markdown report written to {md_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build ML-ready tabular datasets from split datasets'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Experiment name (e.g., exp_time_generalisation or exp_time_generalisation__train_eval)'
    )
    parser.add_argument(
        '--max_partitions',
        type=int,
        default=None,
        help='Maximum number of partitions to process per split (for testing, default: all)'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files per partition to process (for testing, default: all)'
    )
    parser.add_argument(
        '--keep_extra',
        action='store_true',
        help='Keep extra columns beyond required ones (disk_id, model, smart_day, y_30, n_*, r_*)'
    )
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / 'configs' / 'data_config.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    log_dir = Path(config['logging']['log_dir'])
    logger = setup_logging(log_dir, config['logging']['level'], args.experiment)
    
    # Base directory
    base_dir = Path(__file__).parent.parent
    
    # Input and output directories
    input_base_dir = base_dir / 'data_splits'
    output_base_dir = base_dir / 'data_ml'
    
    # Build ML dataset
    try:
        stats = build_ml_dataset(
            args.experiment,
            input_base_dir,
            output_base_dir,
            logger,
            args.max_partitions,
            args.max_files,
            args.keep_extra
        )
        
        # Write reports
        write_reports(stats, output_base_dir, logger)
        
        logger.info("=" * 60)
        logger.info("SUCCESS: ML dataset building complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during ML dataset building: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

