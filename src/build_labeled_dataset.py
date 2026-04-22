"""
Build labeled dataset for binary disk failure prediction.

This module joins SMART daily logs with failure labels to create a labeled dataset
following the design in reports/labeling_design.md.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
import yaml
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa


@dataclass
class LabelingStats:
    """Statistics tracked during labeling process."""
    total_input_rows: int = 0
    total_output_rows: int = 0
    rows_dropped_ds_parse_failure: int = 0
    rows_dropped_missing_disk_id: int = 0
    rows_dropped_post_failure: int = 0
    unique_disks_processed: set = field(default_factory=set)
    rows_with_failure_date: int = 0
    label_counts: Dict[int, Dict[str, int]] = field(default_factory=lambda: {7: {'pos': 0, 'neg': 0}, 
                                                                             14: {'pos': 0, 'neg': 0}, 
                                                                             30: {'pos': 0, 'neg': 0}})
    min_smart_day: Optional[date] = None
    max_smart_day: Optional[date] = None
    files_processed: int = 0
    files_failed: int = 0
    processing_time_seconds: float = 0.0


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_failure_map(failure_labels_path: Path, logger: logging.Logger) -> Dict[Any, date]:
    """
    Load failure labels and build a mapping from disk_id to failure_date.
    
    Returns:
        Dictionary mapping disk_id -> failure_date (Python date object)
    """
    logger.info(f"Loading failure labels from {failure_labels_path}")
    df_fail = pd.read_parquet(failure_labels_path)
    
    # Extract failure_date from failure_time
    if 'failure_time_parsed' in df_fail.columns:
        df_fail['failure_date'] = df_fail['failure_time_parsed'].dt.date
    elif 'failure_time' in df_fail.columns:
        # Try to parse if it's datetime
        if pd.api.types.is_datetime64_any_dtype(df_fail['failure_time']):
            df_fail['failure_date'] = df_fail['failure_time'].dt.date
        else:
            # Parse as datetime first
            df_fail['failure_time_parsed'] = pd.to_datetime(df_fail['failure_time'], errors='coerce')
            df_fail['failure_date'] = df_fail['failure_time_parsed'].dt.date
    else:
        raise ValueError("No failure_time or failure_time_parsed column found in failure labels")
    
    # Build mapping with standardized disk_id types
    failure_map = {}
    for _, row in df_fail.iterrows():
        disk_id = row['disk_id']
        failure_date = row['failure_date']
        
        if pd.notna(failure_date) and pd.notna(disk_id):
            # Standardize disk_id type (try int, fallback to string)
            try:
                disk_id_std = int(disk_id)
            except (ValueError, TypeError):
                disk_id_std = str(disk_id)
            
            # Handle duplicates by keeping earliest failure date
            if disk_id_std in failure_map:
                if failure_date < failure_map[disk_id_std]:
                    failure_map[disk_id_std] = failure_date
            else:
                failure_map[disk_id_std] = failure_date
    
    logger.info(f"Loaded {len(failure_map):,} disk failure records")
    
    # Check for duplicates (shouldn't happen if dedup was done correctly)
    if len(failure_map) < len(df_fail):
        logger.warning(f"Found {len(df_fail) - len(failure_map)} duplicate disk_ids in failure labels")
    
    return failure_map


def iter_csv_files(dataset_path: Path, file_pattern: str, max_files: Optional[int] = None) -> Iterator[Path]:
    """Iterate over CSV files in dataset directory."""
    csv_files = sorted(dataset_path.glob(file_pattern))
    if max_files:
        csv_files = csv_files[:max_files]
    return iter(csv_files)


def parse_ds_to_date(ds_series: pd.Series, logger: logging.Logger) -> Tuple[pd.Series, int]:
    """
    Parse ds column (YYYYMMDD) to date.
    
    Returns:
        Tuple of (parsed_date_series, count_of_failures)
    """
    # Convert to string, handling both int and string inputs
    ds_str = ds_series.astype(str)
    
    # Zero-pad to 8 digits if needed (e.g., "2018011" -> "20180101")
    ds_str = ds_str.str.zfill(8)
    
    # Parse to datetime
    try:
        parsed = pd.to_datetime(ds_str, format='%Y%m%d', errors='coerce')
        failures = parsed.isna().sum()
        return parsed.dt.date, int(failures)
    except Exception as e:
        logger.error(f"Error parsing ds column: {e}")
        # Return all NaT on error
        return pd.Series([None] * len(ds_series), dtype='object'), len(ds_series)


def standardize_disk_id(disk_id_series: pd.Series) -> pd.Series:
    """
    Standardize disk_id to consistent type (prefer int64, fallback to string).
    
    Returns:
        Standardized disk_id series
    """
    # Try to convert to int64
    try:
        numeric = pd.to_numeric(disk_id_series, errors='coerce')
        if numeric.notna().sum() == len(disk_id_series):
            return numeric.astype('int64')
    except Exception:
        pass
    
    # Fallback to string
    return disk_id_series.astype(str)


def label_chunk(
    chunk: pd.DataFrame,
    failure_map: Dict[Any, date],
    horizons: List[int],
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Label a chunk of SMART log data.
    
    Returns:
        Tuple of (labeled_chunk, drop_stats)
    """
    drop_stats = {
        'ds_parse_failure': 0,
        'missing_disk_id': 0,
        'post_failure': 0
    }
    
    original_len = len(chunk)
    
    # Standardize disk_id
    chunk['disk_id'] = standardize_disk_id(chunk['disk_id'])
    
    # Parse ds to smart_day
    chunk['smart_day'], parse_failures = parse_ds_to_date(chunk['ds'], logger)
    drop_stats['ds_parse_failure'] = parse_failures
    
    # Drop rows with failed ds parsing
    chunk = chunk[chunk['smart_day'].notna()].copy()
    
    # Drop rows with missing disk_id
    missing_disk_id = chunk['disk_id'].isna().sum()
    drop_stats['missing_disk_id'] = int(missing_disk_id)
    chunk = chunk[chunk['disk_id'].notna()].copy()
    
    if len(chunk) == 0:
        return chunk, drop_stats
    
    # Map failure dates (disk_id is already standardized, failure_map keys are standardized)
    chunk['failure_date'] = chunk['disk_id'].map(failure_map)
    
    # Compute days_to_failure
    chunk['days_to_failure'] = chunk.apply(
        lambda row: (row['failure_date'] - row['smart_day']).days 
        if pd.notna(row['failure_date']) and pd.notna(row['smart_day']) 
        else None,
        axis=1
    )
    
    # Remove post-failure rows (days_to_failure < 0)
    post_failure_mask = (chunk['days_to_failure'].notna()) & (chunk['days_to_failure'] < 0)
    drop_stats['post_failure'] = int(post_failure_mask.sum())
    chunk = chunk[~post_failure_mask].copy()
    
    # Create labels for each horizon
    for horizon in horizons:
        label_col = f'y_{horizon}'
        chunk[label_col] = chunk.apply(
            lambda row: 1 if (pd.notna(row['days_to_failure']) and 
                             0 <= row['days_to_failure'] <= horizon)
            else 0,
            axis=1
        )
    
    return chunk, drop_stats


def write_chunk_partitioned(
    chunk: pd.DataFrame,
    output_base_dir: Path,
    partition_cols: List[str],
    logger: logging.Logger,
    buffer_size: int = 100000
) -> None:
    """
    Write chunk to partitioned parquet files.
    
    Partitions by year and month extracted from smart_day.
    Uses buffering to avoid writing too many small files.
    """
    if len(chunk) == 0:
        return
    
    # Extract year and month from smart_day
    chunk['year'] = chunk['smart_day'].apply(
        lambda x: x.year if pd.notna(x) and isinstance(x, date) else None
    )
    chunk['month'] = chunk['smart_day'].apply(
        lambda x: f"{x.month:02d}" if pd.notna(x) and isinstance(x, date) else None
    )
    
    # Group by partition
    for (year, month), group in chunk.groupby(['year', 'month']):
        if pd.isna(year) or pd.isna(month):
            continue
        
        partition_path = output_base_dir / f"year={year}" / f"month={month}"
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Write to parquet (append if file exists)
        parquet_file = partition_path / "data.parquet"
        
        # Convert date columns to string for parquet compatibility
        group_export = group.copy()
        if 'smart_day' in group_export.columns:
            group_export['smart_day'] = group_export['smart_day'].apply(
                lambda x: x.isoformat() if pd.notna(x) and isinstance(x, date) else None
            )
        if 'failure_date' in group_export.columns:
            group_export['failure_date'] = group_export['failure_date'].apply(
                lambda x: x.isoformat() if pd.notna(x) and isinstance(x, date) else None
            )
        
        # Drop partition columns before writing
        group_export = group_export.drop(columns=['year', 'month'], errors='ignore')
        
        try:
            if parquet_file.exists():
                # Read existing and append
                existing = pd.read_parquet(parquet_file)
                combined = pd.concat([existing, group_export], ignore_index=True)
                combined.to_parquet(parquet_file, index=False, engine='pyarrow')
            else:
                # Write new file
                group_export.to_parquet(parquet_file, index=False, engine='pyarrow')
        except Exception as e:
            logger.error(f"Error writing to {parquet_file}: {e}")
            raise


def update_stats(
    stats: LabelingStats,
    chunk: pd.DataFrame,
    drop_stats: Dict[str, int],
    horizons: List[int],
    original_chunk_size: int
) -> None:
    """Update statistics with chunk data."""
    # Original chunk size is the input (before any processing)
    stats.total_input_rows += original_chunk_size
    stats.total_output_rows += len(chunk)
    stats.rows_dropped_ds_parse_failure += drop_stats['ds_parse_failure']
    stats.rows_dropped_missing_disk_id += drop_stats['missing_disk_id']
    stats.rows_dropped_post_failure += drop_stats['post_failure']
    
    # Update unique disks
    if 'disk_id' in chunk.columns:
        stats.unique_disks_processed.update(chunk['disk_id'].dropna().unique())
    
    # Update rows with failure_date
    if 'failure_date' in chunk.columns:
        stats.rows_with_failure_date += chunk['failure_date'].notna().sum()
    
    # Update label counts
    for horizon in horizons:
        label_col = f'y_{horizon}'
        if label_col in chunk.columns:
            pos_count = (chunk[label_col] == 1).sum()
            neg_count = (chunk[label_col] == 0).sum()
            stats.label_counts[horizon]['pos'] += int(pos_count)
            stats.label_counts[horizon]['neg'] += int(neg_count)
    
    # Update date range
    if 'smart_day' in chunk.columns:
        valid_dates = chunk['smart_day'].dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            if stats.min_smart_day is None or min_date < stats.min_smart_day:
                stats.min_smart_day = min_date
            if stats.max_smart_day is None or max_date > stats.max_smart_day:
                stats.max_smart_day = max_date


def process_dataset(
    dataset_name: str,
    config: Dict[str, Any],
    failure_map: Dict[Any, date],
    horizons: List[int],
    chunksize: int,
    output_dir: Path,
    max_files: Optional[int],
    logger: logging.Logger
) -> LabelingStats:
    """Process entire dataset and return statistics."""
    stats = LabelingStats()
    start_time = datetime.now()
    
    # Setup paths
    data_raw_dir = Path(config['data_raw_dir'])
    dataset_path = data_raw_dir / config['datasets'][dataset_name]['path']
    file_pattern = config['file_pattern']
    
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate over CSV files
    csv_files = list(iter_csv_files(dataset_path, file_pattern, max_files))
    total_files = len(csv_files)
    logger.info(f"Found {total_files} CSV files to process")
    
    for file_idx, csv_file in enumerate(csv_files, 1):
        file_start = datetime.now()
        try:
            logger.info(f"Processing file {file_idx}/{total_files}: {csv_file.name}")
            
            # Process file in chunks
            chunk_iter = pd.read_csv(csv_file, chunksize=chunksize, low_memory=False)
            chunk_count = 0
            
            for chunk in chunk_iter:
                chunk_count += 1
                original_chunk_size = len(chunk)
                
                # Label the chunk
                labeled_chunk, drop_stats = label_chunk(chunk, failure_map, horizons, logger)
                
                # Write chunk
                if len(labeled_chunk) > 0:
                    write_chunk_partitioned(labeled_chunk, output_dir, ['year', 'month'], logger)
                
                # Update statistics
                update_stats(stats, labeled_chunk, drop_stats, horizons, original_chunk_size)
                
                # Periodic logging
                if chunk_count % 10 == 0:
                    logger.info(f"  Processed {chunk_count} chunks, {stats.total_output_rows:,} rows written so far")
            
            stats.files_processed += 1
            file_elapsed = (datetime.now() - file_start).total_seconds()
            logger.info(f"  Completed {csv_file.name} in {file_elapsed:.2f}s ({chunk_count} chunks)")
            
        except Exception as e:
            stats.files_failed += 1
            logger.error(f"Error processing {csv_file.name}: {e}", exc_info=True)
            continue
    
    stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()
    
    return stats


def write_reports(
    stats: LabelingStats,
    dataset_name: str,
    horizons: List[int],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Write JSON and Markdown reports."""
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert stats to dict for JSON
    stats_dict = asdict(stats)
    # Convert set to count for unique_disks_processed
    if isinstance(stats_dict['unique_disks_processed'], set):
        stats_dict['unique_disks_processed'] = len(stats_dict['unique_disks_processed'])
    # Convert date objects to strings
    if stats_dict['min_smart_day']:
        stats_dict['min_smart_day'] = str(stats_dict['min_smart_day'])
    if stats_dict['max_smart_day']:
        stats_dict['max_smart_day'] = str(stats_dict['max_smart_day'])
    
    # JSON report
    json_path = reports_dir / f"labeling_summary_{dataset_name}.json"
    report_json = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'output_directory': str(output_dir),
        'statistics': stats_dict
    }
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2, default=str)
    logger.info(f"JSON report written to {json_path}")
    
    # Markdown report
    md_path = reports_dir / f"labeling_summary_{dataset_name}.md"
    with open(md_path, 'w') as f:
        f.write(f"# Labeling Summary: {dataset_name}\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Output Directory:** `{output_dir}`\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Total input rows:** {stats.total_input_rows:,}\n")
        f.write(f"- **Total output rows:** {stats.total_output_rows:,}\n")
        f.write(f"- **Files processed:** {stats.files_processed}\n")
        f.write(f"- **Files failed:** {stats.files_failed}\n")
        f.write(f"- **Processing time:** {stats.processing_time_seconds:.2f} seconds\n")
        f.write(f"- **Unique disks processed:** {len(stats.unique_disks_processed):,}\n")
        f.write("\n")
        
        f.write("## Rows Dropped\n\n")
        f.write(f"- **DS parse failures:** {stats.rows_dropped_ds_parse_failure:,}\n")
        f.write(f"- **Missing disk_id:** {stats.rows_dropped_missing_disk_id:,}\n")
        f.write(f"- **Post-failure rows (days_to_failure < 0):** {stats.rows_dropped_post_failure:,}\n")
        f.write("\n")
        
        f.write("## Failure Date Coverage\n\n")
        if stats.total_output_rows > 0:
            failure_coverage = (stats.rows_with_failure_date / stats.total_output_rows * 100)
            f.write(f"- **Rows with failure_date:** {stats.rows_with_failure_date:,} ({failure_coverage:.2f}%)\n")
        f.write("\n")
        
        f.write("## Date Range\n\n")
        if stats.min_smart_day and stats.max_smart_day:
            f.write(f"- **Min smart_day:** {stats.min_smart_day}\n")
            f.write(f"- **Max smart_day:** {stats.max_smart_day}\n")
        f.write("\n")
        
        f.write("## Label Distribution\n\n")
        for horizon in horizons:
            pos = stats.label_counts[horizon]['pos']
            neg = stats.label_counts[horizon]['neg']
            total = pos + neg
            if total > 0:
                pos_pct = (pos / total * 100)
                f.write(f"### Horizon H={horizon} days\n\n")
                f.write(f"- **Positive (y_{horizon}=1):** {pos:,} ({pos_pct:.2f}%)\n")
                f.write(f"- **Negative (y_{horizon}=0):** {neg:,} ({100-pos_pct:.2f}%)\n")
                f.write(f"- **Total:** {total:,}\n")
                f.write("\n")
    
    logger.info(f"Markdown report written to {md_path}")


def setup_logging(log_dir: Path, log_level: str, dataset_name: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"labeling_{dataset_name}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Build labeled dataset for disk failure prediction')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., smartlog2018ssd)')
    parser.add_argument('--chunksize', type=int, default=None, help='Chunk size for CSV reading')
    parser.add_argument('--out_dir', type=str, default='data_clean', help='Output directory base')
    parser.add_argument('--horizons', type=int, nargs='+', default=[7, 14, 30], help='Prediction horizons in days')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process (for debugging)')
    parser.add_argument('--write_columns', type=str, default=None, help='Columns to write (default: all)')
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / 'configs' / 'data_config.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Validate dataset
    if args.dataset not in config['datasets']:
        print(f"Error: Dataset '{args.dataset}' not found in config")
        print(f"Available datasets: {list(config['datasets'].keys())}")
        sys.exit(1)
    
    log_dir = Path(config['logging']['log_dir'])
    logger = setup_logging(log_dir, config['logging']['level'], args.dataset)
    
    logger.info("=" * 60)
    logger.info("BUILDING LABELED DATASET")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Horizons: {args.horizons}")
    
    # Get chunksize
    chunksize = args.chunksize if args.chunksize else config['processing']['chunk_size']
    logger.info(f"Chunk size: {chunksize:,}")
    
    # Load failure map
    failure_labels_path = Path('data_interim') / 'labels' / 'failure_labels_dedup.parquet'
    if not failure_labels_path.exists():
        logger.error(f"Failure labels file not found at {failure_labels_path}")
        sys.exit(1)
    
    failure_map = load_failure_map(failure_labels_path, logger)
    
    # Setup output directory
    output_dir = Path(args.out_dir) / f"labeled_{args.dataset}"
    
    # Process dataset
    try:
        stats = process_dataset(
            args.dataset,
            config,
            failure_map,
            args.horizons,
            chunksize,
            output_dir,
            args.max_files,
            logger
        )
        
        # Write reports
        write_reports(stats, args.dataset, args.horizons, output_dir, logger)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Input rows: {stats.total_input_rows:,}")
        logger.info(f"Output rows: {stats.total_output_rows:,}")
        logger.info(f"Rows dropped:")
        logger.info(f"  - DS parse failures: {stats.rows_dropped_ds_parse_failure:,}")
        logger.info(f"  - Missing disk_id: {stats.rows_dropped_missing_disk_id:,}")
        logger.info(f"  - Post-failure: {stats.rows_dropped_post_failure:,}")
        logger.info(f"Unique disks: {len(stats.unique_disks_processed):,}")
        logger.info(f"Processing time: {stats.processing_time_seconds:.2f}s")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

