"""
Create time-based Train/Val/Test splits from cleaned datasets (Stage 7 output).

This script performs non-destructive time-based splitting with entity-level disjointness
to prevent data leakage. It reads from Stage 7 cleaned output and writes new split datasets.

Split Policy:
- TRAIN: All rows with smart_day in year 2018
- VAL: 2019-01-01 to 2019-06-30 inclusive
- TEST: 2019-07-01 to 2019-12-31 inclusive

Entity Disjointness:
- Remove from TRAIN any entity_id (disk_id, model) that appears in VAL or TEST
- Remove from VAL any entity_id that appears in TEST
- TEST remains unchanged
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
class SplitStats:
    """Statistics for a single split."""
    split_name: str = ""
    rows_before: int = 0
    rows_after_date_filter: int = 0
    rows_after_entity_filter: int = 0
    rows_dropped_date_filter: int = 0
    rows_dropped_entity_overlap: int = 0
    rows_final: int = 0  # Same as rows_after_entity_filter
    entities_before: int = 0
    entities_after: int = 0
    date_range_min: Optional[str] = None
    date_range_max: Optional[str] = None
    label_counts: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        'y_7': {'positive': 0, 'negative': 0},
        'y_14': {'positive': 0, 'negative': 0},
        'y_30': {'positive': 0, 'negative': 0}
    })


@dataclass
class SplittingStats:
    """Overall statistics for the splitting process."""
    experiment_name: str = ""
    train_dataset: str = ""
    eval_dataset: str = ""
    entity_disjoint_policy: str = "none"
    input_rows_total: int = 0
    train_stats: SplitStats = field(default_factory=lambda: SplitStats(split_name='train'))
    val_stats: SplitStats = field(default_factory=lambda: SplitStats(split_name='val'))
    test_stats: SplitStats = field(default_factory=lambda: SplitStats(split_name='test'))
    processing_time_seconds: float = 0.0


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path, log_level: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"splits_{experiment_name}.log"
    
    # Get root logger
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


def locate_stage7_dir(dataset_name: str, base_dir: Path) -> Path:
    """
    Locate the final cleaned dataset directory for a dataset.
    
    Stage 7 is read-only validation, so we look for the cleaned data that Stage 7 validates.
    Checks in order (matching Stage 7 input detection logic):
    1. data_interim/clean_stage6_<dataset> (Stage 6 output)
    2. data_interim/clean_stage4_<dataset> (Stage 4 output)
    3. data_interim/clean_stage3_<dataset> (Stage 3 output)
    4. data_interim/clean_stage2_<dataset> (Stage 2 output)
    5. data_interim/clean_stage1_<dataset> (Stage 1 output)
    6. data_clean/labeled_<dataset> (labeled dataset)
    
    Args:
        dataset_name: Name of the dataset
        base_dir: Base project directory
        
    Returns:
        Path to cleaned dataset directory
        
    Raises:
        FileNotFoundError: If no cleaned data directory exists
    """
    candidates = [
        base_dir / 'data_interim' / f'clean_stage6_{dataset_name}',
        base_dir / 'data_interim' / f'clean_stage4_{dataset_name}',
        base_dir / 'data_interim' / f'clean_stage3_{dataset_name}',
        base_dir / 'data_interim' / f'clean_stage2_{dataset_name}',
        base_dir / 'data_interim' / f'clean_stage1_{dataset_name}',
        base_dir / 'data_clean' / f'labeled_{dataset_name}',
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Verify it has parquet files
            if any(candidate.rglob("*.parquet")):
                return candidate
    
    raise FileNotFoundError(
        f"Cleaned dataset not found for '{dataset_name}'. "
        f"Checked: {[str(c) for c in candidates]}. "
        f"Please run the cleaning pipeline (Stage 6 or earlier) first."
    )


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
        year_part = partition_dir.parent.name  # e.g., "year=2018"
        month_part = partition_dir.name  # e.g., "month=01"
        partition_name = f"{year_part}/{month_part}"
        
        # Find parquet files in partition
        parquet_files = sorted(partition_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {partition_name}, skipping")
            continue
        
        yield partition_dir, partition_name, parquet_files


def build_entity_set_for_timerange(
    input_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    logger: logging.Logger,
    max_partitions: Optional[int] = None
) -> Set[Tuple[int, str]]:
    """
    Build set of entity_ids (disk_id, model) for a given time range.
    
    Args:
        input_dir: Directory containing partitioned parquet files
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        logger: Logger instance
        
    Returns:
        Set of (disk_id, model) tuples
    """
    entity_set: Set[Tuple[int, str]] = set()
    rows_scanned = 0
    
    logger.info(f"Building entity set for time range: {start_date.date()} to {end_date.date()}")
    
    for partition_dir, partition_name, parquet_files in iter_partitions(input_dir, logger, max_partitions):
        for parquet_file in parquet_files:
            try:
                # Read parquet file
                df = pd.read_parquet(parquet_file)
                rows_scanned += len(df)
                
                # Filter by date range
                if 'smart_day' not in df.columns:
                    logger.warning(f"Column 'smart_day' not found in {partition_name}, skipping")
                    continue
                
                # Convert smart_day to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df['smart_day']):
                    df['smart_day'] = pd.to_datetime(df['smart_day'], errors='coerce')
                
                # Filter rows in date range
                date_mask = (df['smart_day'] >= start_date) & (df['smart_day'] <= end_date)
                df_filtered = df[date_mask]
                
                if len(df_filtered) == 0:
                    continue
                
                # Extract entity_ids
                if 'disk_id' not in df_filtered.columns or 'model' not in df_filtered.columns:
                    logger.warning(f"Missing disk_id or model in {partition_name}, skipping")
                    continue
                
                # Create entity_id tuples (handle NaN by dropping)
                valid_mask = df_filtered['disk_id'].notna() & df_filtered['model'].notna()
                valid_df = df_filtered[valid_mask]
                
                if len(valid_df) > 0:
                    entities = set(zip(
                        valid_df['disk_id'].astype(int),
                        valid_df['model'].astype(str)
                    ))
                    entity_set.update(entities)
                    
            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}", exc_info=True)
                continue
    
    logger.info(f"Built entity set: {len(entity_set):,} unique entities from {rows_scanned:,} rows")
    return entity_set


def filter_and_write_partition(
    partition_dir: Path,
    partition_name: str,
    parquet_files: List[Path],
    output_dir: Path,
    split_name: str,
    date_filter_start: Optional[pd.Timestamp],
    date_filter_end: Optional[pd.Timestamp],
    exclude_entities: Set[Tuple[int, str]],
    stats: SplitStats,
    logger: logging.Logger
) -> None:
    """
    Filter a partition by date and entity exclusion, then write to output.
    
    Args:
        partition_dir: Source partition directory
        partition_name: Partition name (e.g., "year=2018/month=01")
        parquet_files: List of parquet files in partition
        output_dir: Base output directory
        split_name: Name of split (train/val/test)
        date_filter_start: Start date filter (inclusive, None = no filter)
        date_filter_end: End date filter (inclusive, None = no filter)
        exclude_entities: Set of (disk_id, model) tuples to exclude
        stats: SplitStats object to update
        logger: Logger instance
    """
    partition_rows_before = 0
    partition_rows_after_date = 0
    partition_rows_after_entity = 0
    partition_entities_before: Set[Tuple[int, str]] = set()
    partition_entities_after: Set[Tuple[int, str]] = set()
    partition_dates: List[pd.Timestamp] = []
    
    all_dataframes: List[pd.DataFrame] = []
    
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            partition_rows_before += len(df)
            
            # Apply date filter if specified
            if date_filter_start is not None or date_filter_end is not None:
                if 'smart_day' not in df.columns:
                    logger.warning(f"Column 'smart_day' not found in {partition_name}, skipping file")
                    continue
                
                # Convert smart_day to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df['smart_day']):
                    df['smart_day'] = pd.to_datetime(df['smart_day'], errors='coerce')
                
                date_mask = pd.Series(True, index=df.index)
                if date_filter_start is not None:
                    date_mask &= (df['smart_day'] >= date_filter_start)
                if date_filter_end is not None:
                    date_mask &= (df['smart_day'] <= date_filter_end)
                
                df = df[date_mask]
            
            partition_rows_after_date += len(df)
            
            if len(df) == 0:
                continue
            
            # Track entities before entity exclusion
            if 'disk_id' in df.columns and 'model' in df.columns:
                valid_mask = df['disk_id'].notna() & df['model'].notna()
                valid_df = df[valid_mask]
                if len(valid_df) > 0:
                    entities = set(zip(
                        valid_df['disk_id'].astype(int),
                        valid_df['model'].astype(str)
                    ))
                    partition_entities_before.update(entities)
            
            # Apply entity exclusion
            if exclude_entities and 'disk_id' in df.columns and 'model' in df.columns:
                # Create entity_id column for filtering
                valid_mask = df['disk_id'].notna() & df['model'].notna()
                entity_ids = pd.Series(
                    list(zip(
                        df.loc[valid_mask, 'disk_id'].astype(int),
                        df.loc[valid_mask, 'model'].astype(str)
                    )),
                    index=df[valid_mask].index
                )
                
                # Filter out excluded entities
                exclude_mask = entity_ids.isin(exclude_entities)
                df = df[~exclude_mask.reindex(df.index, fill_value=False)]
            
            partition_rows_after_entity += len(df)
            
            if len(df) == 0:
                continue
            
            # Track entities after exclusion
            if 'disk_id' in df.columns and 'model' in df.columns:
                valid_mask = df['disk_id'].notna() & df['model'].notna()
                valid_df = df[valid_mask]
                if len(valid_df) > 0:
                    entities = set(zip(
                        valid_df['disk_id'].astype(int),
                        valid_df['model'].astype(str)
                    ))
                    partition_entities_after.update(entities)
            
            # Track dates
            if 'smart_day' in df.columns:
                valid_dates = df['smart_day'].dropna()
                if len(valid_dates) > 0:
                    partition_dates.extend(valid_dates.tolist())
            
            all_dataframes.append(df)
            
        except Exception as e:
            logger.error(f"Error processing {parquet_file}: {e}", exc_info=True)
            continue
    
    # Combine all dataframes for this partition
    if len(all_dataframes) == 0:
        # No data after filtering - don't create output directory
        logger.debug(f"  {partition_name}: No data after filtering, skipping (no output directory created)")
        return
    
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    # Only create output directory if we have data to write
    output_partition_dir = output_dir / split_name / partition_name
    output_partition_dir.mkdir(parents=True, exist_ok=True)
    
    # Write combined partition
    output_file = output_partition_dir / "data.parquet"
    df_combined.to_parquet(output_file, engine='pyarrow', index=False)
    
    # Update stats
    stats.rows_before += partition_rows_before
    stats.rows_after_date_filter += partition_rows_after_date
    stats.rows_after_entity_filter += partition_rows_after_entity
    stats.rows_dropped_date_filter += (partition_rows_before - partition_rows_after_date)
    stats.rows_dropped_entity_overlap += (partition_rows_after_date - partition_rows_after_entity)
    stats.rows_final += partition_rows_after_entity
    
    logger.info(
        f"  {partition_name}: {partition_rows_before:,} -> {partition_rows_after_date:,} (date) -> "
        f"{partition_rows_after_entity:,} (entity) rows "
        f"(dropped: {partition_rows_before - partition_rows_after_date:,} date, "
        f"{partition_rows_after_date - partition_rows_after_entity:,} entity)"
    )


def compute_label_counts(df: pd.DataFrame, label_cols: List[str] = ['y_7', 'y_14', 'y_30']) -> Dict[str, Dict[str, int]]:
    """Compute positive/negative counts for label columns."""
    counts = {}
    for col in label_cols:
        if col in df.columns:
            positive = int((df[col] == 1).sum())
            negative = int((df[col] == 0).sum())
            counts[col] = {'positive': positive, 'negative': negative}
        else:
            counts[col] = {'positive': 0, 'negative': 0}
    return counts


def create_splits(
    train_input_dir: Path,
    eval_input_dir: Path,
    output_base_dir: Path,
    experiment_name: str,
    overwrite: bool,
    logger: logging.Logger,
    max_partitions: Optional[int] = None,
    entity_disjoint: str = "none"
) -> SplittingStats:
    """
    Create time-based train/val/test splits with entity disjointness.
    
    Args:
        train_input_dir: Directory containing 2018 data (TRAIN)
        eval_input_dir: Directory containing 2019 data (VAL/TEST)
        output_base_dir: Base output directory
        experiment_name: Name of the experiment (used for output directory and reports)
        overwrite: Whether to overwrite existing output
        logger: Logger instance
        max_partitions: Maximum number of partitions to process (None = all)
        entity_disjoint: Entity disjointness policy ('none', 'train_eval', or 'all')
        
    Returns:
        SplittingStats object with statistics
    """
    start_time = datetime.now()
    stats = SplittingStats()
    stats.experiment_name = experiment_name
    
    # Define time ranges
    train_start = pd.Timestamp('2018-01-01')
    train_end = pd.Timestamp('2018-12-31 23:59:59')
    val_start = pd.Timestamp('2019-01-01')
    val_end = pd.Timestamp('2019-06-30 23:59:59')
    test_start = pd.Timestamp('2019-07-01')
    test_end = pd.Timestamp('2019-12-31 23:59:59')
    
    # Output directories
    output_dir = output_base_dir / experiment_name
    train_output = output_dir / 'train'
    val_output = output_dir / 'val'
    test_output = output_dir / 'test'
    
    # Handle overwrite - only delete the specific experiment folder
    if overwrite and output_dir.exists():
        logger.info(f"Overwriting existing output directory: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("TIME-BASED DATASET SPLITTING")
    logger.info("=" * 60)
    logger.info(f"Entity Disjoint Policy: {entity_disjoint}")
    logger.info(f"TRAIN: {train_start.date()} to {train_end.date()}")
    logger.info(f"VAL: {val_start.date()} to {val_end.date()}")
    logger.info(f"TEST: {test_start.date()} to {test_end.date()}")
    logger.info("")
    
    stats.entity_disjoint_policy = entity_disjoint
    
    # PASS 1: Build entity sets for VAL and TEST (always build, but may not use)
    logger.info("PASS 1: Building entity sets for VAL and TEST...")
    val_entities = build_entity_set_for_timerange(eval_input_dir, val_start, val_end, logger, max_partitions)
    test_entities = build_entity_set_for_timerange(eval_input_dir, test_start, test_end, logger, max_partitions)
    
    # Apply entity disjointness policy
    if entity_disjoint == "none":
        train_exclude_entities = set()
        val_exclude_entities = set()
        logger.info("Entity disjointness: NONE (no entity exclusions)")
    elif entity_disjoint == "train_eval":
        train_exclude_entities = val_entities | test_entities
        val_exclude_entities = set()
        logger.info(f"Entity disjointness: TRAIN_EVAL (TRAIN excludes VAL ∪ TEST: {len(train_exclude_entities):,} entities)")
    elif entity_disjoint == "all":
        train_exclude_entities = val_entities | test_entities
        val_exclude_entities = test_entities
        logger.info(f"Entity disjointness: ALL (TRAIN excludes VAL ∪ TEST: {len(train_exclude_entities):,}, VAL excludes TEST: {len(val_exclude_entities):,})")
    else:
        raise ValueError(f"Invalid entity_disjoint policy: {entity_disjoint}. Must be 'none', 'train_eval', or 'all'")
    
    # Count total input rows (unique files from both input directories)
    logger.info("Counting total input rows...")
    files_counted: Set[Path] = set()
    
    for partition_dir, partition_name, parquet_files in iter_partitions(train_input_dir, logger, max_partitions):
        for parquet_file in parquet_files:
            if parquet_file not in files_counted:
                try:
                    df = pd.read_parquet(parquet_file)
                    stats.input_rows_total += len(df)
                    files_counted.add(parquet_file)
                except:
                    pass
    
    for partition_dir, partition_name, parquet_files in iter_partitions(eval_input_dir, logger, max_partitions):
        for parquet_file in parquet_files:
            if parquet_file not in files_counted:
                try:
                    df = pd.read_parquet(parquet_file)
                    stats.input_rows_total += len(df)
                    files_counted.add(parquet_file)
                except:
                    pass
    
    logger.info(f"Total input rows: {stats.input_rows_total:,}")
    
    # PASS 2: Write TRAIN partitions (2018 data, exclude VAL/TEST entities)
    logger.info("")
    logger.info("PASS 2: Writing TRAIN partitions...")
    train_entities_before: Set[Tuple[int, str]] = set()
    
    for partition_dir, partition_name, parquet_files in iter_partitions(train_input_dir, logger, max_partitions):
        # Track entities before exclusion (for stats)
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                if 'disk_id' in df.columns and 'model' in df.columns:
                    valid_mask = df['disk_id'].notna() & df['model'].notna()
                    valid_df = df[valid_mask]
                    if len(valid_df) > 0:
                        entities = set(zip(
                            valid_df['disk_id'].astype(int),
                            valid_df['model'].astype(str)
                        ))
                        train_entities_before.update(entities)
            except:
                pass
        
        filter_and_write_partition(
            partition_dir, partition_name, parquet_files,
            output_dir, 'train',
            train_start, train_end,
            train_exclude_entities,
            stats.train_stats,
            logger
        )
    
    # PASS 3: Write VAL partitions (2019-01 to 2019-06, exclude TEST entities)
    logger.info("")
    logger.info("PASS 3: Writing VAL partitions...")
    val_entities_before: Set[Tuple[int, str]] = set()
    
    for partition_dir, partition_name, parquet_files in iter_partitions(eval_input_dir, logger, max_partitions):
        # Track entities before exclusion (only in VAL time range)
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                if 'smart_day' in df.columns and 'disk_id' in df.columns and 'model' in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df['smart_day']):
                        df['smart_day'] = pd.to_datetime(df['smart_day'], errors='coerce')
                    date_mask = (df['smart_day'] >= val_start) & (df['smart_day'] <= val_end)
                    val_df = df[date_mask]
                    if len(val_df) > 0:
                        valid_mask = val_df['disk_id'].notna() & val_df['model'].notna()
                        valid_val_df = val_df[valid_mask]
                        if len(valid_val_df) > 0:
                            entities = set(zip(
                                valid_val_df['disk_id'].astype(int),
                                valid_val_df['model'].astype(str)
                            ))
                            val_entities_before.update(entities)
            except:
                pass
        
        filter_and_write_partition(
            partition_dir, partition_name, parquet_files,
            output_dir, 'val',
            val_start, val_end,
            val_exclude_entities,
            stats.val_stats,
            logger
        )
    
    # PASS 4: Write TEST partitions (2019-07 to 2019-12, no exclusions)
    logger.info("")
    logger.info("PASS 4: Writing TEST partitions...")
    test_entities_before: Set[Tuple[int, str]] = set()
    
    for partition_dir, partition_name, parquet_files in iter_partitions(eval_input_dir, logger, max_partitions):
        # Track entities before (only in TEST time range)
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                if 'smart_day' in df.columns and 'disk_id' in df.columns and 'model' in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df['smart_day']):
                        df['smart_day'] = pd.to_datetime(df['smart_day'], errors='coerce')
                    date_mask = (df['smart_day'] >= test_start) & (df['smart_day'] <= test_end)
                    test_df = df[date_mask]
                    if len(test_df) > 0:
                        valid_mask = test_df['disk_id'].notna() & test_df['model'].notna()
                        valid_test_df = test_df[valid_mask]
                        if len(valid_test_df) > 0:
                            entities = set(zip(
                                valid_test_df['disk_id'].astype(int),
                                valid_test_df['model'].astype(str)
                            ))
                            test_entities_before.update(entities)
            except:
                pass
        
        filter_and_write_partition(
            partition_dir, partition_name, parquet_files,
            output_dir, 'test',
            test_start, test_end,
            set(),  # No exclusions for TEST
            stats.test_stats,
            logger
        )
    
    # Compute final statistics by reading output partitions
    logger.info("")
    logger.info("Computing final statistics...")
    
    # Compute entity counts and label counts for each split
    for split_name, split_output, split_stats in [
        ('train', train_output, stats.train_stats),
        ('val', val_output, stats.val_stats),
        ('test', test_output, stats.test_stats)
    ]:
        if not split_output.exists():
            continue
        
        split_entities: Set[Tuple[int, str]] = set()
        split_dates: List[pd.Timestamp] = []
        all_label_dfs: List[pd.DataFrame] = []
        
        for partition_dir, partition_name, parquet_files in iter_partitions(split_output, logger, None):
            for parquet_file in parquet_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    
                    # Track entities
                    if 'disk_id' in df.columns and 'model' in df.columns:
                        valid_mask = df['disk_id'].notna() & df['model'].notna()
                        valid_df = df[valid_mask]
                        if len(valid_df) > 0:
                            entities = set(zip(
                                valid_df['disk_id'].astype(int),
                                valid_df['model'].astype(str)
                            ))
                            split_entities.update(entities)
                    
                    # Track dates
                    if 'smart_day' in df.columns:
                        valid_dates = df['smart_day'].dropna()
                        if len(valid_dates) > 0:
                            split_dates.extend(valid_dates.tolist())
                    
                    # Collect for label counting
                    label_cols = ['y_7', 'y_14', 'y_30']
                    if any(col in df.columns for col in label_cols):
                        all_label_dfs.append(df[label_cols].copy() if all(col in df.columns for col in label_cols) else df)
                        
                except Exception as e:
                    logger.warning(f"Error reading {parquet_file} for stats: {e}")
                    continue
        
        split_stats.entities_after = len(split_entities)
        
        if split_dates:
            split_stats.date_range_min = str(pd.Series(split_dates).min())
            split_stats.date_range_max = str(pd.Series(split_dates).max())
        
        if all_label_dfs:
            combined_labels = pd.concat(all_label_dfs, ignore_index=True)
            split_stats.label_counts = compute_label_counts(combined_labels)
    
    # Set entity counts (before = entities in time range, after = entities after exclusion)
    stats.train_stats.entities_before = len(train_entities_before)
    stats.val_stats.entities_before = len(val_entities_before)
    stats.test_stats.entities_before = len(test_entities_before)
    
    # Ensure rows_final matches rows_after_entity_filter
    stats.train_stats.rows_final = stats.train_stats.rows_after_entity_filter
    stats.val_stats.rows_final = stats.val_stats.rows_after_entity_filter
    stats.test_stats.rows_final = stats.test_stats.rows_after_entity_filter
    
    stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPLITTING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Input rows: {stats.input_rows_total:,}")
    logger.info(f"TRAIN: {stats.train_stats.rows_final:,} rows, {stats.train_stats.entities_after:,} entities")
    logger.info(f"VAL: {stats.val_stats.rows_final:,} rows, {stats.val_stats.entities_after:,} entities")
    logger.info(f"TEST: {stats.test_stats.rows_final:,} rows, {stats.test_stats.entities_after:,} entities")
    logger.info(f"Processing time: {stats.processing_time_seconds:.2f}s")
    
    return stats


def write_split_reports(stats: SplittingStats, output_dir: Path, logger: logging.Logger) -> None:
    """Write JSON and Markdown reports for the splitting process."""
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Use experiment name for reports
    experiment_name = stats.experiment_name
    
    # JSON report
    json_path = reports_dir / f"splits_summary_{experiment_name}.json"
    
    report_json = {
        'experiment_name': experiment_name,
        'train_dataset': stats.train_dataset,
        'eval_dataset': stats.eval_dataset,
        'entity_disjoint_policy': stats.entity_disjoint_policy,
        'timestamp': datetime.now().isoformat(),
        'time_windows': {
            'train': {'start': '2018-01-01', 'end': '2018-12-31'},
            'val': {'start': '2019-01-01', 'end': '2019-06-30'},
            'test': {'start': '2019-07-01', 'end': '2019-12-31'}
        },
        'input_rows_total': stats.input_rows_total,
        'splits': {
            'train': {
                'rows_before': stats.train_stats.rows_before,
                'rows_after_date_filter': stats.train_stats.rows_after_date_filter,
                'rows_after_entity_filter': stats.train_stats.rows_after_entity_filter,
                'rows_dropped_date_filter': stats.train_stats.rows_dropped_date_filter,
                'rows_dropped_entity_overlap': stats.train_stats.rows_dropped_entity_overlap,
                'rows_final': stats.train_stats.rows_final,
                'entities_before': stats.train_stats.entities_before,
                'entities_after': stats.train_stats.entities_after,
                'date_range_min': stats.train_stats.date_range_min,
                'date_range_max': stats.train_stats.date_range_max,
                'label_counts': stats.train_stats.label_counts
            },
            'val': {
                'rows_before': stats.val_stats.rows_before,
                'rows_after_date_filter': stats.val_stats.rows_after_date_filter,
                'rows_after_entity_filter': stats.val_stats.rows_after_entity_filter,
                'rows_dropped_date_filter': stats.val_stats.rows_dropped_date_filter,
                'rows_dropped_entity_overlap': stats.val_stats.rows_dropped_entity_overlap,
                'rows_final': stats.val_stats.rows_final,
                'entities_before': stats.val_stats.entities_before,
                'entities_after': stats.val_stats.entities_after,
                'date_range_min': stats.val_stats.date_range_min,
                'date_range_max': stats.val_stats.date_range_max,
                'label_counts': stats.val_stats.label_counts
            },
            'test': {
                'rows_before': stats.test_stats.rows_before,
                'rows_after_date_filter': stats.test_stats.rows_after_date_filter,
                'rows_after_entity_filter': stats.test_stats.rows_after_entity_filter,
                'rows_dropped_date_filter': stats.test_stats.rows_dropped_date_filter,
                'rows_dropped_entity_overlap': stats.test_stats.rows_dropped_entity_overlap,
                'rows_final': stats.test_stats.rows_final,
                'entities_before': stats.test_stats.entities_before,
                'entities_after': stats.test_stats.entities_after,
                'date_range_min': stats.test_stats.date_range_min,
                'date_range_max': stats.test_stats.date_range_max,
                'label_counts': stats.test_stats.label_counts
            }
        },
        'processing_time_seconds': stats.processing_time_seconds
    }
    
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2, default=str)
    logger.info(f"JSON report written to {json_path}")
    
    # Markdown report
    md_path = reports_dir / f"splits_summary_{experiment_name}.md"
    with open(md_path, 'w') as f:
        f.write(f"# Time-Based Dataset Splitting Summary - {experiment_name}\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Experiment Name:** `{experiment_name}`\n")
        f.write(f"- **Train Dataset:** {stats.train_dataset}\n")
        f.write(f"- **Eval Dataset:** {stats.eval_dataset}\n")
        f.write(f"- **Entity Disjoint Policy:** `{stats.entity_disjoint_policy}`\n")
        f.write(f"- **Total Input Rows:** {stats.input_rows_total:,}\n")
        f.write(f"- **Processing Time:** {stats.processing_time_seconds:.2f}s\n\n")
        
        f.write("## Split Definitions\n\n")
        f.write("- **TRAIN:** All rows with `smart_day` in year 2018\n")
        f.write("- **VAL:** 2019-01-01 to 2019-06-30 (inclusive)\n")
        f.write("- **TEST:** 2019-07-01 to 2019-12-31 (inclusive)\n\n")
        
        f.write("## Entity Disjointness Policy\n\n")
        if stats.entity_disjoint_policy == "none":
            f.write("**Policy: NONE** - No entity exclusions. Splits are purely time-based.\n\n")
        elif stats.entity_disjoint_policy == "train_eval":
            f.write("**Policy: TRAIN_EVAL** - Removed from TRAIN any entity (disk_id, model) that appears in VAL or TEST. VAL has no exclusions.\n\n")
        elif stats.entity_disjoint_policy == "all":
            f.write("**Policy: ALL** - Removed from TRAIN any entity (disk_id, model) that appears in VAL or TEST. Removed from VAL any entity that appears in TEST. TEST remains unchanged.\n\n")
        f.write("\n")
        
        f.write("## Split Statistics\n\n")
        
        for split_name, split_stats in [
            ('TRAIN', stats.train_stats),
            ('VAL', stats.val_stats),
            ('TEST', stats.test_stats)
        ]:
            f.write(f"### {split_name}\n\n")
            f.write(f"- **Rows (before filtering):** {split_stats.rows_before:,}\n")
            f.write(f"- **Rows (after date filter):** {split_stats.rows_after_date_filter:,}\n")
            f.write(f"- **Rows (after entity filter):** {split_stats.rows_after_entity_filter:,}\n")
            f.write(f"- **Rows dropped (date filter):** {split_stats.rows_dropped_date_filter:,}\n")
            f.write(f"- **Rows dropped (entity overlap):** {split_stats.rows_dropped_entity_overlap:,}\n")
            f.write(f"- **Rows (final):** {split_stats.rows_final:,}\n")
            f.write(f"- **Entities (before):** {split_stats.entities_before:,}\n")
            f.write(f"- **Entities (after):** {split_stats.entities_after:,}\n")
            if split_stats.date_range_min and split_stats.date_range_max:
                f.write(f"- **Date Range:** {split_stats.date_range_min} to {split_stats.date_range_max}\n")
            f.write("\n")
            
            f.write("**Label Distribution:**\n\n")
            f.write("| Label | Positive | Negative | Total |\n")
            f.write("|-------|----------|----------|-------|\n")
            for label_col in ['y_7', 'y_14', 'y_30']:
                if label_col in split_stats.label_counts:
                    counts = split_stats.label_counts[label_col]
                    positive = counts.get('positive', 0)
                    negative = counts.get('negative', 0)
                    total = positive + negative
                    f.write(f"| `{label_col}` | {positive:,} | {negative:,} | {total:,} |\n")
            f.write("\n")
    
    logger.info(f"Markdown report written to {md_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create time-based train/val/test splits from cleaned datasets'
    )
    parser.add_argument(
        '--train_dataset',
        type=str,
        default='smartlog2018ssd',
        help='Dataset name for training data (default: smartlog2018ssd)'
    )
    parser.add_argument(
        '--eval_dataset',
        type=str,
        default='smartlog2019ssd',
        help='Dataset name for validation/test data (default: smartlog2019ssd)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output directory if it exists'
    )
    parser.add_argument(
        '--max_partitions',
        type=int,
        default=None,
        help='Maximum number of partitions to process per dataset (for testing, default: all)'
    )
    parser.add_argument(
        '--entity_disjoint',
        type=str,
        choices=['none', 'train_eval', 'all'],
        default='none',
        help='Entity disjointness policy: none (no exclusions), train_eval (TRAIN excludes VAL∪TEST), all (TRAIN excludes VAL∪TEST, VAL excludes TEST)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name for output directory and reports. If not provided, auto-builds as "exp_time_generalisation__{entity_disjoint}"'
    )
    
    args = parser.parse_args()
    
    # Auto-build experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"exp_time_generalisation__{args.entity_disjoint}"
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'data_config.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Setup logging (experiment_name needed for log filename)
    log_dir = Path(config['logging']['log_dir'])
    logger = setup_logging(log_dir, config['logging']['level'], args.experiment_name)
    
    # Base directory
    base_dir = Path(__file__).parent.parent
    
    # Locate Stage 7 output directories
    try:
        train_input_dir = locate_stage7_dir(args.train_dataset, base_dir)
        logger.info(f"Train input directory: {train_input_dir}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    try:
        eval_input_dir = locate_stage7_dir(args.eval_dataset, base_dir)
        logger.info(f"Eval input directory: {eval_input_dir}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Output directory
    output_base_dir = base_dir / 'data_splits'
    
    # Create splits
    try:
        stats = create_splits(
            train_input_dir,
            eval_input_dir,
            output_base_dir,
            args.experiment_name,
            args.overwrite,
            logger,
            args.max_partitions,
            args.entity_disjoint
        )
        stats.train_dataset = args.train_dataset
        stats.eval_dataset = args.eval_dataset
        
        # Write reports
        write_split_reports(stats, output_base_dir, logger)
        
        logger.info("=" * 60)
        logger.info("SUCCESS: Splitting complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during splitting: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

