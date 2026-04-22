"""
Prepare and deduplicate failure labels.

This script:
- Reads the failure labels CSV file
- Parses failure_time using the same logic as audit_labels.py
- Resolves duplicates by taking the earliest failure_time per disk_id
- Outputs cleaned data to parquet format
- Generates a summary report
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import yaml
import pandas as pd
import numpy as np
from datetime import datetime


def parse_datetime_column(df: pd.DataFrame, column: str, logger: logging.Logger) -> pd.Series:
    """
    Parse datetime column using the same logic as audit_labels.py.
    
    Returns:
        Parsed datetime series
    """
    if column not in df.columns:
        return pd.Series(dtype='datetime64[ns]')
    
    parsed_series = None
    
    # Step 1: Try pandas with UTC and infer_datetime_format
    try:
        parsed = pd.to_datetime(df[column], errors='coerce', utc=True, infer_datetime_format=True)
        success_count = parsed.notna().sum()
        success_rate = success_count / len(df) if len(df) > 0 else 0.0
        
        if success_rate >= 0.95:
            logger.info(f"Parsed {column} with pandas_infer_utc: {success_rate:.2%} success rate")
            return parsed
        
        # Store as best so far if better than nothing
        if success_count > 0:
            parsed_series = parsed
    except Exception as e:
        logger.debug(f"pandas infer_datetime_format with UTC failed: {e}")
    
    # Step 2: If parse rate < 0.95, try specific formats
    if parsed_series is None or parsed_series.notna().sum() / len(df) < 0.95:
        specific_formats = [
            '%Y%m%d',                    # 20190322
            '%Y-%m-%d',                 # 2019-03-22
            '%Y/%m/%d',                 # 2019/03/22
            '%Y%m%d%H%M%S',             # 20190322102438
            '%Y-%m-%d %H:%M:%S',        # 2019-03-22 10:24:38
        ]
        
        best_parsed = parsed_series
        best_success = parsed_series.notna().sum() if parsed_series is not None else 0
        
        for fmt in specific_formats:
            try:
                parsed = pd.to_datetime(df[column], format=fmt, errors='coerce')
                success_count = parsed.notna().sum()
                
                if success_count > best_success:
                    best_parsed = parsed
                    best_success = success_count
                    
                    # If we get 95%+ success, we can stop
                    if success_count / len(df) >= 0.95:
                        logger.info(f"Parsed {column} with format {fmt}: {success_count/len(df):.2%} success rate")
                        return parsed
            except Exception:
                continue
        
        if best_parsed is not None and best_success > (parsed_series.notna().sum() if parsed_series is not None else 0):
            parsed_series = best_parsed
    
    # Step 3: If still low success rate, try Unix timestamps
    if parsed_series is None or parsed_series.notna().sum() / len(df) < 0.95:
        try:
            numeric_vals = pd.to_numeric(df[column], errors='coerce')
            if numeric_vals.notna().sum() > 0:
                # Check digit length on sample to determine seconds vs milliseconds
                sample_vals = numeric_vals.dropna().head(100)
                if len(sample_vals) > 0:
                    # Convert to string to check digit length (memory efficient)
                    sample_str = sample_vals.astype(int).astype(str)
                    digit_lengths = sample_str.str.len()
                    
                    # Try seconds (10 digits) - Unix timestamp range: 2000-01-01 to 2100-01-01
                    if (digit_lengths == 10).any():
                        min_ts = 946684800  # 2000-01-01
                        max_ts = 4102444800  # 2100-01-01
                        
                        # Create mask for valid range
                        valid_mask = (numeric_vals >= min_ts) & (numeric_vals <= max_ts) & numeric_vals.notna()
                        if valid_mask.sum() > 0:
                            # Parse only valid values, then map to full series
                            valid_values = numeric_vals[valid_mask]
                            parsed_valid = pd.to_datetime(valid_values, unit='s', errors='coerce')
                            
                            # Create full series aligned with original index
                            full_parsed = pd.Series(index=df.index, dtype='datetime64[ns]')
                            full_parsed.loc[valid_mask] = parsed_valid
                            
                            success_count = full_parsed.notna().sum()
                            if success_count > (parsed_series.notna().sum() if parsed_series is not None else 0):
                                parsed_series = full_parsed
                                logger.info(f"Parsed {column} as unix_seconds: {success_count/len(df):.2%} success rate")
                    
                    # Try milliseconds (13 digits)
                    if (digit_lengths == 13).any():
                        min_ts_ms = 946684800000  # 2000-01-01
                        max_ts_ms = 4102444800000  # 2100-01-01
                        
                        # Create mask for valid range
                        valid_mask = (numeric_vals >= min_ts_ms) & (numeric_vals <= max_ts_ms) & numeric_vals.notna()
                        if valid_mask.sum() > 0:
                            # Parse only valid values, then map to full series
                            valid_values = numeric_vals[valid_mask]
                            parsed_valid = pd.to_datetime(valid_values, unit='ms', errors='coerce')
                            
                            # Create full series aligned with original index
                            full_parsed = pd.Series(index=df.index, dtype='datetime64[ns]')
                            full_parsed.loc[valid_mask] = parsed_valid
                            
                            success_count = full_parsed.notna().sum()
                            if success_count > (parsed_series.notna().sum() if parsed_series is not None else 0):
                                parsed_series = full_parsed
                                logger.info(f"Parsed {column} as unix_milliseconds: {success_count/len(df):.2%} success rate")
        except Exception as e:
            logger.debug(f"Unix timestamp parsing failed: {e}")
    
    # Return best result found, or empty series
    if parsed_series is not None:
        return parsed_series
    
    logger.warning(f"Could not parse {column} with any format")
    return pd.Series(dtype='datetime64[ns]')


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def prepare_failure_labels(
    input_file: Path,
    output_file: Path,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, dict]:
    """
    Prepare and deduplicate failure labels.
    
    Returns:
        Tuple of (cleaned_dataframe, summary_stats)
    """
    logger.info(f"Reading failure labels from {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    
    rows_before = len(df)
    logger.info(f"Loaded {rows_before:,} rows")
    
    # Detect disk_id column
    disk_id_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'disk_id' in col_lower or col_lower == 'disk_id':
            disk_id_col = col
            break
    
    if disk_id_col is None:
        raise ValueError("Could not find disk_id column in the labels file")
    
    logger.info(f"Using disk_id column: {disk_id_col}")
    
    # Parse failure_time column
    failure_time_col = 'failure_time'
    if failure_time_col not in df.columns:
        raise ValueError(f"Column '{failure_time_col}' not found in the labels file")
    
    logger.info(f"Parsing {failure_time_col} column")
    df['failure_time_parsed'] = parse_datetime_column(df, failure_time_col, logger)
    
    # Check parse success
    parse_success = df['failure_time_parsed'].notna().sum()
    parse_rate = parse_success / len(df) if len(df) > 0 else 0.0
    logger.info(f"Parsed {parse_success:,} / {len(df):,} failure times ({parse_rate:.2%})")
    
    if parse_rate < 0.95:
        logger.warning(f"Low parse success rate: {parse_rate:.2%}")
    
    # Identify duplicates
    duplicate_mask = df.duplicated(subset=[disk_id_col], keep=False)
    disks_with_duplicates = df[duplicate_mask][disk_id_col].nunique()
    total_duplicate_rows = duplicate_mask.sum()
    
    logger.info(f"Found {disks_with_duplicates:,} disks with multiple records ({total_duplicate_rows:,} duplicate rows)")
    
    # Identify conflicting failure_times (same disk_id but different failure_time)
    if total_duplicate_rows > 0:
        # Group by disk_id and check for different failure_times
        # Only count if there are multiple unique failure_times (not just duplicate rows with same time)
        duplicate_df = df[duplicate_mask].copy()
        grouped = duplicate_df.groupby(disk_id_col)['failure_time_parsed']
        conflicting_disks = 0
        for disk_id, group in grouped:
            # Count unique non-null failure times
            unique_times = group.dropna().nunique()
            if unique_times > 1:
                conflicting_disks += 1
        
        logger.info(f"Found {conflicting_disks:,} disks with conflicting failure_times")
    else:
        conflicting_disks = 0
    
    # Deduplicate: take earliest failure_time per disk_id
    logger.info("Deduplicating by taking earliest failure_time per disk_id")
    
    # Sort by disk_id and failure_time_parsed (nulls last)
    df_sorted = df.sort_values(
        by=[disk_id_col, 'failure_time_parsed'],
        na_position='last'
    )
    
    # Keep first occurrence (earliest failure_time) per disk_id
    df_dedup = df_sorted.drop_duplicates(subset=[disk_id_col], keep='first')
    
    rows_after = len(df_dedup)
    rows_removed = rows_before - rows_after
    
    logger.info(f"Deduplication complete: {rows_before:,} -> {rows_after:,} rows ({rows_removed:,} removed)")
    
    # Prepare output dataframe
    # Keep original columns plus parsed failure_time
    output_df = df_dedup.copy()
    
    # Replace original failure_time with parsed version (if we want to keep both, we can rename)
    # For now, keep both: failure_time (original) and failure_time_parsed
    # Optionally, we could replace failure_time with parsed version
    # output_df['failure_time'] = output_df['failure_time_parsed']
    
    # Calculate summary statistics
    summary = {
        'rows_before': rows_before,
        'rows_after': rows_after,
        'rows_removed': rows_removed,
        'disks_with_multiple_records': int(disks_with_duplicates),
        'disks_with_conflicting_failure_time': int(conflicting_disks),
        'parse_success_rate': float(parse_rate),
        'parse_success_count': int(parse_success),
        'total_disks': int(df_dedup[disk_id_col].nunique()),
    }
    
    # Min/max failure dates
    valid_dates = df_dedup['failure_time_parsed'].dropna()
    if len(valid_dates) > 0:
        summary['min_failure_date'] = str(valid_dates.min())
        summary['max_failure_date'] = str(valid_dates.max())
    else:
        summary['min_failure_date'] = None
        summary['max_failure_date'] = None
    
    return output_df, summary


def write_summary_report(summary: dict, output_path: Path, logger: logging.Logger):
    """Write summary markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Failure Labels Deduplication Summary\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Rows before deduplication:** {summary['rows_before']:,}\n")
        f.write(f"- **Rows after deduplication:** {summary['rows_after']:,}\n")
        f.write(f"- **Rows removed:** {summary['rows_removed']:,}\n")
        f.write(f"- **Total unique disks:** {summary['total_disks']:,}\n")
        f.write("\n")
        
        f.write("## Duplicate Analysis\n\n")
        f.write(f"- **Disks with multiple records:** {summary['disks_with_multiple_records']:,}\n")
        f.write(f"- **Disks with conflicting failure_time:** {summary['disks_with_conflicting_failure_time']:,}\n")
        f.write("\n")
        f.write("**Note:** Duplicates were resolved by keeping the record with the earliest `failure_time` per `disk_id`.\n")
        f.write("\n")
        
        f.write("## Datetime Parsing\n\n")
        f.write(f"- **Parse success rate:** {summary['parse_success_rate']:.2%}\n")
        f.write(f"- **Successfully parsed:** {summary['parse_success_count']:,} / {summary['rows_before']:,}\n")
        f.write("\n")
        
        if summary['min_failure_date'] and summary['max_failure_date']:
            f.write("## Failure Date Range\n\n")
            f.write(f"- **Min failure date:** {summary['min_failure_date']}\n")
            f.write(f"- **Max failure date:** {summary['max_failure_date']}\n")
            f.write("\n")
        
        f.write("## Output\n\n")
        f.write("Cleaned data has been saved to `data_interim/labels/failure_labels_dedup.parquet`\n")
        f.write("\n")
        f.write("The output includes:\n")
        f.write("- All original columns\n")
        f.write("- `failure_time_parsed`: Parsed datetime version of `failure_time`\n")
        f.write("- One record per `disk_id` (earliest `failure_time` kept)\n")
    
    logger.info(f"Summary report written to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Prepare and deduplicate failure labels')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'data_config.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Setup logging
    log_level = config.get('logging', {}).get('level', 'INFO')
    logger = setup_logging(log_level)
    
    logger.info("Starting failure labels preparation")
    
    # Locate input file
    data_raw_dir = Path(config['data_raw_dir'])
    labels_file = data_raw_dir / config['labels']['failure_label_file']
    
    if not labels_file.exists():
        logger.error(f"Labels file not found at {labels_file}")
        sys.exit(1)
    
    # Prepare output path
    output_dir = Path('data_interim') / 'labels'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'failure_labels_dedup.parquet'
    
    # Process the file
    try:
        df_cleaned, summary = prepare_failure_labels(labels_file, output_file, logger)
        
        # Save to parquet
        logger.info(f"Saving cleaned data to {output_file}")
        df_cleaned.to_parquet(output_file, index=False, engine='pyarrow')
        logger.info(f"Saved {len(df_cleaned):,} rows to {output_file}")
        
        # Write summary report
        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_path = reports_dir / 'failure_labels_dedup_summary.md'
        write_summary_report(summary, summary_path, logger)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PREPARATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Rows before: {summary['rows_before']:,}")
        logger.info(f"Rows after: {summary['rows_after']:,}")
        logger.info(f"Rows removed: {summary['rows_removed']:,}")
        logger.info(f"Disks with multiple records: {summary['disks_with_multiple_records']:,}")
        logger.info(f"Disks with conflicting failure_time: {summary['disks_with_conflicting_failure_time']:,}")
        if summary['min_failure_date'] and summary['max_failure_date']:
            logger.info(f"Failure date range: {summary['min_failure_date']} to {summary['max_failure_date']}")
        logger.info("=" * 60)
        logger.info(f"Output files:")
        logger.info(f"  - {output_file}")
        logger.info(f"  - {summary_path}")
        
    except Exception as e:
        logger.error(f"Error preparing failure labels: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

