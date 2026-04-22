"""
Audit failure labels CSV file.

This script performs comprehensive data quality audits on the failure labels file including:
- Row counts
- Column schema and dtype inference
- Missingness analysis
- Duplicate detection on disk ID
- Datetime parsing with multiple format attempts
- Failure date statistics
- Unique disk counts
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


@dataclass
class LabelAuditResult:
    """Results of auditing the failure labels file."""
    file_path: str
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    missingness: Dict[str, float] = field(default_factory=dict)
    duplicate_disk_ids: Optional[int] = None
    duplicate_rate: Optional[float] = None
    datetime_parse_success_rate: Optional[float] = None
    datetime_column: Optional[str] = None
    datetime_formats_tried: List[str] = field(default_factory=list)
    datetime_format_used: Optional[str] = None
    inferred_format: Optional[str] = None
    min_failure_date: Optional[str] = None
    max_failure_date: Optional[str] = None
    unparsed_examples: List[str] = field(default_factory=list)
    unique_disks_with_failures: Optional[int] = None
    disk_id_column: Optional[str] = None
    error: Optional[str] = None
    processing_time_seconds: float = 0.0


def parse_datetime_column(df: pd.DataFrame, column: str, logger: logging.Logger) -> Tuple[pd.Series, str, str, List[str], List[str]]:
    """
    Attempt to parse datetime column using specific order of formats.
    
    Returns:
        Tuple of (parsed_series, format_used, inferred_format, formats_tried, unparsed_examples)
    """
    if column not in df.columns:
        return pd.Series(dtype='datetime64[ns]'), None, None, [], []
    
    formats_tried = []
    parsed_series = None
    format_used = None
    inferred_format = None
    
    # Step 1: Try pandas with UTC and infer_datetime_format
    try:
        parsed = pd.to_datetime(df[column], errors='coerce', utc=True, infer_datetime_format=True)
        success_count = parsed.notna().sum()
        success_rate = success_count / len(df) if len(df) > 0 else 0.0
        formats_tried.append('pandas_infer_utc')
        
        if success_rate >= 0.95:
            # Good enough, return immediately
            inferred_format = 'pandas_infer_utc'
            unparsed = df.loc[parsed.isna(), column].dropna().unique()[:20].tolist()
            return parsed, 'pandas_infer_utc', inferred_format, formats_tried, unparsed
        
        # Store as best so far if better than nothing
        if success_count > 0:
            parsed_series = parsed
            format_used = 'pandas_infer_utc'
            inferred_format = 'pandas_infer_utc'
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
        best_format = format_used
        best_success = parsed_series.notna().sum() if parsed_series is not None else 0
        
        for fmt in specific_formats:
            formats_tried.append(fmt)
            try:
                parsed = pd.to_datetime(df[column], format=fmt, errors='coerce')
                success_count = parsed.notna().sum()
                
                if success_count > best_success:
                    best_parsed = parsed
                    best_format = fmt
                    best_success = success_count
                    inferred_format = fmt
                    
                    # If we get 95%+ success, we can stop
                    if success_count / len(df) >= 0.95:
                        break
            except Exception:
                continue
        
        if best_parsed is not None and best_success > (parsed_series.notna().sum() if parsed_series is not None else 0):
            parsed_series = best_parsed
            format_used = best_format
    
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
                        formats_tried.append('unix_seconds')
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
                                format_used = 'unix_seconds'
                                inferred_format = 'unix_seconds (10 digits)'
                    
                    # Try milliseconds (13 digits)
                    if (digit_lengths == 13).any():
                        formats_tried.append('unix_milliseconds')
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
                                format_used = 'unix_milliseconds'
                                inferred_format = 'unix_milliseconds (13 digits)'
        except Exception as e:
            logger.debug(f"Unix timestamp parsing failed: {e}")
    
    # Collect unparsed examples
    unparsed_examples = []
    if parsed_series is not None:
        unparsed_mask = parsed_series.isna() & df[column].notna()
        unparsed_values = df.loc[unparsed_mask, column].unique()[:20]
        unparsed_examples = [str(val) for val in unparsed_values]
    elif column in df.columns:
        # If nothing parsed, show some examples
        sample_values = df[column].dropna().unique()[:20]
        unparsed_examples = [str(val) for val in sample_values]
    
    # Return best result found, or empty series
    if parsed_series is not None:
        return parsed_series, format_used, inferred_format, formats_tried, unparsed_examples
    
    return pd.Series(dtype='datetime64[ns]'), None, None, formats_tried, unparsed_examples


def detect_datetime_column(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    """Detect which column contains datetime data. Prioritizes 'failure_time'."""
    # First, check if 'failure_time' column exists
    if 'failure_time' in df.columns:
        logger.info("Found 'failure_time' column, using it as datetime candidate")
        return 'failure_time'
    
    # Common datetime column names
    datetime_keywords = ['time', 'date', 'timestamp', 'failure_date', 
                        'fail_time', 'fail_date', 'event_time', 'event_date']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in datetime_keywords):
            # Check if it looks like datetime
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                # Try quick parse
                try:
                    pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                    return col
                except Exception:
                    pass
    
    # If no obvious match, try all columns
    for col in df.columns:
        sample = df[col].dropna().head(100)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                if parsed.notna().sum() / len(sample) > 0.5:  # >50% parseable
                    return col
            except Exception:
                continue
    
    return None


def detect_disk_id_column(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    """Detect which column contains disk ID."""
    # Common disk ID column names
    disk_id_keywords = ['disk_id', 'diskid', 'serial', 'serial_number', 'serialnumber',
                       'device_id', 'deviceid', 'id']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in disk_id_keywords):
            return col
    
    # If no match, return None
    return None


def audit_labels_file(file_path: Path, logger: logging.Logger) -> LabelAuditResult:
    """Audit the failure labels CSV file."""
    start_time = datetime.now()
    result = LabelAuditResult(file_path=str(file_path))
    
    try:
        logger.info(f"Auditing labels file: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        
        # Basic statistics
        result.row_count = len(df)
        result.columns = list(df.columns)
        
        # Infer dtypes
        for col in df.columns:
            result.dtypes[col] = str(df[col].dtype)
        
        # Missingness
        for col in df.columns:
            missing_count = df[col].isna().sum()
            result.missingness[col] = (missing_count / result.row_count * 100) if result.row_count > 0 else 0.0
        
        # Detect and check duplicates on disk ID column
        disk_id_col = detect_disk_id_column(df, logger)
        result.disk_id_column = disk_id_col
        
        if disk_id_col:
            logger.info(f"Detected disk ID column: {disk_id_col}")
            duplicates = df[disk_id_col].duplicated().sum()
            result.duplicate_disk_ids = int(duplicates)
            result.duplicate_rate = (duplicates / result.row_count * 100) if result.row_count > 0 else 0.0
            
            # Count unique disks
            unique_disks = df[disk_id_col].nunique()
            result.unique_disks_with_failures = int(unique_disks)
        else:
            logger.warning("Could not detect disk ID column")
        
        # Detect and parse datetime column
        datetime_col = detect_datetime_column(df, logger)
        result.datetime_column = datetime_col
        
        if datetime_col:
            logger.info(f"Detected datetime column: {datetime_col}")
            parsed_dt, format_used, inferred_format, formats_tried, unparsed_examples = parse_datetime_column(df, datetime_col, logger)
            result.datetime_formats_tried = formats_tried
            result.datetime_format_used = format_used
            result.inferred_format = inferred_format
            result.unparsed_examples = unparsed_examples
            
            if format_used:
                # Calculate success rate
                success_count = parsed_dt.notna().sum()
                result.datetime_parse_success_rate = (success_count / result.row_count * 100) if result.row_count > 0 else 0.0
                
                # Min/max dates
                valid_dates = parsed_dt.dropna()
                if len(valid_dates) > 0:
                    result.min_failure_date = str(valid_dates.min())
                    result.max_failure_date = str(valid_dates.max())
                
                logger.info(f"Parsed datetime with format '{format_used}': {result.datetime_parse_success_rate:.2f}% success rate")
                if unparsed_examples:
                    logger.info(f"Found {len(unparsed_examples)} unparsed value examples")
            else:
                logger.warning(f"Could not parse datetime column '{datetime_col}' with any format")
                result.datetime_parse_success_rate = 0.0
        else:
            logger.warning("Could not detect datetime column")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        result.processing_time_seconds = elapsed
        logger.info(f"Completed audit in {elapsed:.2f}s")
        
    except Exception as e:
        result.error = str(e)
        logger.error(f"Error auditing labels file: {e}", exc_info=True)
    
    return result


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path, log_level: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "audit_failure_labels.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def write_json_report(result: LabelAuditResult, output_path: Path, logger: logging.Logger):
    """Write JSON audit report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'audit_timestamp': datetime.now().isoformat(),
        'result': asdict(result)
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"JSON report written to {output_path}")


def write_markdown_report(result: LabelAuditResult, output_path: Path, logger: logging.Logger):
    """Write markdown audit report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Failure Labels Audit Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**File:** `{result.file_path}`\n\n")
        
        if result.error:
            f.write(f"## Error\n\n")
            f.write(f"❌ **Error occurred:** {result.error}\n\n")
            return
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Total rows:** {result.row_count:,}\n")
        f.write(f"- **Columns:** {len(result.columns)}\n")
        f.write(f"- **Processing time:** {result.processing_time_seconds:.2f}s\n\n")
        
        # Columns and dtypes
        f.write("## Schema\n\n")
        f.write("| Column | Data Type |\n")
        f.write("|--------|-----------|\n")
        for col in result.columns:
            dtype = result.dtypes.get(col, 'unknown')
            f.write(f"| `{col}` | `{dtype}` |\n")
        f.write("\n")
        
        # Missingness
        f.write("## Missingness\n\n")
        if result.missingness:
            f.write("| Column | Missing % |\n")
            f.write("|--------|----------|\n")
            sorted_missing = sorted(result.missingness.items(), key=lambda x: x[1], reverse=True)
            for col, pct in sorted_missing:
                f.write(f"| `{col}` | {pct:.2f}% |\n")
            f.write("\n")
        else:
            f.write("No missing values detected.\n\n")
        
        # Duplicates
        if result.disk_id_column:
            f.write("## Duplicate Detection\n\n")
            f.write(f"**Disk ID column:** `{result.disk_id_column}`\n\n")
            if result.duplicate_disk_ids is not None:
                f.write(f"- **Duplicate rows:** {result.duplicate_disk_ids:,}\n")
                f.write(f"- **Duplicate rate:** {result.duplicate_rate:.2f}%\n")
            if result.unique_disks_with_failures is not None:
                f.write(f"- **Unique disks with failures:** {result.unique_disks_with_failures:,}\n")
            f.write("\n")
        
        # Datetime parsing
        if result.datetime_column:
            f.write("## Datetime Parsing\n\n")
            f.write(f"**Datetime column:** `{result.datetime_column}`\n\n")
            
            if result.datetime_format_used:
                f.write(f"- **Format used:** `{result.datetime_format_used}`\n")
                f.write(f"- **Inferred format:** `{result.inferred_format}`\n")
                f.write(f"- **Parse success rate:** {result.datetime_parse_success_rate:.2f}%\n")
                f.write(f"- **Formats tried:** {len(result.datetime_formats_tried)}\n")
                if result.datetime_formats_tried:
                    f.write("  - " + "\n  - ".join(result.datetime_formats_tried) + "\n")
                f.write("\n")
                
                if result.min_failure_date and result.max_failure_date:
                    f.write("### Parsed Failure Time Range\n\n")
                    f.write(f"- **Min failure date:** {result.min_failure_date}\n")
                    f.write(f"- **Max failure date:** {result.max_failure_date}\n")
                    f.write("\n")
                
                if result.unparsed_examples:
                    f.write("### Unparsed Values\n\n")
                    f.write(f"Found {len(result.unparsed_examples)} example(s) of values that could not be parsed:\n\n")
                    for example in result.unparsed_examples:
                        f.write(f"- `{example}`\n")
                    f.write("\n")
            else:
                f.write("❌ **Could not parse datetime column with any format**\n\n")
        else:
            f.write("## Datetime Parsing\n\n")
            f.write("❌ **Could not detect datetime column**\n\n")
        
        # Data quality summary
        f.write("## Data Quality Summary\n\n")
        issues = []
        if result.datetime_parse_success_rate is not None and result.datetime_parse_success_rate < 100:
            issues.append(f"Datetime parse success rate is {result.datetime_parse_success_rate:.2f}% (not 100%)")
        if result.duplicate_rate and result.duplicate_rate > 0:
            issues.append(f"Duplicate rate is {result.duplicate_rate:.2f}%")
        if any(pct > 0 for pct in result.missingness.values()):
            high_missing = [(col, pct) for col, pct in result.missingness.items() if pct > 5]
            if high_missing:
                issues.append(f"High missingness (>5%): {', '.join([f'{col} ({pct:.1f}%)' for col, pct in high_missing])}")
        
        if issues:
            f.write("⚠️ **Issues detected:**\n\n")
            for issue in issues:
                f.write(f"- {issue}\n")
        else:
            f.write("✓ **No major issues detected**\n")
        f.write("\n")
    
    logger.info(f"Markdown report written to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Audit failure labels CSV file')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'data_config.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Validate labels config exists
    if 'labels' not in config or 'failure_label_file' not in config['labels']:
        print("Error: 'labels.failure_label_file' not found in config")
        sys.exit(1)
    
    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    logger = setup_logging(log_dir, config['logging']['level'])
    
    logger.info("Starting failure labels audit")
    
    # Locate labels file
    data_raw_dir = Path(config['data_raw_dir'])
    labels_file = data_raw_dir / config['labels']['failure_label_file']
    
    if not labels_file.exists():
        logger.error(f"Labels file not found at {labels_file}")
        sys.exit(1)
    
    logger.info(f"Labels file: {labels_file}")
    
    # Audit the file
    result = audit_labels_file(labels_file, logger)
    
    # Write outputs
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = reports_dir / "audit_failure_labels.json"
    md_path = reports_dir / "audit_failure_labels.md"
    
    write_json_report(result, json_path, logger)
    write_markdown_report(result, md_path, logger)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 60)
    if result.error:
        logger.error(f"Error: {result.error}")
    else:
        logger.info(f"Total rows: {result.row_count:,}")
        logger.info(f"Columns: {len(result.columns)}")
        if result.disk_id_column:
            logger.info(f"Disk ID column: {result.disk_id_column}")
            logger.info(f"Unique disks with failures: {result.unique_disks_with_failures:,}")
            logger.info(f"Duplicate rate: {result.duplicate_rate:.2f}%")
        if result.datetime_column:
            logger.info(f"Datetime column: {result.datetime_column}")
            logger.info(f"Parse success rate: {result.datetime_parse_success_rate:.2f}%")
            if result.min_failure_date and result.max_failure_date:
                logger.info(f"Failure date range: {result.min_failure_date} to {result.max_failure_date}")
    logger.info("=" * 60)
    logger.info(f"Reports written to:")
    logger.info(f"  - {json_path}")
    logger.info(f"  - {md_path}")
    logger.info(f"  - {log_dir / 'audit_failure_labels.log'}")


if __name__ == '__main__':
    main()

