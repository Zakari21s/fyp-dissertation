"""
Audit raw CSV files in a dataset folder without loading everything into memory.

This script audits raw CSV quality, including:
- Row counts (streaming, chunked)
- Column schema and dtype inference
- Missingness analysis
- Numeric statistics
- Duplicate detection
- Schema drift detection
"""

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import yaml
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class FileAuditResult:
    """Results of auditing a single CSV file."""
    file_path: str
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    missingness: Dict[str, float] = field(default_factory=dict)
    numeric_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    duplicate_rate: Optional[float] = None
    exact_duplicates: Optional[int] = None
    error: Optional[str] = None
    processing_time_seconds: float = 0.0


@dataclass
class AuditSummary:
    """Summary statistics across all files."""
    total_files: int = 0
    total_rows: int = 0
    files_processed: int = 0
    files_failed: int = 0
    common_columns: List[str] = field(default_factory=list)
    top_missingness_columns: List[Tuple[str, float]] = field(default_factory=list)
    schema_drift_files: List[str] = field(default_factory=list)
    dtype_inconsistencies: Dict[str, Set[str]] = field(default_factory=dict)


class ReservoirSampler:
    """Reservoir sampling for approximate quantiles on large datasets."""
    
    def __init__(self, reservoir_size: int = 10000, seed: int = 42):
        self.reservoir_size = reservoir_size
        self.reservoir: Dict[str, List[float]] = defaultdict(list)
        self.count: Dict[str, int] = defaultdict(int)
        np.random.seed(seed)
    
    def update(self, column: str, values: pd.Series):
        """Update reservoir with new values."""
        numeric_values = pd.to_numeric(values, errors='coerce').dropna()
        if len(numeric_values) == 0:
            return
        
        for val in numeric_values:
            self.count[column] += 1
            n = self.count[column]
            
            if len(self.reservoir[column]) < self.reservoir_size:
                self.reservoir[column].append(float(val))
            else:
                # Replace with probability reservoir_size / n
                j = np.random.randint(0, n)
                if j < self.reservoir_size:
                    self.reservoir[column][j] = float(val)
    
    def get_quantiles(self, column: str, quantiles: List[float]) -> Dict[str, float]:
        """Get approximate quantiles from reservoir sample."""
        if column not in self.reservoir or len(self.reservoir[column]) == 0:
            return {}
        
        sample = np.array(self.reservoir[column])
        result = {}
        for q in quantiles:
            result[f"q{int(q*100)}"] = float(np.quantile(sample, q))
        return result


class CSVAuditor:
    """Audits CSV files using chunked processing."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str, logger: logging.Logger):
        self.config = config
        self.dataset_name = dataset_name
        self.logger = logger
        
        # Extract config values
        self.data_raw_dir = Path(config['data_raw_dir'])
        self.dataset_path = self.data_raw_dir / config['datasets'][dataset_name]['path']
        self.file_pattern = config['file_pattern']
        self.chunk_size = config['processing']['chunk_size']
        self.key_columns = config.get('key_columns', {})
        self.serial_col = self.key_columns.get('serial_number')
        self.date_col = self.key_columns.get('date')
        
        # Small file threshold for exact duplicate check (10MB)
        self.small_file_threshold = 10 * 1024 * 1024
        
        # Results storage
        self.file_results: List[FileAuditResult] = []
        self.schema_counter: Counter = Counter()
        self.column_dtypes: Dict[str, Set[str]] = defaultdict(set)
    
    def find_csv_files(self) -> List[Path]:
        """Find all CSV files matching the pattern."""
        pattern = self.file_pattern.replace('*', '')
        csv_files = sorted(self.dataset_path.glob(self.file_pattern))
        self.logger.info(f"Found {len(csv_files)} CSV files in {self.dataset_path}")
        return csv_files
    
    def audit_file(self, file_path: Path) -> FileAuditResult:
        """Audit a single CSV file."""
        start_time = datetime.now()
        result = FileAuditResult(file_path=str(file_path))
        
        try:
            self.logger.info(f"Auditing {file_path.name}")
            
            # Check file size for duplicate detection strategy
            file_size = file_path.stat().st_size
            use_exact_duplicates = file_size < self.small_file_threshold
            
            # Initialize accumulators
            row_count = 0
            columns_set: Optional[Set[str]] = None
            dtypes_dict: Dict[str, str] = {}
            missing_counts: Dict[str, int] = defaultdict(int)
            numeric_accumulators: Dict[str, Dict[str, float]] = defaultdict(lambda: {
                'min': float('inf'),
                'max': float('-inf'),
                'sum': 0.0,
                'count': 0
            })
            reservoir = ReservoirSampler(reservoir_size=10000)
            
            # For duplicate detection
            key_hash_set: Set[str] = set()
            total_key_pairs = 0
            exact_duplicate_rows = 0
            
            # Process file in chunks
            chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size, low_memory=False)
            
            for chunk_idx, chunk in enumerate(chunk_iter):
                row_count += len(chunk)
                
                # Capture columns and dtypes from first chunk
                if chunk_idx == 0:
                    columns_set = set(chunk.columns)
                    result.columns = list(chunk.columns)
                    # Infer dtypes (best effort)
                    for col in chunk.columns:
                        dtype_str = str(chunk[col].dtype)
                        dtypes_dict[col] = dtype_str
                        result.dtypes[col] = dtype_str
                
                # Missingness
                for col in chunk.columns:
                    missing_count = chunk[col].isna().sum()
                    missing_counts[col] += int(missing_count)
                
                # Numeric statistics
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    numeric_vals = pd.to_numeric(chunk[col], errors='coerce').dropna()
                    if len(numeric_vals) > 0:
                        acc = numeric_accumulators[col]
                        acc['min'] = min(acc['min'], float(numeric_vals.min()))
                        acc['max'] = max(acc['max'], float(numeric_vals.max()))
                        acc['sum'] += float(numeric_vals.sum())
                        acc['count'] += len(numeric_vals)
                        # Update reservoir for quantiles
                        reservoir.update(col, numeric_vals)
                
                # Hash-based approximate duplicate detection
                if self.serial_col and self.date_col:
                    if self.serial_col in chunk.columns and self.date_col in chunk.columns:
                        # Create key pairs
                        key_pairs = chunk[[self.serial_col, self.date_col]].dropna()
                        total_key_pairs += len(key_pairs)
                        
                        # Hash-based approximate duplicate detection
                        for _, row in key_pairs.iterrows():
                            key_str = f"{row[self.serial_col]}|{row[self.date_col]}"
                            key_hash = hashlib.md5(key_str.encode()).hexdigest()
                            key_hash_set.add(key_hash)
            
            # Exact duplicate check for small files (after chunking is complete)
            if use_exact_duplicates and self.serial_col and self.date_col:
                try:
                    df_full = pd.read_csv(file_path, low_memory=False)
                    if self.serial_col in df_full.columns and self.date_col in df_full.columns:
                        key_cols = df_full[[self.serial_col, self.date_col]].dropna()
                        duplicates = key_cols.duplicated().sum()
                        exact_duplicate_rows = int(duplicates)
                except Exception as e:
                    self.logger.warning(f"Could not check exact duplicates for {file_path.name}: {e}")
            
            # Calculate final statistics
            result.row_count = row_count
            
            # Missingness percentages
            for col, missing_count in missing_counts.items():
                result.missingness[col] = (missing_count / row_count * 100) if row_count > 0 else 0.0
            
            # Numeric statistics
            for col, acc in numeric_accumulators.items():
                if acc['count'] > 0:
                    mean_val = acc['sum'] / acc['count']
                    quantiles = reservoir.get_quantiles(col, [0.01, 0.50, 0.99])
                    result.numeric_stats[col] = {
                        'min': acc['min'] if acc['min'] != float('inf') else None,
                        'max': acc['max'] if acc['max'] != float('-inf') else None,
                        'mean': mean_val,
                        **quantiles
                    }
            
            # Duplicate rate
            if total_key_pairs > 0:
                # Approximate: hash collisions are rare, so unique hashes ≈ unique pairs
                unique_pairs = len(key_hash_set)
                duplicate_pairs = total_key_pairs - unique_pairs
                result.duplicate_rate = (duplicate_pairs / total_key_pairs * 100) if total_key_pairs > 0 else 0.0
                result.exact_duplicates = exact_duplicate_rows
            
            # Track schema
            schema_tuple = tuple(sorted(result.columns))
            self.schema_counter[schema_tuple] += 1
            
            # Track dtype inconsistencies
            for col, dtype in result.dtypes.items():
                self.column_dtypes[col].add(dtype)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            result.processing_time_seconds = elapsed
            self.logger.info(f"Completed {file_path.name}: {row_count} rows in {elapsed:.2f}s")
            
        except Exception as e:
            result.error = str(e)
            self.logger.error(f"Error auditing {file_path.name}: {e}", exc_info=True)
        
        return result
    
    def audit_all(self) -> Tuple[List[FileAuditResult], AuditSummary]:
        """Audit all CSV files in the dataset."""
        csv_files = self.find_csv_files()
        
        if not csv_files:
            self.logger.warning(f"No CSV files found in {self.dataset_path}")
            return [], AuditSummary()
        
        # Process each file
        for file_path in csv_files:
            result = self.audit_file(file_path)
            self.file_results.append(result)
        
        # Generate summary
        summary = self._generate_summary()
        
        return self.file_results, summary
    
    def _generate_summary(self) -> AuditSummary:
        """Generate summary statistics."""
        summary = AuditSummary()
        summary.total_files = len(self.file_results)
        
        # Count rows and files
        total_rows = 0
        files_processed = 0
        files_failed = 0
        
        for result in self.file_results:
            if result.error:
                files_failed += 1
            else:
                files_processed += 1
                total_rows += result.row_count
        
        summary.total_rows = total_rows
        summary.files_processed = files_processed
        summary.files_failed = files_failed
        
        # Find most common schema
        if self.schema_counter:
            most_common_schema = self.schema_counter.most_common(1)[0][0]
            summary.common_columns = list(most_common_schema)
            
            # Find files with schema drift
            for result in self.file_results:
                if result.error:
                    continue
                file_schema = tuple(sorted(result.columns))
                if file_schema != most_common_schema:
                    summary.schema_drift_files.append(result.file_path)
        
        # Top missingness columns (aggregate across all files)
        missingness_agg: Dict[str, List[float]] = defaultdict(list)
        for result in self.file_results:
            if result.error:
                continue
            for col, pct in result.missingness.items():
                missingness_agg[col].append(pct)
        
        # Average missingness per column
        avg_missingness = {
            col: np.mean(pcts) if pcts else 0.0
            for col, pcts in missingness_agg.items()
        }
        summary.top_missingness_columns = sorted(
            avg_missingness.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10
        
        # Dtype inconsistencies
        summary.dtype_inconsistencies = {
            col: dtypes for col, dtypes in self.column_dtypes.items()
            if len(dtypes) > 1
        }
        
        return summary


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path, log_level: str, dataset_name: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"audit_{dataset_name}.log"
    
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


def write_json_report(results: List[FileAuditResult], summary: AuditSummary, 
                     output_path: Path, logger: logging.Logger):
    """Write JSON audit report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert summary to dict, handling sets
    summary_dict = asdict(summary)
    # Convert sets to lists for JSON serialization
    summary_dict['dtype_inconsistencies'] = {
        col: list(dtypes) for col, dtypes in summary.dtype_inconsistencies.items()
    }
    
    report = {
        'dataset': results[0].file_path.split('/')[-2] if results else 'unknown',
        'audit_timestamp': datetime.now().isoformat(),
        'summary': summary_dict,
        'files': [asdict(result) for result in results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"JSON report written to {output_path}")


def write_schema_drift_report(summary: AuditSummary, output_path: Path, 
                             logger: logging.Logger):
    """Write schema drift markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Schema Drift Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        # Most common schema
        f.write("## Most Common Schema\n\n")
        f.write(f"Columns ({len(summary.common_columns)}):\n")
        for col in summary.common_columns:
            f.write(f"- `{col}`\n")
        f.write("\n")
        
        # Schema drift files
        if summary.schema_drift_files:
            f.write(f"## Files with Schema Drift ({len(summary.schema_drift_files)})\n\n")
            f.write("The following files have missing or extra columns compared to the most common schema:\n\n")
            for file_path in summary.schema_drift_files:
                f.write(f"- `{file_path}`\n")
            f.write("\n")
        else:
            f.write("## Schema Consistency\n\n")
            f.write("✓ All files have consistent schemas.\n\n")
        
        # Dtype inconsistencies
        if summary.dtype_inconsistencies:
            f.write("## Data Type Inconsistencies\n\n")
            f.write("The following columns have inconsistent data types across files:\n\n")
            for col, dtypes in summary.dtype_inconsistencies.items():
                f.write(f"### `{col}`\n\n")
                f.write(f"Types found: {', '.join(sorted(dtypes))}\n\n")
        else:
            f.write("## Data Type Consistency\n\n")
            f.write("✓ All columns have consistent data types across files.\n\n")
    
    logger.info(f"Schema drift report written to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Audit raw CSV files in a dataset')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., smartlog2018ssd)')
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / 'configs' / 'data_config.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Validate dataset exists
    if args.dataset not in config['datasets']:
        print(f"Error: Dataset '{args.dataset}' not found in config")
        print(f"Available datasets: {list(config['datasets'].keys())}")
        sys.exit(1)
    
    log_dir = Path(config['logging']['log_dir'])
    logger = setup_logging(log_dir, config['logging']['level'], args.dataset)
    
    logger.info(f"Starting audit for dataset: {args.dataset}")
    
    # Create auditor and run audit
    auditor = CSVAuditor(config, args.dataset, logger)
    results, summary = auditor.audit_all()
    
    # Write outputs
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = reports_dir / f"audit_{args.dataset}.json"
    md_path = reports_dir / f"schema_drift_{args.dataset}.md"
    
    write_json_report(results, summary, json_path, logger)
    write_schema_drift_report(summary, md_path, logger)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files: {summary.total_files}")
    logger.info(f"Files processed: {summary.files_processed}")
    logger.info(f"Files failed: {summary.files_failed}")
    logger.info(f"Total rows: {summary.total_rows:,}")
    logger.info(f"Common columns: {len(summary.common_columns)}")
    logger.info(f"Schema drift files: {len(summary.schema_drift_files)}")
    logger.info(f"Dtype inconsistencies: {len(summary.dtype_inconsistencies)}")
    logger.info("=" * 60)
    logger.info(f"Reports written to:")
    logger.info(f"  - {json_path}")
    logger.info(f"  - {md_path}")
    logger.info(f"  - {log_dir / f'audit_{args.dataset}.log'}")


if __name__ == '__main__':
    main()

