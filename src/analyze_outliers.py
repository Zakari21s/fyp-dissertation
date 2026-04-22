"""
Stage 5: Outlier Analysis for SMART Features (Analysis Only - No Data Modification).

This module analyzes extreme values in SMART features to understand their distribution
and support decisions on outlier handling. This is a read-only analysis stage.

Note: this stage is read-only and does not change data.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow.parquet as pq


@dataclass
class FeatureStats:
    """Statistics for a single SMART feature."""
    feature_name: str = ""
    count: int = 0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    sum_val: float = 0.0
    sum_sq: float = 0.0
    zero_count: int = 0
    negative_count: int = 0
    extreme_count: int = 0
    percentiles: Dict[float, float] = field(default_factory=dict)
    # For approximate quantiles using reservoir sampling
    sample_values: List[float] = field(default_factory=list)
    max_sample_size: int = 10000


@dataclass
class OutlierAnalysisStats:
    """Overall statistics for outlier analysis."""
    dataset: str = ""
    total_rows: int = 0
    total_partitions: int = 0
    partitions_processed: int = 0
    partitions_failed: int = 0
    smart_features_analyzed: int = 0
    features_with_negatives: int = 0
    features_with_extreme_skew: int = 0
    features_with_heavy_tails: int = 0
    feature_stats: Dict[str, FeatureStats] = field(default_factory=dict)
    processing_time_seconds: float = 0.0


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path, log_level: str, dataset_name: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"clean_stage5_{dataset_name}.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    root_logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w')
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


def update_reservoir_sample(sample: List[float], new_value: float, max_size: int, count: int, rng=None):
    """Update reservoir sample with new value using reservoir sampling algorithm."""
    if rng is None:
        import random
        rng = random.Random()
    
    if len(sample) < max_size:
        sample.append(new_value)
    else:
        # Reservoir sampling: replace with probability max_size / count
        j = rng.randint(0, count - 1)
        if j < max_size:
            sample[j] = new_value


def analyze_feature_chunk(
    df: pd.DataFrame,
    feature_name: str,
    stats: FeatureStats,
    percentiles: List[float],
    rng,
    batch_size: int = 250000
):
    """Analyze a chunk of data for a single feature, updating statistics incrementally."""
    if feature_name not in df.columns:
        return
    
    series = df[feature_name]
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return
    
    # Update counts
    # Update counts
    stats.count += len(non_null)
    
    # Update min/max
    # Update min/max
    chunk_min = non_null.min()
    chunk_max = non_null.max()
    if stats.min_val is None or chunk_min < stats.min_val:
        stats.min_val = float(chunk_min)
    if stats.max_val is None or chunk_max > stats.max_val:
        stats.max_val = float(chunk_max)
    
    # Update sum and sum of squares for mean/std
    # Update sum and sum of squares for mean/std
    stats.sum_val += float(non_null.sum())
    stats.sum_sq += float((non_null ** 2).sum())
    
    # Count zeros and negatives
    # Count zeros and negatives
    stats.zero_count += int((non_null == 0).sum())
    stats.negative_count += int((non_null < 0).sum())
    
    # Update reservoir sample for percentiles
    # Update reservoir sample for percentiles
    for val in non_null.values:
        update_reservoir_sample(stats.sample_values, float(val), stats.max_sample_size, stats.count, rng)


def compute_final_statistics(stats: FeatureStats, percentiles: List[float]) -> FeatureStats:
    """Compute final statistics including mean, std, and percentiles from reservoir sample."""
    if stats.count == 0:
        return stats
    
    # Compute mean
    mean_val = stats.sum_val / stats.count
    
    # Compute standard deviation
    if stats.count > 1:
        variance = (stats.sum_sq / stats.count) - (mean_val ** 2)
        std_val = np.sqrt(max(0, variance))
    else:
        std_val = 0.0
    
    # Compute percentiles from reservoir sample
    if len(stats.sample_values) > 0:
        sorted_sample = sorted(stats.sample_values)
        for p in percentiles:
            idx = int((p / 100.0) * (len(sorted_sample) - 1))
            idx = max(0, min(idx, len(sorted_sample) - 1))
            stats.percentiles[p] = float(sorted_sample[idx])
    
    # Estimate extreme frequency from the sample, then scale to full count.
    if 99.9 in stats.percentiles and len(stats.sample_values) > 0:
        extreme_threshold = stats.percentiles[99.9]
        # Count values above threshold in sample
        sample_extreme_count = sum(1 for v in stats.sample_values if v > extreme_threshold)
        # Scale to full dataset (approximate)
        extreme_rate = sample_extreme_count / len(stats.sample_values)
        stats.extreme_count = int(stats.count * extreme_rate)
    
    return stats


def analyze_outliers(
    input_dir: Path,
    dataset_name: str,
    logger: logging.Logger,
    max_features: Optional[int] = None,
    sample_partitions: bool = False,
    percentiles: List[float] = None
) -> OutlierAnalysisStats:
    """Analyze outliers in SMART features (read-only analysis)."""
    start_time = datetime.now()
    stats = OutlierAnalysisStats(dataset=dataset_name)
    
    if percentiles is None:
        percentiles = [0.1, 1.0, 99.0, 99.9]
    
    logger.info("=" * 60)
    logger.info("STAGE 5: OUTLIER ANALYSIS (READ-ONLY)")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Percentiles: {percentiles}")
    logger.info("Read-only analysis only: no data changes")
    
    # Find partitions
    partitions = find_partitions(input_dir)
    stats.total_partitions = len(partitions)
    logger.info(f"Found {stats.total_partitions} partitions")
    
    if stats.total_partitions == 0:
        logger.error(f"No partitions found in {input_dir}")
        return stats
    
    # Limit partitions if sampling
    if sample_partitions:
        partitions = partitions[:2]
        logger.info(f"Sampling mode: processing only {len(partitions)} partitions")
    
    # Initialize feature statistics
    feature_stats: Dict[str, FeatureStats] = {}
    all_smart_features: set = set()
    batch_size = 250000
    
    # First pass: identify all SMART features
    logger.info("=" * 60)
    logger.info("PASS 1: Identifying SMART features")
    logger.info("=" * 60)
    
    for partition_idx, partition_dir in enumerate(partitions, 1):
        partition_name = f"{partition_dir.parent.name}/{partition_dir.name}"
        logger.info(f"Scanning partition {partition_idx}/{len(partitions)}: {partition_name}")
        
        parquet_files = sorted(partition_dir.glob("*.parquet"))
        if len(parquet_files) == 0:
            continue
        
        for parquet_file in parquet_files:
            try:
                parquet_file_obj = pq.ParquetFile(parquet_file)
                # Read first batch to get schema
                first_batch = next(parquet_file_obj.iter_batches(batch_size=1000))
                df_sample = first_batch.to_pandas()
                
                # Identify SMART features
                smart_features = [col for col in df_sample.columns 
                                if (col.startswith('n_') or col.startswith('r_')) 
                                and col not in ['disk_id', 'model', 'ds', 'smart_day', 'y_7', 'y_14', 'y_30']]
                all_smart_features.update(smart_features)
                
            except Exception as e:
                logger.warning(f"Error scanning {parquet_file}: {e}")
                continue
        
        if max_features and len(all_smart_features) > max_features:
            all_smart_features = set(list(all_smart_features)[:max_features])
            logger.info(f"Limited to {max_features} features for analysis")
            break
    
    # Initialize statistics for each feature
    import random
    base_rng = random.Random(42)
    feature_rngs = {}
    
    for feature in sorted(all_smart_features):
        feature_stats[feature] = FeatureStats(
            feature_name=feature,
            max_sample_size=10000
        )
        feature_rngs[feature] = random.Random(base_rng.random())
    
    stats.smart_features_analyzed = len(feature_stats)
    logger.info(f"Found {stats.smart_features_analyzed} SMART features to analyze")
    
    # Second pass: compute statistics
    logger.info("=" * 60)
    logger.info("PASS 2: Computing statistics")
    logger.info("=" * 60)
    
    for partition_idx, partition_dir in enumerate(partitions, 1):
        partition_name = f"{partition_dir.parent.name}/{partition_dir.name}"
        logger.info(f"Processing partition {partition_idx}/{len(partitions)}: {partition_name}")
        
        try:
            parquet_files = sorted(partition_dir.glob("*.parquet"))
            if len(parquet_files) == 0:
                continue
            
            for parquet_file in parquet_files:
                try:
                    parquet_file_obj = pq.ParquetFile(parquet_file)
                    
                    for batch in parquet_file_obj.iter_batches(batch_size=batch_size):
                        df_batch = batch.to_pandas()
                        stats.total_rows += len(df_batch)
                        
                        # Analyze each feature
                        for feature_name in feature_stats.keys():
                            analyze_feature_chunk(
                                df_batch,
                                feature_name,
                                feature_stats[feature_name],
                                percentiles,
                                feature_rngs[feature_name],
                                batch_size
                            )
                
                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}", exc_info=True)
                    continue
            
            stats.partitions_processed += 1
            
        except Exception as e:
            stats.partitions_failed += 1
            logger.error(f"Error processing partition {partition_name}: {e}", exc_info=True)
            continue
    
    # Compute final statistics
    logger.info("=" * 60)
    logger.info("Computing final statistics")
    logger.info("=" * 60)
    
    for feature_name, feature_stat in feature_stats.items():
        feature_stat = compute_final_statistics(feature_stat, percentiles)
        
        # Identify features with issues
        if feature_stat.negative_count > 0:
            stats.features_with_negatives += 1
        
        # Check for extreme skew (if p99.9 >> p99)
        if 99.0 in feature_stat.percentiles and 99.9 in feature_stat.percentiles:
            p99 = feature_stat.percentiles[99.0]
            p99_9 = feature_stat.percentiles[99.9]
            if p99 > 0 and p99_9 / p99 > 10:  # 10x jump
                stats.features_with_extreme_skew += 1
        
        # Check for heavy tails (if max >> p99.9)
        if feature_stat.max_val is not None and 99.9 in feature_stat.percentiles:
            p99_9 = feature_stat.percentiles[99.9]
            if p99_9 > 0 and feature_stat.max_val / p99_9 > 10:  # 10x jump
                stats.features_with_heavy_tails += 1
    
    stats.feature_stats = feature_stats
    stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 60)
    logger.info(f"Analysis complete: {stats.partitions_processed}/{stats.total_partitions} partitions processed")
    logger.info(f"Total rows analyzed: {stats.total_rows:,}")
    logger.info(f"SMART features analyzed: {stats.smart_features_analyzed}")
    logger.info(f"Features with negatives: {stats.features_with_negatives}")
    logger.info(f"Features with extreme skew: {stats.features_with_extreme_skew}")
    logger.info(f"Features with heavy tails: {stats.features_with_heavy_tails}")
    logger.info(f"Processing time: {stats.processing_time_seconds:.2f}s")
    
    return stats


def write_reports(stats: OutlierAnalysisStats, dataset_name: str, logger: logging.Logger):
    """Write JSON and Markdown reports."""
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON report
    json_path = reports_dir / f"clean_stage5_outliers_{dataset_name}.json"
    
    # Convert feature_stats to dict for JSON serialization
    feature_stats_dict = {}
    for feature_name, feature_stat in stats.feature_stats.items():
        feature_stats_dict[feature_name] = {
            'feature_name': feature_stat.feature_name,
            'count': feature_stat.count,
            'min_val': feature_stat.min_val,
            'max_val': feature_stat.max_val,
            'mean': feature_stat.sum_val / feature_stat.count if feature_stat.count > 0 else None,
            'std': np.sqrt((feature_stat.sum_sq / feature_stat.count) - ((feature_stat.sum_val / feature_stat.count) ** 2)) if feature_stat.count > 1 else 0.0,
            'zero_count': feature_stat.zero_count,
            'negative_count': feature_stat.negative_count,
            'extreme_count': feature_stat.extreme_count,
            'percent_zero': (feature_stat.zero_count / feature_stat.count * 100) if feature_stat.count > 0 else 0.0,
            'percent_negative': (feature_stat.negative_count / feature_stat.count * 100) if feature_stat.count > 0 else 0.0,
            'percent_extreme': (feature_stat.extreme_count / feature_stat.count * 100) if feature_stat.count > 0 else 0.0,
            'percentiles': feature_stat.percentiles
        }
    
    report_json = {
        'dataset': dataset_name,
        'stage': 5,
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            'total_rows': stats.total_rows,
            'total_partitions': stats.total_partitions,
            'partitions_processed': stats.partitions_processed,
            'partitions_failed': stats.partitions_failed,
            'smart_features_analyzed': stats.smart_features_analyzed,
            'features_with_negatives': stats.features_with_negatives,
            'features_with_extreme_skew': stats.features_with_extreme_skew,
            'features_with_heavy_tails': stats.features_with_heavy_tails,
            'processing_time_seconds': stats.processing_time_seconds,
            'feature_stats': feature_stats_dict
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2, default=str)
    logger.info(f"JSON report written to {json_path}")
    
    # Markdown report
    md_path = reports_dir / f"clean_stage5_outliers_{dataset_name}.md"
    with open(md_path, 'w') as f:
        f.write(f"# Stage 5: Outlier Analysis - {dataset_name}\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("This stage is read-only. No data was modified.\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Dataset:** {dataset_name}\n")
        f.write(f"- **Total rows analyzed:** {stats.total_rows:,}\n")
        f.write(f"- **Total partitions:** {stats.total_partitions}\n")
        f.write(f"- **Partitions processed:** {stats.partitions_processed}\n")
        f.write(f"- **Partitions failed:** {stats.partitions_failed}\n")
        f.write(f"- **SMART features analyzed:** {stats.smart_features_analyzed}\n")
        f.write(f"- **Processing time:** {stats.processing_time_seconds:.2f} seconds\n\n")
        
        f.write("## Global Summary\n\n")
        f.write(f"- **Features with negative values:** {stats.features_with_negatives}\n")
        f.write(f"- **Features with extreme skew:** {stats.features_with_extreme_skew}\n")
        f.write(f"- **Features with heavy tails:** {stats.features_with_heavy_tails}\n")
        f.write("\n")
        
        # Top 30 most extreme features
        f.write("## Top 30 Most Extreme Features\n\n")
        
        # Calculate "extremeness" score for ranking
        feature_scores = []
        for feature_name, feature_stat in stats.feature_stats.items():
            if feature_stat.count == 0:
                continue
            
            # Calculate extremeness score
            score = 0.0
            if feature_stat.max_val is not None and feature_stat.min_val is not None:
                if feature_stat.min_val != feature_stat.max_val:
                    score += abs(feature_stat.max_val - feature_stat.min_val) / abs(feature_stat.min_val) if feature_stat.min_val != 0 else abs(feature_stat.max_val)
            
            if 99.9 in feature_stat.percentiles and 1.0 in feature_stat.percentiles:
                p1 = feature_stat.percentiles[1.0]
                p99_9 = feature_stat.percentiles[99.9]
                if p1 > 0:
                    score += (p99_9 / p1) if p1 != 0 else p99_9
            
            score += (feature_stat.negative_count / feature_stat.count * 100) if feature_stat.count > 0 else 0
            
            feature_scores.append((feature_name, feature_stat, score))
        
        # Sort by extremeness score
        feature_scores.sort(key=lambda x: x[2], reverse=True)
        
        f.write("| Feature | Min | Max | P1 | P99 | P99.9 | % Negative | % Zero |\n")
        f.write("|---------|-----|-----|----|----|-------|------------|--------|\n")
        
        for feature_name, feature_stat, _ in feature_scores[:30]:
            p1 = feature_stat.percentiles.get(1.0, None)
            p99 = feature_stat.percentiles.get(99.0, None)
            p99_9 = feature_stat.percentiles.get(99.9, None)
            
            pct_neg = (feature_stat.negative_count / feature_stat.count * 100) if feature_stat.count > 0 else 0.0
            pct_zero = (feature_stat.zero_count / feature_stat.count * 100) if feature_stat.count > 0 else 0.0
            
            # Format values (handle None cases)
            min_str = f"{feature_stat.min_val:.2f}" if feature_stat.min_val is not None else "N/A"
            max_str = f"{feature_stat.max_val:.2f}" if feature_stat.max_val is not None else "N/A"
            p1_str = f"{p1:.2f}" if p1 is not None else "N/A"
            p99_str = f"{p99:.2f}" if p99 is not None else "N/A"
            p99_9_str = f"{p99_9:.2f}" if p99_9 is not None else "N/A"
            
            f.write(
                f"| `{feature_name}` | "
                f"{min_str} | "
                f"{max_str} | "
                f"{p1_str} | "
                f"{p99_str} | "
                f"{p99_9_str} | "
                f"{pct_neg:.2f}% | "
                f"{pct_zero:.2f}% |\n"
            )
        f.write("\n")
        
        f.write("## Notes\n\n")
        f.write("### Are Outliers Likely Noise or Signal?\n\n")
        f.write("SMART attributes are hardware sensor readings that may exhibit extreme values when:\n")
        f.write("- Hardware is approaching failure (predictive signal)\n")
        f.write("- Sensors malfunction (noise)\n")
        f.write("- Data collection errors occur (noise)\n\n")
        f.write("**Analysis:** ")
        if stats.features_with_heavy_tails > 0:
            f.write(f"{stats.features_with_heavy_tails} features show heavy-tailed distributions, ")
            f.write("suggesting that extreme values may be meaningful failure indicators rather than noise.\n\n")
        else:
            f.write("Most features show relatively normal distributions with few extreme outliers.\n\n")
        
        f.write("### Which SMART Attributes Look Physically Implausible?\n\n")
        if stats.features_with_negatives > 0:
            f.write(f"**{stats.features_with_negatives} features contain negative values**, which may be:\n")
            f.write("- Data collection errors\n")
            f.write("- Signed integer overflow\n")
            f.write("- Invalid sensor readings\n\n")
        else:
            f.write("No features contain negative values, which is expected for most SMART attributes.\n\n")
        
        f.write("### Are Distributions Model-Dependent?\n\n")
        f.write("This analysis aggregates across all disk models. Model-specific analysis would require ")
        f.write("grouping by the `model` column to identify if certain models exhibit different outlier patterns.\n\n")
        
        f.write("## Recommendation\n\n")
        f.write("### No Outlier Clipping Applied\n\n")
        f.write("**At this stage, no outlier clipping has been applied.**\n\n")
        f.write("### Future Clipping Considerations\n\n")
        f.write("If you later add clipping during model training, it should:\n")
        f.write("1. Be computed on training data only (to prevent data leakage)\n")
        f.write("2. Use train-set percentiles (e.g., 1st and 99th) as clipping boundaries\n")
        f.write("3. Apply the same boundaries to test data\n")
        f.write("4. Be documented as a model-specific preprocessing step\n\n")
        f.write("### Recommendation\n\n")
        if stats.features_with_heavy_tails > stats.smart_features_analyzed * 0.1:
            f.write("Given the prevalence of heavy-tailed distributions, consider:\n")
            f.write("- Using robust models (e.g., tree-based) that handle outliers naturally\n")
            f.write("- Applying log transformations for highly skewed features\n")
            f.write("- Training-set-only clipping if using linear models\n\n")
        else:
            f.write("Most features show reasonable distributions. Outlier clipping may not be necessary.\n")
            f.write("Consider model-specific preprocessing during training if needed.\n\n")
    
    logger.info(f"Markdown report written to {md_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze outliers in SMART features (Stage 5 - Analysis Only)')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., smartlog2018ssd)')
    parser.add_argument('--max_features', type=int, default=None, help='Limit number of features to analyze (for debugging)')
    parser.add_argument('--sample_partitions', action='store_true', help='Process only 1-2 partitions (for testing)')
    parser.add_argument('--percentiles', nargs='+', type=float, default=[0.1, 1.0, 99.0, 99.9],
                       help='Percentiles to compute (default: 0.1 1.0 99.0 99.9)')
    
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
    logger.info("STAGE 5: OUTLIER ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info("This is an analysis-only stage. No data will be modified.")
    
    # Determine input directory
    input_dir = Path('data_interim') / f"clean_stage4_{args.dataset}"
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error(f"Please run Stage 4 first: python -m src.clean_labeled_pipeline --dataset {args.dataset} --stage 4")
        sys.exit(1)
    
    # Run analysis
    stats = analyze_outliers(
        input_dir,
        args.dataset,
        logger,
        max_features=args.max_features,
        sample_partitions=args.sample_partitions,
        percentiles=args.percentiles
    )
    
    # Write reports
    write_reports(stats, args.dataset, logger)
    
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Reports written to:")
    logger.info(f"  - reports/clean_stage5_outliers_{args.dataset}.json")
    logger.info(f"  - reports/clean_stage5_outliers_{args.dataset}.md")


if __name__ == '__main__':
    main()

