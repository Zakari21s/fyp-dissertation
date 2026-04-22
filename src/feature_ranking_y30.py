"""
Feature ranking by univariate ROC-AUC vs y_30 on TRAIN split (streaming, out-of-core).

ROOT CAUSE OF OOM (previous version): Pass A and Pass B each requested ALL feature columns
plus the label in every batch (columns=[label_col] + feature_cols). With 56+ features,
each batch was ~250k rows × 57 columns × 8 bytes ≈ 114 MB. Over many files and batches,
PyArrow decoding, to_pydict() copies, and OS buffering pushed peak memory over 16GB and
the process was killed. We never concatenated full partitions, but loading 57 columns per
batch was the memory bottleneck.

FIX (v1): Process ONE FEATURE AT A TIME; only two columns per batch.

FIX (v2): to_pydict() was removed — it converts each Arrow RecordBatch into Python
lists (one per column), which massively expands memory and triggers macOS OOM on full
TRAIN (~133M rows). Streaming is now memory-safe by yielding numpy arrays directly:
we use batch.column(i).to_numpy(zero_copy_only=False) and cast to float32 for the
feature column (and float64 for label to handle NaN). We only keep in memory:
histogram arrays (2 × 256 int64), min/max scalars, and counts. No Python dicts/lists
from to_pydict(), no pandas Series. Optional default cap: unless FULL_RUN=1, we
default --max_rows 20_000_000 so local runs don't OOM.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Constants
EXCLUDE_COLS = {"disk_id", "model", "smart_day", "ds", "y_7", "y_14", "y_30"}
REPORTS_DIR = Path("reports")
TOP_N_MD = 30
TOP_N_CONSOLE = 20
MIN_VALID_COUNT = 50_000
DEFAULT_BINS = 256
DEFAULT_BATCH_SIZE = 250_000
LOG_ROWS_EVERY = 5_000_000  # log progress every N rows scanned
# Conservative default cap for local runs; set env FULL_RUN=1 for full 133M+ scan
DEFAULT_MAX_ROWS_CAP = 20_000_000


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure logging to file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "feature_ranking_y30.log"

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


def find_partitions(input_dir: Path) -> list[Path]:
    """Return partition directories year=YYYY/month=MM under input_dir."""
    partitions = []
    for year_dir in sorted(input_dir.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            if month_dir.is_dir():
                partitions.append(month_dir)
    return partitions


def collect_parquet_files(
    train_dir: Path,
    max_partitions: int | None,
    logger: logging.Logger,
) -> tuple[list[Path], int]:
    """
    List parquet files under train_dir. Optionally limit partitions.
    Returns (list of file paths, number of partitions represented).
    """
    partitions = find_partitions(train_dir)
    n_partitions = len(partitions)
    if max_partitions is not None:
        partitions = partitions[:max_partitions]
        n_partitions = len(partitions)
        logger.info("Limiting to first %d partitions (--max_partitions)", max_partitions)
    files = []
    for part_dir in partitions:
        for f in sorted(part_dir.glob("*.parquet")):
            files.append(f)
    return files, n_partitions


def get_numeric_feature_columns(schema, label_col: str, logger: logging.Logger) -> list[str]:
    """From PyArrow schema, return numeric columns excluding EXCLUDE_COLS and labels."""
    exclude = EXCLUDE_COLS | {label_col}
    numeric = []
    for i in range(len(schema)):
        field = schema.field(i)
        if field.name in exclude:
            continue
        t = field.type
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            numeric.append(field.name)
    if not numeric:
        for i in range(len(schema)):
            field = schema.field(i)
            if field.name not in exclude:
                numeric.append(field.name)
    logger.info("Detected %d numeric feature columns", len(numeric))
    return sorted(numeric)


def _auc_from_histograms(hist_pos: np.ndarray, hist_neg: np.ndarray, total_pos: int, total_neg: int) -> float:
    """Compute ROC-AUC from pos/neg histograms. Trapezoidal integration."""
    if total_pos == 0 or total_neg == 0:
        return 0.5
    n = len(hist_pos)
    cum_neg_rev = np.cumsum(hist_neg[::-1])[::-1]
    cum_pos_rev = np.cumsum(hist_pos[::-1])[::-1]
    fpr = np.array([cum_neg_rev[i] / total_neg for i in range(n)] + [0.0])
    tpr = np.array([cum_pos_rev[i] / total_pos for i in range(n)] + [0.0])
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(trapz_fn(tpr, fpr))


def stream_batches(
    parquet_files: list[Path],
    columns: list[str],
    batch_size: int,
    max_rows: int | None,
    logger: logging.Logger,
):
    """
    Generator: yield numpy arrays per batch. MEMORY-SAFE: we do NOT use to_pydict()
    (which converts Arrow → Python lists and explodes memory). We yield either:
    - len(columns)==1 (row count): (None, n) so caller does rows_scanned += n.
    - len(columns)==2 (label + feature): (y_np, x_np) with y_np float64 (for NaN),
      x_np float32 to bound memory. Built via column(i).to_numpy(zero_copy_only=False).
    Stops after max_rows total if set.
    """
    total = 0
    file_idx = 0
    n_cols = len(columns)
    for path in parquet_files:
        file_idx += 1
        logger.info("Scanning partition/file %d/%d: %s", file_idx, len(parquet_files), path.name)
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            logger.error("Open %s: %s", path, e)
            continue
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            n = batch.num_rows
            if max_rows is not None and total + n > max_rows:
                n = max_rows - total
                if n <= 0:
                    return
                batch = batch.slice(0, n)
            total += n
            if total % LOG_ROWS_EVERY < n or total == n:
                logger.info("Rows scanned so far: %d", total)
            if n_cols == 1:
                yield None, n
            else:
                # columns = [label_col, feature_col]: column(0)=label, column(1)=feature
                y_chunk = batch.column(0)
                x_chunk = batch.column(1)
                # to_numpy(zero_copy_only=False) avoids holding Arrow buffer; copy into numpy
                y_np = np.ascontiguousarray(y_chunk.to_numpy(zero_copy_only=False).astype(np.float64))
                x_np = np.ascontiguousarray(x_chunk.to_numpy(zero_copy_only=False).astype(np.float32))
                yield y_np, x_np
            if max_rows is not None and total >= max_rows:
                return


def process_one_feature(
    parquet_files: list[Path],
    col: str,
    label_col: str,
    n_bins: int,
    batch_size: int,
    max_rows: int | None,
    logger: logging.Logger,
) -> dict | None:
    """
    Two-pass streaming for a single feature. Consumes (y_np, x_np) from stream_batches
    only — no to_pydict(), no pd.Series. Uses numpy masks (~np.isnan(x) & (y==0)|(y==1))
    and keeps in memory only: hist_pos/hist_neg (2 × n_bins int64), min_val, max_val, counts.
    """
    columns = [label_col, col]
    # Pass 1: min/max and counts
    min_val, max_val = np.inf, -np.inf
    valid_count = 0
    pos_count = 0
    neg_count = 0

    for y_np, x_np in stream_batches(parquet_files, columns, batch_size, max_rows, logger):
        if y_np is None:
            continue
        y_ok = np.isfinite(y_np)
        x_ok = np.isfinite(x_np)
        valid = x_ok & y_ok
        if not np.any(valid):
            continue
        valid_count += int(np.sum(valid))
        xv = x_np[valid]
        yv = y_np[valid]
        min_val = min(min_val, float(np.min(xv)))
        max_val = max(max_val, float(np.max(xv)))
        pos_count += int(np.sum(yv == 1))
        neg_count += int(np.sum(yv == 0))

    if valid_count < MIN_VALID_COUNT or min_val >= max_val or pos_count == 0 or neg_count == 0:
        return None

    # Pass 2: histograms — only histogram arrays kept in memory
    edges = np.linspace(min_val, max_val, n_bins + 1, dtype=np.float32)
    hist_pos = np.zeros(n_bins, dtype=np.int64)
    hist_neg = np.zeros(n_bins, dtype=np.int64)

    for y_np, x_np in stream_batches(parquet_files, columns, batch_size, max_rows, logger):
        if y_np is None:
            continue
        valid = np.isfinite(x_np) & np.isfinite(y_np)
        if not np.any(valid):
            continue
        xv = x_np[valid]
        yv = y_np[valid]
        bins = np.searchsorted(edges, xv, side="right") - 1
        bins = np.clip(bins, 0, n_bins - 1)
        # Vectorized histogram update: no Python-level loop over rows
        np.add.at(hist_pos, bins, (yv == 1).astype(np.int64))
        np.add.at(hist_neg, bins, (yv == 0).astype(np.int64))

    pos_sum = int(np.sum(hist_pos))
    neg_sum = int(np.sum(hist_neg))
    if pos_sum == 0 or neg_sum == 0:
        return None
    auc = _auc_from_histograms(hist_pos, hist_neg, pos_sum, neg_sum)
    return {
        "feature_name": col,
        "auc": round(auc, 6),
        "abs_auc_minus_0_5": round(abs(auc - 0.5), 6),
        "valid_count": valid_count,
        "pos_count": pos_sum,
        "neg_count": neg_sum,
        "min": round(min_val, 6),
        "max": round(max_val, 6),
    }


def run_feature_by_feature(
    train_dir: Path,
    label_col: str,
    feature_cols: list[str],
    n_bins: int,
    batch_size: int,
    max_partitions: int | None,
    max_rows: int | None,
    logger: logging.Logger,
) -> tuple[list[dict], int, int]:
    """
    Process each feature independently with two-pass streaming (only 2 columns per batch).
    Returns (results list, total_rows_scanned_estimate, partitions_scanned).
    We don't have a single "total rows" across all features because we stop at max_rows per stream;
    we report the row limit or None. For summary we use the number of rows from the first feature's
    stream (or we could count once). Actually we need to report rows_scanned: we can count rows
    during the first feature's first pass and reuse that as the nominal "rows scanned" for the run.
    """
    parquet_files, partitions_scanned = collect_parquet_files(train_dir, max_partitions, logger)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {train_dir}")

    rows_scanned = 0
    results = []
    for i, col in enumerate(feature_cols):
        logger.info("Feature %d/%d: %s", i + 1, len(feature_cols), col)
        rec = process_one_feature(
            parquet_files, col, label_col, n_bins, batch_size, max_rows, logger
        )
        if rec is None:
            continue
        results.append(rec)
        # Count rows once (from first feature's first pass we don't have it here; count in process_one_feature and return?)
        # Simpler: count rows in a single lightweight pass, or report "see per-feature valid_count". User asked for rows_scanned.
        # We can run a tiny "count pass" that only reads label column and counts, with max_rows. That's one extra pass but minimal memory.
        # Alternatively: sum valid_count across features is wrong. So we need one dedicated row-count pass that only reads one column and stops at max_rows.
        # For now, set rows_scanned from the first feature: we don't have it. So we'll add a quick row-count at the start that streams only label_col and counts rows (with max_rows). That gives us rows_scanned and partitions_scanned.
        # Actually user said "rows_scanned and partitions_scanned summary line". So we need to know total rows (capped by max_rows). Easiest: in run_feature_by_feature, do one initial pass that only reads label_col and counts rows (with max_rows). Then we have rows_scanned. partitions_scanned we already have from collect_parquet_files.
    # Count rows in one lightweight pass (only label column)
    if not parquet_files:
        return results, 0, partitions_scanned
    rows_scanned = 0
    for d, n in stream_batches(parquet_files, [label_col], batch_size, max_rows, logger):
        rows_scanned += n
    return results, rows_scanned, partitions_scanned


def run_feature_by_feature_with_row_count(
    train_dir: Path,
    label_col: str,
    feature_cols: list[str],
    n_bins: int,
    batch_size: int,
    max_partitions: int | None,
    max_rows: int | None,
    logger: logging.Logger,
) -> tuple[list[dict], int, int]:
    """Run feature-by-feature ranking and count total rows in a separate lightweight pass."""
    parquet_files, partitions_scanned = collect_parquet_files(train_dir, max_partitions, logger)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {train_dir}")

    # Lightweight row count (only label column, bounded by max_rows); no to_pydict, yields (None, n)
    logger.info("Counting rows (label column only)...")
    rows_scanned = 0
    for _, n in stream_batches(parquet_files, [label_col], batch_size, max_rows, logger):
        rows_scanned += n
    logger.info("Rows to scan per feature: %d (partitions: %d)", rows_scanned, partitions_scanned)

    results = []
    for i, col in enumerate(feature_cols):
        logger.info("Feature %d/%d: %s", i + 1, len(feature_cols), col)
        rec = process_one_feature(
            parquet_files, col, label_col, n_bins, batch_size, max_rows, logger
        )
        if rec is None:
            continue
        results.append(rec)
    return results, rows_scanned, partitions_scanned


def rank_and_save(
    results: list[dict],
    train_dir: Path,
    label_col: str,
    rows_scanned: int,
    partitions_scanned: int,
    logger: logging.Logger,
) -> list[dict]:
    """Sort by abs_auc_minus_0_5 descending; write CSV, JSON, MD; return sorted list."""
    if not results:
        logger.warning("No features to rank")
        return []

    sorted_list = sorted(results, key=lambda x: x["abs_auc_minus_0_5"], reverse=True)
    for i, rec in enumerate(sorted_list, start=1):
        rec["rank"] = i

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["rank", "feature_name", "auc", "abs_auc_minus_0_5", "valid_count", "pos_count", "neg_count", "min", "max"]
    csv_path = REPORTS_DIR / "feature_ranking_y30.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for rec in sorted_list:
            f.write(",".join(str(rec.get(k, "")) for k in cols) + "\n")
    logger.info("Wrote %s", csv_path)

    json_path = REPORTS_DIR / "feature_ranking_y30.json"
    payload = {
        "label_column": label_col,
        "train_dir": str(train_dir),
        "rows_scanned": rows_scanned,
        "partitions_scanned": partitions_scanned,
        "num_features_ranked": len(sorted_list),
        "ranking": sorted_list,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", json_path)

    md_path = REPORTS_DIR / "feature_ranking_y30.md"
    top = sorted_list[:TOP_N_MD]
    lines = [
        "# Feature ranking (y_30, TRAIN, streaming)",
        "",
        "Approximate univariate ROC-AUC from histograms; sorted by |AUC - 0.5| descending.",
        "",
        "**Summary:** rows_scanned = {} | partitions_scanned = {}".format(rows_scanned, partitions_scanned),
        "",
        "| Rank | Feature | ROC-AUC | |AUC-0.5| | ValidCount | PosCount | NegCount | Min | Max |",
        "|------|---------|--------|---------|------------|--------|--------|-----|-----|",
    ]
    for rec in top:
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {} | {} | {} | {:.4f} | {:.4f} |".format(
                rec["rank"],
                rec["feature_name"],
                rec["auc"],
                rec["abs_auc_minus_0_5"],
                rec["valid_count"],
                rec["pos_count"],
                rec["neg_count"],
                rec["min"],
                rec["max"],
            )
        )
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote %s (top %d)", md_path, TOP_N_MD)

    return sorted_list


def print_top_n(ranking: list[dict], n: int = TOP_N_CONSOLE) -> None:
    """Print top n to console."""
    top = ranking[:n]
    print("\nTop {} features by |AUC - 0.5|:\n".format(n))
    print("{:>6} {:36} {:>10} {:>10} {:>12} {:>10} {:>10}".format(
        "Rank", "Feature", "ROC-AUC", "|AUC-0.5|", "ValidCount", "PosCount", "NegCount"))
    print("-" * 100)
    for rec in top:
        print("{:>6} {:36} {:10.4f} {:10.4f} {:12} {:10} {:10}".format(
            rec["rank"],
            (rec["feature_name"])[:36],
            rec["auc"],
            rec["abs_auc_minus_0_5"],
            rec["valid_count"],
            rec["pos_count"],
            rec["neg_count"],
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ranking y_30 (streaming, TRAIN only)")
    parser.add_argument("--experiment", default="exp_time_generalisation", help="Experiment name")
    parser.add_argument("--split", default="train", help="Split (e.g. train)")
    parser.add_argument("--label", default="y_30", help="Label column")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of histogram bins")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Rows per batch")
    parser.add_argument("--max_partitions", type=int, default=None, help="Limit partitions (for testing)")
    parser.add_argument("--max_rows", type=int, default=None, help="Stop after scanning this many rows (default: 20M unless FULL_RUN=1)")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature names to restrict (default: all numeric)")
    args = parser.parse_args()

    # Safety default: cap rows unless FULL_RUN=1 so local runs don't OOM
    if args.max_rows is None and os.environ.get("FULL_RUN") != "1":
        args.max_rows = DEFAULT_MAX_ROWS_CAP
    capped_mode = args.max_rows is not None

    start_time = datetime.now()
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    logger = setup_logging(log_dir)

    train_dir = base_dir / "data_splits" / args.experiment / args.split
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    if capped_mode:
        logger.info("MODE: capped (max_rows=%s). Set env FULL_RUN=1 for full scan.", args.max_rows)
        print("MODE: capped (max_rows={}). Set env FULL_RUN=1 for full scan.".format(args.max_rows))
    else:
        logger.info("MODE: full run (no row cap)")
        print("MODE: full run (no row cap)")
    logger.info("=== Feature ranking (streaming, feature-by-feature) label=%s, split=%s ===", args.label, args.split)
    logger.info("Train dir: %s", train_dir)
    logger.info("Bins: %d, batch_size: %d, max_partitions: %s, max_rows: %s", args.bins, args.batch_size, args.max_partitions, args.max_rows)

    parquet_files, partitions_scanned = collect_parquet_files(train_dir, args.max_partitions, logger)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {train_dir}")

    pf0 = pq.ParquetFile(parquet_files[0])
    schema = getattr(pf0, "schema_arrow", pf0.schema)
    schema_names = [schema.field(i).name for i in range(len(schema))]
    if args.label not in schema_names:
        raise ValueError(f"Label column '{args.label}' not in schema")
    feature_cols = get_numeric_feature_columns(schema, args.label, logger)
    if args.features:
        requested = [s.strip() for s in args.features.split(",") if s.strip()]
        feature_cols = [c for c in feature_cols if c in requested]
        if len(feature_cols) != len(requested):
            logger.warning("Some --features not found or not numeric; using %d features", len(feature_cols))
        logger.info("Restricted to %d features (--features)", len(feature_cols))
    if not feature_cols:
        raise ValueError("No feature columns to process")

    results, rows_scanned, _ = run_feature_by_feature_with_row_count(
        train_dir,
        args.label,
        feature_cols,
        args.bins,
        args.batch_size,
        args.max_partitions,
        args.max_rows,
        logger,
    )
    # Re-get partitions_scanned for report
    _, partitions_scanned = collect_parquet_files(train_dir, args.max_partitions, logger)

    logger.info("Total rows scanned: %d, partitions_scanned: %d", rows_scanned, partitions_scanned)
    print("Rows scanned: {}".format(rows_scanned))
    print("Partitions scanned: {}".format(partitions_scanned))

    ranking = rank_and_save(results, train_dir, args.label, rows_scanned, partitions_scanned, logger)
    print_top_n(ranking, TOP_N_CONSOLE)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("Execution time: %.2f seconds", elapsed)
    print("\nExecution time: {:.2f} seconds".format(elapsed))

    try:
        import psutil
        proc = psutil.Process()
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        logger.info("Peak memory (RSS): %.1f MB", mem_mb)
        print("Peak memory (RSS): {:.1f} MB".format(mem_mb))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
