"""
Memory-safe smoke-test for the feature engineering pipeline.

Reads a small sample (one parquet file or N rows via PyArrow scanner) from
input and output partitions. Never loads a full month into RAM.

Checks: row count (sample), engineered columns exist, disk_id/smart_day present,
age_days >= 0, log1p_r_* finite where r_* present, temporal/instability columns present.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def find_partitions(input_dir: Path) -> list[Path]:
    """Return partition directories year=YYYY/month=MM under input_dir, sorted."""
    partitions = []
    for year_dir in sorted(input_dir.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            if month_dir.is_dir():
                partitions.append(month_dir)
    return partitions


def read_sample_from_partition(
    partition_dir: Path,
    sample_rows: int | None,
    use_first_file_only: bool,
) -> pa.Table:
    """
    Read a bounded sample from partition without loading full partition.
    If use_first_file_only: read only the first parquet file.
    Else if sample_rows: scan with PyArrow dataset and take first sample_rows rows.
    Else: take first 200_000 rows by default.
    """
    files = sorted(partition_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {partition_dir}")

    if use_first_file_only:
        return pq.read_table(files[0])
    n = sample_rows or 200_000
    dataset = ds.dataset(partition_dir, format="parquet")
    scanner = dataset.scanner(batch_size=min(n, 100_000))
    batches = []
    got = 0
    for batch in scanner.to_batches():
        need = n - got
        if batch.num_rows <= need:
            batches.append(batch)
            got += batch.num_rows
        else:
            batches.append(batch.slice(0, need))
            got += need
        if got >= n:
            break
    if not batches:
        return pa.table({})
    return pa.Table.from_batches(batches)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Memory-safe smoke-test: sample from input vs output partition"
    )
    parser.add_argument("--experiment", type=str, default="exp_time_generalisation")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="e.g. year=2018/month=01; default: first partition",
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=None,
        help="Max rows to sample from input/output (default: 200000). Use 0 with --first_file to read only one file.",
    )
    parser.add_argument(
        "--first_file",
        action="store_true",
        help="Read only the first parquet file in the partition (ignore sample_rows)",
    )
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    base = Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parent.parent
    input_dir = base / "data_splits" / args.experiment / args.split
    output_dir = base / "data_engineered" / args.experiment / args.split

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}. Run feature_engineering_pipeline first.")
        return 1

    partitions = find_partitions(input_dir)
    if not partitions:
        print(f"ERROR: No partitions under {input_dir}")
        return 1

    if args.partition:
        parts = args.partition.strip().split("/")
        if len(parts) != 2:
            print("ERROR: --partition should be like year=2018/month=01")
            return 1
        partition_dir_in = input_dir / parts[0] / parts[1]
        if not partition_dir_in.exists():
            print(f"ERROR: Partition not found: {partition_dir_in}")
            return 1
    else:
        partition_dir_in = partitions[0]
        args.partition = f"{partition_dir_in.parent.name}/{partition_dir_in.name}"

    year_part = partition_dir_in.parent.name
    month_part = partition_dir_in.name
    output_partition_dir = output_dir / year_part / month_part
    output_file = output_partition_dir / "data.parquet"

    if not output_file.exists():
        print(f"ERROR: Output partition file not found: {output_file}")
        return 1

    sample_rows = args.sample_rows if args.sample_rows is not None else 200_000
    if args.first_file:
        sample_rows = None

    try:
        tbl_in = read_sample_from_partition(
            partition_dir_in,
            sample_rows=sample_rows,
            use_first_file_only=args.first_file,
        )
    except Exception as e:
        print(f"ERROR: Failed to read input sample: {e}")
        return 1

    # Use the same sampling approach for output so the comparison is fair.
    if args.first_file:
        tbl_out = pq.read_table(output_file)
    else:
        dataset_out = ds.dataset(output_file, format="parquet")
        n = sample_rows or 200_000
        scanner = dataset_out.scanner(batch_size=min(n, 100_000))
        batches = []
        got = 0
        for batch in scanner.to_batches():
            need = n - got
            if batch.num_rows <= need:
                batches.append(batch)
                got += batch.num_rows
            else:
                batches.append(batch.slice(0, need))
                got += need
            if got >= n:
                break
        tbl_out = pa.Table.from_batches(batches) if batches else pa.table({})

    n_in, n_out = tbl_in.num_rows, tbl_out.num_rows
    print(f"Sample: input rows={n_in}, output rows={n_out}")

    # Output can include extra rows, but it should not have fewer than the sampled input.
    if n_in > 0 and n_out < n_in:
        print(f"FAIL: Output has fewer rows ({n_out}) than input sample ({n_in})")
        return 1
    print("OK: Row count (sample) consistent")

    # Required keys
    if "disk_id" not in tbl_out.schema.names or "smart_day" not in tbl_out.schema.names:
        print("FAIL: disk_id or smart_day missing in output")
        return 1
    print("OK: disk_id, smart_day present")

    # Expected engineered columns
    has_age = "age_days" in tbl_out.schema.names
    has_model_code = "model_code" in tbl_out.schema.names
    log1p_cols = [c for c in tbl_out.schema.names if c.startswith("log1p_")]
    delta_cols = [c for c in tbl_out.schema.names if c.startswith("delta1_") or c.startswith("delta7_")]
    roll_cols = [
        c
        for c in tbl_out.schema.names
        if c.startswith("rollmean7_") or c.startswith("rollstd7_") or c.startswith("rollvar7_")
    ]
    instab_cols = [c for c in tbl_out.schema.names if c.startswith("instab_")]
    if not has_age:
        print("FAIL: age_days missing")
        return 1
    if not has_model_code:
        print("WARN: model_code missing")
    if not log1p_cols:
        print("WARN: No log1p_* columns (feature_set may have no raw_features)")
    if not delta_cols:
        print("WARN: No delta1_/delta7_* columns (temporal block may be disabled)")
    if not roll_cols:
        print("WARN: No rollmean7_/rollstd7_/rollvar7_* columns")
    if not instab_cols:
        print("WARN: No instab_* columns")
    print(
        f"OK: Engineered columns (age_days, model_code={has_model_code}, log1p_*={len(log1p_cols)}, "
        f"delta={len(delta_cols)}, roll={len(roll_cols)}, instab={len(instab_cols)})"
    )

    # age_days should never be negative
    if n_out > 0 and has_age:
        age = tbl_out.column("age_days")
        neg = 0
        for i in range(age.length()):
            v = age[i].as_py()
            if v is not None and v is not pa.scalar(None) and (isinstance(v, (int, float)) and v < 0):
                neg += 1
        if neg > 0:
            print(f"FAIL: {neg} rows with age_days < 0")
            return 1
    print("OK: age_days >= 0")

    # Spot-check that log1p values are finite where the base value exists.
    for logcol in log1p_cols[:5]:
        base_col = logcol.replace("log1p_", "")
        if base_col not in tbl_out.schema.names:
            continue
        log_arr = tbl_out.column(logcol)
        base_arr = tbl_out.column(base_col)
        bad = 0
        for i in range(min(log_arr.length(), 50_000)):
            b = base_arr[i].as_py()
            if b is None or (isinstance(b, float) and math.isnan(b)):
                continue
            lv = log_arr[i].as_py()
            if lv is not None and not (isinstance(lv, float) and math.isnan(lv)):
                if not (isinstance(lv, (int, float)) and math.isfinite(lv)):
                    bad += 1
        if bad > 0:
            print(f"WARN: {logcol} has {bad} non-finite values where base present")
        else:
            print(f"OK: {logcol} finite where {base_col} present (spot check)")
    if not log1p_cols:
        print("OK: log1p spot check skipped (no log1p columns)")

    # Basic temporal column presence check.
    for dc in delta_cols[:3]:
        print(f"OK: {dc} column present (temporal block)")

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
