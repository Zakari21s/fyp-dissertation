"""
Memory-aware feature engineering pipeline for SMART SSD split data.

Reads from data_splits/<experiment>/<split> using PyArrow dataset streaming.
Writes to data_engineered/<experiment>/<split>/year=YYYY/month=MM/data.parquet.

Temporal features (calendar-day, past-only, per disk_id) are computed on
log1p(raw) series after sorting each partition by (smart_day, disk_id) so that
per-disk time order is correct. Sorting uses Dataset.sort_by (materializes the
partition in memory — see module docstring).

age_days uses global first_seen per disk_id: min(smart_day) over the full input
split, merged with train for val/test when train JSON exists (see
_pass_merge_first_seen).

Row caps (train/val/test) apply to a single chronological stream: partition
order (year/month), then sort keys. No shuffle before temporal features.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

# Constants
REPORTS_DIR = Path("reports")
CONFIG_DIR = Path("configs")
FEATURE_SET_PATH = CONFIG_DIR / "feature_set.yaml"
REQUIRED_COLUMNS = ("disk_id", "smart_day", "model")


def setup_logging(
    log_dir: Path,
    experiment: str,
    split: str,
    log_level: str = "INFO",
) -> logging.Logger:
    """Configure logging to file and console; log file per experiment/split."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"feature_engineering_{experiment}_{split}.log"
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
    """Return partition directories year=YYYY/month=MM under input_dir, sorted."""
    partitions = []
    for year_dir in sorted(input_dir.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            if month_dir.is_dir():
                partitions.append(month_dir)
    return partitions


def get_partition_schema(partition_dir: Path, batch_size: int) -> pa.Schema:
    """Read first batch from partition to get schema."""
    dataset = ds.dataset(partition_dir, format="parquet")
    scanner = dataset.scanner(batch_size=min(batch_size, 100_000))
    for batch in scanner.to_batches():
        return batch.schema
    raise ValueError(f"No data in partition {partition_dir}")


def _to_date_py(val: Any) -> date | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if hasattr(val, "as_py"):
        return _to_date_py(val.as_py())
    return None


def load_feature_set(
    config_path: Path, schema: pa.Schema, logger: logging.Logger
) -> tuple[list[str], list[str], dict[str, Any]]:
    """
    Load raw_features, norm_features, and temporal/row_caps from YAML.
    """
    if not config_path.exists():
        logger.warning("Feature set config not found at %s; using empty lists", config_path)
        return [], [], {}
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    raw = list(cfg.get("raw_features") or [])
    norm = list(cfg.get("norm_features") or [])
    names = set(schema.names)
    raw = [c for c in raw if c in names]
    norm = [c for c in norm if c in names]
    meta = {
        "row_caps": cfg.get("row_caps") or {},
        "temporal": cfg.get("temporal") or {},
        "instability": cfg.get("instability") or {},
    }
    logger.info("Feature set: %d raw, %d norm (from config)", len(raw), len(norm))
    return raw, norm, meta


def _row_cap_for_split(split: str, meta: dict[str, Any]) -> int | None:
    caps = meta.get("row_caps") or {}
    if split not in caps:
        return None
    v = caps[split]
    return int(v) if v is not None else None


def _temporal_base_columns(raw_features: list[str], meta: dict[str, Any]) -> list[str]:
    t = meta.get("temporal") or {}
    tb = t.get("base_columns")
    if tb is None:
        return list(raw_features)
    return [c for c in tb if c in raw_features]


def disk_first_seen_path(base_dir: Path, experiment: str) -> Path:
    return base_dir / REPORTS_DIR / f"disk_first_seen_{experiment}.json"


def load_disk_first_seen_train(base_dir: Path, experiment: str, logger: logging.Logger) -> dict[str, str]:
    """Load disk_id -> ISO date string from train run."""
    p = disk_first_seen_path(base_dir, experiment)
    if not p.exists():
        logger.warning("disk_first_seen not found at %s; val/test age may use split-local min only", p)
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        logger.warning("Could not load %s: %s", p, e)
        return {}


def save_disk_first_seen(base_dir: Path, experiment: str, first_seen: dict[int, Any], logger: logging.Logger) -> None:
    """Save disk_id -> ISO date for first observation (train)."""
    out = disk_first_seen_path(base_dir, experiment)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(did): (v.isoformat() if isinstance(v, date) else str(v)) for did, v in first_seen.items()}
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved disk first_seen map to %s (%d disks)", out, len(serializable))


def pass0_stream_min_disk_day(
    partition_dir: Path,
    batch_size: int,
    acc: dict[int, date],
) -> None:
    """Merge min calendar day per disk_id into acc (mutation)."""
    dataset = ds.dataset(partition_dir, format="parquet")
    scanner = dataset.scanner(batch_size=batch_size)
    for batch in scanner.to_batches():
        disk_id = batch.column("disk_id")
        smart_day = batch.column("smart_day")
        for i in range(batch.num_rows):
            try:
                did = int(disk_id[i].as_py()) if disk_id[i].as_py() is not None else None
            except (TypeError, ValueError):
                did = None
            if did is None:
                continue
            d = _to_date_py(smart_day[i])
            if d is None:
                continue
            if did not in acc or d < acc[did]:
                acc[did] = d


def pass0_collect_first_seen_and_models(
    partitions: list[Path],
    batch_size: int,
    logger: logging.Logger,
) -> tuple[dict[int, date], set[str]]:
    """
    Stream all partitions (unsorted): global min(smart_day) per disk_id and unique model strings.
    Lightweight columns only where possible; full scan of rows for train split.
    """
    first_seen: dict[int, date] = {}
    models_seen: set[str] = set()
    for partition_dir in partitions:
        dataset = ds.dataset(partition_dir, format="parquet")
        scanner = dataset.scanner(batch_size=batch_size)
        for batch in scanner.to_batches():
            disk_id = batch.column("disk_id")
            smart_day = batch.column("smart_day")
            model_col = batch.column("model") if "model" in batch.schema.names else None
            for i in range(batch.num_rows):
                try:
                    did = int(disk_id[i].as_py()) if disk_id[i].as_py() is not None else None
                except (TypeError, ValueError):
                    did = None
                if did is None:
                    continue
                d = _to_date_py(smart_day[i])
                if d is None:
                    continue
                if did not in first_seen or d < first_seen[did]:
                    first_seen[did] = d
                if model_col is not None:
                    m = model_col[i].as_py()
                    if m is not None:
                        models_seen.add(str(m))
    logger.info("Pass0 collect: %d disks, %d models", len(first_seen), len(models_seen))
    return first_seen, models_seen


def pass0_merge_first_seen_non_train(
    partitions: list[Path],
    batch_size: int,
    train_fs: dict[str, str],
    logger: logging.Logger,
) -> dict[int, date]:
    """
    For val/test: merge train first_seen with min(smart_day) in this split per disk.
    Disks only in val/test get split-local min; disks in train use train min.
    """
    split_min: dict[int, date] = {}
    for partition_dir in partitions:
        pass0_stream_min_disk_day(partition_dir, batch_size, split_min)

    merged: dict[int, date] = {}
    train_parsed: dict[int, date] = {}
    for k, v in train_fs.items():
        try:
            train_parsed[int(k)] = date.fromisoformat(v)
        except (TypeError, ValueError):
            continue

    all_disks = set(split_min.keys()) | set(train_parsed.keys())
    for did in all_disks:
        candidates = []
        if did in train_parsed:
            candidates.append(train_parsed[did])
        if did in split_min:
            candidates.append(split_min[did])
        if candidates:
            merged[did] = min(candidates)
    logger.info("Merged first_seen: %d disks (train keys + split)", len(merged))
    return merged


def load_model_mapping(experiment: str, logger: logging.Logger) -> dict[str, int] | None:
    path = REPORTS_DIR / f"model_mapping_{experiment}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return {k: int(v) for k, v in data.items()}
    except Exception as e:
        logger.warning("Could not load model mapping %s: %s", path, e)
        return None


def save_model_mapping(experiment: str, mapping: dict[str, int], logger: logging.Logger) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"model_mapping_{experiment}.json"
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved model mapping to %s (%d models)", path, len(mapping))


class TemporalState:
    """
    Per disk_id, per log1p column name: map calendar day ordinal -> log1p value.
    Past-only deltas and 7-day inclusive rolling stats ending at D.
    """

    def __init__(
        self,
        log1p_names: list[str],
        roll_dispersion: str,
        instability_metric: str,
        inst_prefix: str,
    ) -> None:
        self.log1p_names = log1p_names
        self.roll_dispersion = roll_dispersion.lower()
        self.instability_metric = instability_metric.lower()
        self.inst_prefix = inst_prefix
        # disk_id -> col -> dict[ordinal, float]
        self._hist: dict[int, dict[str, dict[int, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def nan_outputs(self) -> dict[str, float]:
        """Same keys as update() for rows without a valid disk/day."""
        out: dict[str, float] = {}
        for col in self.log1p_names:
            out[f"delta1_{col}"] = float("nan")
            out[f"delta7_{col}"] = float("nan")
            out[f"rollmean7_{col}"] = float("nan")
            if self.roll_dispersion == "var":
                out[f"rollvar7_{col}"] = float("nan")
            else:
                out[f"rollstd7_{col}"] = float("nan")
            out[f"{self.inst_prefix}{col}"] = float("nan")
        return out

    def _roll_stats(self, vals: list[float]) -> tuple[float, float]:
        arr = np.asarray(vals, dtype=np.float64)
        mask = np.isfinite(arr)
        if mask.sum() == 0:
            return float("nan"), float("nan")
        m = float(np.nanmean(arr))
        if self.roll_dispersion == "var":
            if mask.sum() < 2:
                disp = float("nan")
            else:
                disp = float(np.nanvar(arr, ddof=1))
        else:
            if mask.sum() < 2:
                disp = float("nan")
            else:
                disp = float(np.nanstd(arr, ddof=1))
        return m, disp

    def update(
        self,
        disk_id: int,
        day_ord: int,
        log1p_vals: dict[str, float],
    ) -> dict[str, float]:
        """Returns flat engineered column name -> value for this row."""
        out: dict[str, float] = {}
        hdisk = self._hist[disk_id]
        for col in self.log1p_names:
            v = log1p_vals.get(col)
            if v is None or not math.isfinite(v):
                v = float("nan")
            hist = hdisk[col]
            d1 = hist.get(day_ord - 1)
            d7 = hist.get(day_ord - 7)
            delta1 = v - d1 if d1 is not None and math.isfinite(d1) and math.isfinite(v) else float("nan")
            delta7 = v - d7 if d7 is not None and math.isfinite(d7) and math.isfinite(v) else float("nan")
            win = [hist.get(day_ord - k) for k in range(6, -1, -1)]
            win[-1] = v if math.isfinite(v) else float("nan")
            wvals = []
            for x in win:
                if x is not None and math.isfinite(x):
                    wvals.append(float(x))
                else:
                    wvals.append(float("nan"))
            rm, rd = self._roll_stats(wvals)
            dname_d1 = f"delta1_{col}"
            dname_d7 = f"delta7_{col}"
            out[dname_d1] = delta1
            out[dname_d7] = delta7
            out[f"rollmean7_{col}"] = rm
            if self.roll_dispersion == "var":
                out[f"rollvar7_{col}"] = rd
            else:
                out[f"rollstd7_{col}"] = rd
            if self.instability_metric == "abs_delta1":
                out[f"{self.inst_prefix}{col}"] = abs(delta1) if math.isfinite(delta1) else float("nan")
            hist[day_ord] = v if math.isfinite(v) else float("nan")
            for ko in list(hist.keys()):
                if ko < day_ord - 8:
                    del hist[ko]
        return out


def _log1p_from_raw(raw_val: Any) -> float:
    try:
        x = float(raw_val.as_py() if hasattr(raw_val, "as_py") else raw_val)
    except (TypeError, ValueError):
        return float("nan")
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return float("nan")
    x = max(0.0, x)
    return math.log1p(x)


def _age_days(smart_day: Any, first_py: date | None) -> float | None:
    if first_py is None:
        return None
    d = _to_date_py(smart_day)
    if d is None:
        return None
    return max(0.0, float((d - first_py).days))


def process_sorted_batch(
    batch: pa.RecordBatch,
    first_seen: dict[int, date],
    raw_features: list[str],
    model_mapping: dict[str, int],
    temporal: TemporalState,
) -> tuple[pa.RecordBatch, float | None, float | None, float | None, int]:
    """
    Build engineered batch: originals + age_days + log1p_* + temporal + model_code.
    Returns batch, age_min, age_max, age_mean (over finite ages), n_rows.
    """
    names = list(batch.schema.names)
    n = batch.num_rows
    disk_id_col = batch.column("disk_id")
    smart_day_col = batch.column("smart_day")
    model_col = batch.column("model") if "model" in batch.schema.names else None

    log1p_names_all = [f"log1p_{c}" for c in raw_features]

    age_vals: list[float | None] = []
    log1p_arrays: dict[str, list[float]] = {ln: [] for ln in log1p_names_all}
    tkeys = list(temporal.nan_outputs().keys())
    temporal_cols: dict[str, list[float]] = {tk: [] for tk in tkeys}

    codes: list[int] = []
    age_sum, age_cnt = 0.0, 0
    age_min: float | None = None
    age_max: float | None = None

    for i in range(n):
        try:
            did = int(disk_id_col[i].as_py()) if disk_id_col[i].as_py() is not None else None
        except (TypeError, ValueError):
            did = None
        day = _to_date_py(smart_day_col[i])
        day_ord = day.toordinal() if day is not None else None

        fs = first_seen.get(did) if did is not None else None
        a = _age_days(smart_day_col[i], fs)
        age_vals.append(a)
        if a is not None and math.isfinite(a):
            age_sum += a
            age_cnt += 1
            age_min = a if age_min is None else min(age_min, a)
            age_max = a if age_max is None else max(age_max, a)

        log1p_row_temporal: dict[str, float] = {}
        for ln in temporal.log1p_names:
            col = ln[6:] if ln.startswith("log1p_") else ln
            if col not in batch.schema.names:
                log1p_row_temporal[ln] = float("nan")
            else:
                log1p_row_temporal[ln] = _log1p_from_raw(batch.column(col)[i])

        for col in raw_features:
            ln = f"log1p_{col}"
            if col not in batch.schema.names:
                log1p_arrays[ln].append(float("nan"))
                continue
            lv = _log1p_from_raw(batch.column(col)[i])
            log1p_arrays[ln].append(lv)

        if did is not None and day_ord is not None:
            trow = temporal.update(did, day_ord, log1p_row_temporal)
        else:
            trow = temporal.nan_outputs()

        for tk in tkeys:
            temporal_cols[tk].append(trow.get(tk, float("nan")))

        if model_col is not None and model_mapping:
            m = model_col[i].as_py()
            s = str(m) if m is not None else ""
            codes.append(model_mapping.get(s, -1))
        else:
            codes.append(-1)

    age_days_arr = pa.array(age_vals, type=pa.float64())
    out_arrays = [batch.column(nm) for nm in names]
    out_names = list(names)
    out_arrays.append(age_days_arr)
    out_names.append("age_days")
    for ln in log1p_names_all:
        out_arrays.append(pa.array(log1p_arrays[ln], type=pa.float64()))
        out_names.append(ln)
    for tk in sorted(temporal_cols.keys()):
        out_arrays.append(pa.array(temporal_cols[tk], type=pa.float64()))
        out_names.append(tk)
    out_arrays.append(pa.array(codes, type=pa.int32()))
    out_names.append("model_code")

    out_batch = pa.RecordBatch.from_arrays(out_arrays, names=out_names)
    age_mean = age_sum / age_cnt if age_cnt else None
    return out_batch, age_min, age_max, age_mean, n


def run_sorted_partition_writer(
    partition_dir: Path,
    output_path: Path,
    first_seen: dict[int, date],
    raw_features: list[str],
    model_mapping: dict[str, int],
    temporal: TemporalState,
    batch_size: int,
    logger: logging.Logger,
    sort_partition: bool,
    row_cap_global: int | None,
    rows_written_global: list[int],
) -> tuple[int, float | None, float | None, float | None]:
    """
    Stream partition, optionally sort, write until global row cap (if set).
    rows_written_global is a one-element list total rows written across all partitions.
    Remaining allowance = row_cap_global - rows_written_global[0].
    """
    dataset = ds.dataset(partition_dir, format="parquet")
    if sort_partition:
        try:
            logger.info(
                "sort_by (smart_day, disk_id): materializing sorted partition — may use significant RAM/time: %s",
                partition_dir,
            )
            scan_ds = dataset.sort_by([("smart_day", "ascending"), ("disk_id", "ascending")])
        except Exception as e:
            logger.error("sort_by failed (%s); try --no_sort_partitions or more RAM", e)
            raise
    else:
        scan_ds = dataset
    scanner = scan_ds.scanner(batch_size=batch_size)
    writer: pq.ParquetWriter | None = None
    rows_out = 0
    age_min, age_max, age_sum, age_count = None, None, 0.0, 0

    for bi, batch in enumerate(scanner.to_batches()):
        if bi > 0 and bi % 50 == 0:
            logger.info(
                "Scan progress: batch_index=%d, rows_out_partition=%d, total_written=%d",
                bi,
                rows_out,
                rows_written_global[0],
            )
        if row_cap_global is not None:
            remaining = row_cap_global - rows_written_global[0]
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)

        out_batch, a_min, a_max, a_mean, n = process_sorted_batch(
            batch, first_seen, raw_features, model_mapping, temporal
        )
        if writer is None:
            writer = pq.ParquetWriter(output_path, out_batch.schema)
        writer.write_batch(out_batch)
        rows_out += n
        rows_written_global[0] += n
        if a_min is not None:
            age_min = a_min if age_min is None else min(age_min, a_min)
        if a_max is not None:
            age_max = a_max if age_max is None else max(age_max, a_max)
        if a_mean is not None and n > 0:
            age_sum += a_mean * n
            age_count += n
        if row_cap_global is not None and rows_written_global[0] >= row_cap_global:
            break

    if writer is not None:
        writer.close()
    age_mean = age_sum / age_count if age_count else None
    return rows_out, age_min, age_max, age_mean


def run_pipeline(
    base_dir: Path,
    experiment: str,
    split: str,
    max_partitions: int | None,
    batch_size: int,
    overwrite: bool,
    log_dir: Path,
    log_level: str,
    sort_partitions: bool,
) -> None:
    logger = setup_logging(log_dir, experiment, split, log_level)
    input_dir = base_dir / "data_splits" / experiment / split
    output_dir = base_dir / "data_engineered" / experiment / split

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if output_dir.exists() and not overwrite:
        raise RuntimeError(
            f"Output directory already exists: {output_dir}. Use --overwrite to overwrite."
        )

    partitions = find_partitions(input_dir)
    if not partitions:
        raise ValueError(f"No partitions (year=*/month=*) found under {input_dir}")

    if max_partitions is not None:
        partitions = partitions[:max_partitions]
        logger.info("Limiting to %d partitions (--max_partitions)", len(partitions))

    first_partition = partitions[0]
    schema = get_partition_schema(first_partition, batch_size)
    for col in REQUIRED_COLUMNS:
        if col not in schema.names:
            raise ValueError(f"Required column missing: {col}")

    config_path = base_dir / FEATURE_SET_PATH
    raw_features, norm_features, meta = load_feature_set(config_path, schema, logger)
    row_cap = _row_cap_for_split(split, meta)
    if row_cap is not None:
        logger.info("Row cap for split %s: %d (chronological stream)", split, row_cap)

    tcfg = meta.get("temporal") or {}
    roll_dispersion = str(tcfg.get("roll_dispersion", "std")).lower()
    inst = meta.get("instability") or {}
    inst_metric = str(inst.get("metric", "abs_delta1"))
    inst_prefix = str(inst.get("column_prefix", "instab_"))

    bases = _temporal_base_columns(raw_features, meta)
    bases = [b for b in bases if b in raw_features]
    log1p_names = [f"log1p_{b}" for b in bases]
    temporal_state = TemporalState(
        log1p_names=log1p_names,
        roll_dispersion=roll_dispersion,
        instability_metric=inst_metric,
        inst_prefix=inst_prefix,
    )

    engineered_cols = (
        ["age_days", "model_code"]
        + [f"log1p_{c}" for c in raw_features]
    )
    for ln in log1p_names:
        engineered_cols.extend(
            [
                f"delta1_{ln}",
                f"delta7_{ln}",
                f"rollmean7_{ln}",
                f"rollvar7_{ln}" if roll_dispersion == "var" else f"rollstd7_{ln}",
                f"{inst_prefix}{ln}",
            ]
        )

    is_train = split == "train"
    rows_written_global = [0]

    # Build first_seen from train, then merge it into val/test.
    if is_train:
        first_seen_dict, models_seen = pass0_collect_first_seen_and_models(
            partitions, batch_size, logger
        )
        save_disk_first_seen(base_dir, experiment, first_seen_dict, logger)
        model_mapping = {m: i for i, m in enumerate(sorted(models_seen), start=0)}
        save_model_mapping(experiment, model_mapping, logger)
    else:
        train_fs = load_disk_first_seen_train(base_dir, experiment, logger)
        first_seen_dict = pass0_merge_first_seen_non_train(
            partitions, batch_size, train_fs, logger
        )
        model_mapping = load_model_mapping(experiment, logger) or {}

    # Keep a plain-English definition of age_days for the report.
    age_note = (
        "age_days = (smart_day.date() - first_seen[disk]).days >= 0; "
        "first_seen is global min smart_day per disk over this split, "
        "merged with train JSON for val/test."
    )

    total_rows_read = 0
    total_rows_written = 0
    partition_stats: list[dict[str, Any]] = []
    age_days_min, age_days_max, age_days_sum, age_days_count = None, None, 0.0, 0

    for partition_dir in partitions:
        if row_cap is not None and rows_written_global[0] >= row_cap:
            logger.info("Row cap reached; skipping remaining partitions")
            break

        year_part = partition_dir.parent.name
        month_part = partition_dir.name
        partition_name = f"{year_part}/{month_part}"
        output_partition_dir = output_dir / year_part / month_part
        output_path = output_partition_dir / "data.parquet"

        try:
            partition_rows = int(ds.dataset(partition_dir, format="parquet").count_rows())
        except Exception:
            partition_rows = sum(
                b.num_rows
                for b in ds.dataset(partition_dir, format="parquet").scanner(batch_size=batch_size).to_batches()
            )
        total_rows_read += partition_rows

        output_partition_dir.mkdir(parents=True, exist_ok=True)
        rows_out, a_min, a_max, a_mean = run_sorted_partition_writer(
            partition_dir,
            output_path,
            first_seen_dict,
            raw_features,
            model_mapping,
            temporal_state,
            batch_size,
            logger,
            sort_partitions,
            row_cap,
            rows_written_global,
        )
        total_rows_written += rows_out
        if a_min is not None:
            age_days_min = a_min if age_days_min is None else min(age_days_min, a_min)
        if a_max is not None:
            age_days_max = a_max if age_days_max is None else max(age_days_max, a_max)
        if a_mean is not None and rows_out > 0:
            age_days_sum += a_mean * rows_out
            age_days_count += rows_out

        partition_stats.append({
            "partition": partition_name,
            "rows_read": partition_rows,
            "rows_written": rows_out,
            "age_days_min": a_min,
            "age_days_max": a_max,
            "age_days_mean": a_mean,
        })
        logger.info("%s: read %d, wrote %d (total written %s)", partition_name, partition_rows, rows_out, rows_written_global[0])

    age_days_mean = (age_days_sum / age_days_count) if age_days_count else None
    report = {
        "experiment": experiment,
        "split": split,
        "input_rows_scanned": total_rows_read,
        "output_rows_written": total_rows_written,
        "partitions_processed": len(partitions),
        "engineered_columns": engineered_cols,
        "age_days_min": age_days_min,
        "age_days_max": age_days_max,
        "age_days_mean": age_days_mean,
        "age_days_definition": age_note,
        "row_cap_applied": row_cap,
        "sort_partitions": sort_partitions,
        "partition_stats": partition_stats,
        "missingness_notes": "Compute per-feature missingness in post-processing if needed.",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fe_dir = REPORTS_DIR / "feature_engineering"
    fe_dir.mkdir(parents=True, exist_ok=True)
    report_path = fe_dir / f"feature_engineering_summary_{experiment}_{split}"
    with open(report_path.with_suffix(".json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    md_lines = [
        f"# Feature engineering summary – {experiment} / {split}",
        "",
        f"- **Rows read (partition totals):** {total_rows_read:,}",
        f"- **Rows written:** {total_rows_written:,}",
        f"- **Partitions processed:** {len(partitions)}",
        f"- **Engineered columns:** {len(engineered_cols)}",
        f"- **Row cap:** {row_cap}",
        f"- **Sort partitions (temporal correctness):** {sort_partitions}",
        f"- **age_days min / max / mean:** {age_days_min} / {age_days_max} / {age_days_mean}",
        "",
        "## age_days",
        "",
        age_note,
        "",
        "## Engineered columns",
        "",
    ] + [f"- {c}" for c in engineered_cols]
    with open(report_path.with_suffix(".md"), "w") as f:
        f.write("\n".join(md_lines))
    logger.info("Reports written to %s", report_path)
    logger.info("Done. Read %d, wrote %d", total_rows_read, total_rows_written)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-safe feature engineering for SMART SSD splits")
    parser.add_argument("--experiment", type=str, default="exp_time_generalisation")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--max_partitions", type=int, default=None, help="Limit partitions (e.g. 1 for debugging)")
    parser.add_argument("--batch_size", type=int, default=500_000, help="PyArrow scan batch size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument(
        "--no_sort_partitions",
        action="store_true",
        help="Do not sort by (smart_day, disk_id) inside each partition (faster/less RAM; temporal features may be wrong if input order is not time-ordered per disk).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parent.parent
    log_dir = base_dir / args.log_dir
    sort_partitions = not args.no_sort_partitions
    try:
        run_pipeline(
            base_dir=base_dir,
            experiment=args.experiment,
            split=args.split,
            max_partitions=args.max_partitions,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            log_dir=log_dir,
            log_level=args.log_level,
            sort_partitions=sort_partitions,
        )
    except Exception as e:
        logging.getLogger().exception("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
