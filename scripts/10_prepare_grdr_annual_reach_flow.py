"""Prepare reach-level annual mean discharge tables from GRDR regional NetCDF files.

This script is intended for server-side preprocessing before Figure 8 analysis.
It scans a folder of ``GRDR_v1.0.0_region_*.nc`` files, computes annual mean
discharge for every reach, and writes output tables that mirror the packaged
reach-discharge layout used elsewhere in the repository:

- one global table per year with ``COMID,qout``
- one long-term mean table with ``COMID,qout`` across the selected years

Optional per-region long tables can also be written for debugging or reuse.
The implementation streams data by year from each region file to avoid loading
the full ``time x reach`` array into memory at once.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date


DEFAULT_INPUT_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data/figure8/grdr")
DEFAULT_PATTERN = "GRDR_v1.0.0_region_*.nc"
DEFAULT_START_YEAR = 1985


@dataclass(frozen=True)
class RegionYearSummary:
    """Processing summary for one region-year block."""

    region_name: str
    source_file: str
    year: int
    num_reaches_total: int
    num_reaches_with_data: int
    num_days: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare reach-level GRDR annual mean discharge tables."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing all GRDR regional NetCDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for annual tables and metadata.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern used to discover regional NetCDF files.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="First calendar year to retain.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last calendar year to retain. Defaults to the latest available year.",
    )
    parser.add_argument(
        "--write-per-region",
        action="store_true",
        help="Also write per-region long tables with columns COMID,year,qout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    parser.add_argument(
        "--reach-chunk-size",
        type=int,
        default=5000,
        help=(
            "Number of reaches to process per read chunk. "
            "Increase on high-memory nodes to improve throughput."
        ),
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        default=1,
        help="Gzip compression level for output CSVs (0-9). Lower is faster.",
    )
    return parser.parse_args()


def infer_region_name(path: Path) -> str:
    """Infer a short region name from the NetCDF filename."""
    match = re.search(r"region[_-]?(\d+)", path.stem, re.IGNORECASE)
    if match:
        return f"region_{int(match.group(1))}"
    return path.stem


def ensure_clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create the output directory or fail if it already contains files."""
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite or choose a new path."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annual_tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "per_region").mkdir(parents=True, exist_ok=True)


def append_csv_gz(frame: pd.DataFrame, path: Path, compresslevel: int = 1) -> None:
    """Append a frame to a gzipped CSV, writing the header only once."""
    if frame.empty:
        return
    header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="at", newline="", compresslevel=compresslevel) as handle:
        frame.to_csv(handle, index=False, header=header)


def write_csv_gz(frame: pd.DataFrame, path: Path, compresslevel: int = 1) -> None:
    """Write a dataframe to a gzipped CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="wt", newline="", compresslevel=compresslevel) as handle:
        frame.to_csv(handle, index=False)


def iter_region_files(input_dir: Path, pattern: str) -> list[Path]:
    """List GRDR region NetCDF files."""
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No regional GRDR NetCDF files found in {input_dir} with pattern {pattern!r}"
        )
    return files


def standardized_grdr_bundle_exists(output_dir: Path) -> bool:
    """Return True when a standardized GRDR annual-table bundle already exists."""
    annual_dir = output_dir / "annual_tables"
    metadata_json = output_dir / "GRDR_processing_metadata.json"
    return annual_dir.exists() and any(annual_dir.glob("GRDR_annual_mean_*.csv.gz")) and metadata_json.exists()


def build_time_index(nc: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Decode the time coordinate and return datetimes and years."""
    time_var = nc.variables["time"]
    time_values = time_var[:]
    dates = num2date(
        time_values,
        units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard"),
    )
    years = np.fromiter((dt.year for dt in dates), dtype=np.int32, count=len(dates))
    return np.asarray(dates), years


def selected_years_from_array(
    years: np.ndarray,
    start_year: int,
    end_year: int | None,
) -> list[int]:
    """Return the sorted years to process for one region file."""
    available = sorted({int(year) for year in years})
    if end_year is None:
        return [year for year in available if year >= start_year]
    return [year for year in available if start_year <= year <= end_year]


def build_year_slices(years: np.ndarray, selected_years: Iterable[int]) -> dict[int, slice]:
    """Map each year to a contiguous time slice in the NetCDF file."""
    year_slices: dict[int, slice] = {}
    for year in selected_years:
        indices = np.flatnonzero(years == year)
        if indices.size == 0:
            continue
        year_slices[year] = slice(int(indices[0]), int(indices[-1]) + 1)
    return year_slices


def compute_block_mean(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sums, valid counts, and mean values along the time axis."""
    valid_counts = np.isfinite(block).sum(axis=0, dtype=np.int32)
    sums = np.nansum(block, axis=0, dtype=np.float64)
    means = np.full(block.shape[1], np.nan, dtype=np.float64)
    valid_mask = valid_counts > 0
    means[valid_mask] = sums[valid_mask] / valid_counts[valid_mask]
    return sums, valid_counts, means


def append_per_region_frame(path: Path, frame: pd.DataFrame, compresslevel: int = 1) -> None:
    """Append a per-region long-format frame to a gzipped CSV."""
    if frame.empty:
        return
    header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="at", newline="", compresslevel=compresslevel) as handle:
        frame.to_csv(handle, index=False, header=header)


def process_region_file(
    path: Path,
    output_dir: Path,
    start_year: int,
    end_year: int | None,
    write_per_region: bool,
    reach_chunk_size: int,
    gzip_level: int,
) -> tuple[list[RegionYearSummary], pd.DataFrame, dict[str, object]]:
    """Process a single regional GRDR NetCDF file."""
    region_name = infer_region_name(path)
    summaries: list[RegionYearSummary] = []

    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        nc.set_auto_maskandscale(False)
        _, years = build_time_index(nc)
        selected_years = selected_years_from_array(years, start_year, end_year)
        year_slices = build_year_slices(years, selected_years)
        reach_ids = np.asarray(nc.variables["reach"][:], dtype=np.int64)
        discharge_var = nc.variables["discharge"]
        chunking = discharge_var.chunking()
        region_per_year_counts = {year: 0 for year in selected_years}
        region_per_year_days = {
            year: int(year_slices[year].stop - year_slices[year].start)
            for year in selected_years
            if year in year_slices
        }
        long_term_frames: list[pd.DataFrame] = []
        per_region_path = output_dir / "per_region" / f"{region_name}_annual_mean_long.csv.gz"
        if write_per_region and per_region_path.exists():
            per_region_path.unlink()

        total_reaches = reach_ids.size
        num_chunks = (total_reaches + reach_chunk_size - 1) // reach_chunk_size
        annual_paths = {
            year: output_dir / "annual_tables" / f"GRDR_annual_mean_{year}.csv.gz"
            for year in selected_years
        }
        annual_headers = {year: not annual_paths[year].exists() for year in selected_years}
        annual_handles = {}
        region_handle = None
        try:
            for year, annual_path in annual_paths.items():
                annual_path.parent.mkdir(parents=True, exist_ok=True)
                annual_handles[year] = gzip.open(
                    annual_path,
                    mode="at",
                    newline="",
                    compresslevel=gzip_level,
                )

            if write_per_region:
                per_region_path.parent.mkdir(parents=True, exist_ok=True)
                region_handle = gzip.open(
                    per_region_path,
                    mode="at",
                    newline="",
                    compresslevel=gzip_level,
                )
                region_header = not per_region_path.exists() or per_region_path.stat().st_size == 0
            else:
                region_header = False

            for chunk_index, start in enumerate(range(0, total_reaches, reach_chunk_size), start=1):
                stop = min(start + reach_chunk_size, total_reaches)
                print(
                    f"Processing {region_name}: reach chunk {chunk_index}/{num_chunks} "
                    f"({start}:{stop})",
                    flush=True,
                )
                block = np.asarray(discharge_var[:, start:stop], dtype=np.float32)
                chunk_reaches = reach_ids[start:stop]

                long_term_sums_chunk = np.zeros(chunk_reaches.size, dtype=np.float64)
                long_term_counts_chunk = np.zeros(chunk_reaches.size, dtype=np.int64)

                for year in selected_years:
                    year_slice = year_slices.get(year)
                    if year_slice is None:
                        continue
                    year_block = block[year_slice, :]
                    sums, valid_counts, means = compute_block_mean(year_block)
                    long_term_sums_chunk += sums
                    long_term_counts_chunk += valid_counts

                    frame = pd.DataFrame({"COMID": chunk_reaches, "qout": means})
                    frame = frame.dropna(subset=["qout"]).reset_index(drop=True)
                    region_per_year_counts[year] += int(frame.shape[0])
                    if not frame.empty:
                        frame.to_csv(
                            annual_handles[year],
                            index=False,
                            header=annual_headers[year],
                        )
                        annual_headers[year] = False

                    if write_per_region and not frame.empty:
                        region_frame = frame.assign(year=year)[["COMID", "year", "qout"]]
                        region_frame.to_csv(
                            region_handle,
                            index=False,
                            header=region_header,
                        )
                        region_header = False

                chunk_long_term_means = np.full(chunk_reaches.size, np.nan, dtype=np.float64)
                valid_mask = long_term_counts_chunk > 0
                chunk_long_term_means[valid_mask] = (
                    long_term_sums_chunk[valid_mask] / long_term_counts_chunk[valid_mask]
                )
                chunk_long_term_frame = pd.DataFrame(
                    {"COMID": chunk_reaches, "qout": chunk_long_term_means}
                )
                chunk_long_term_frame = chunk_long_term_frame.dropna(subset=["qout"]).reset_index(
                    drop=True
                )
                long_term_frames.append(chunk_long_term_frame)
        finally:
            for handle in annual_handles.values():
                handle.close()
            if region_handle is not None:
                region_handle.close()

        for year in selected_years:
            summaries.append(
                RegionYearSummary(
                    region_name=region_name,
                    source_file=path.name,
                    year=year,
                    num_reaches_total=int(total_reaches),
                    num_reaches_with_data=int(region_per_year_counts.get(year, 0)),
                    num_days=int(region_per_year_days.get(year, 0)),
                )
            )

    long_term_frame = (
        pd.concat(long_term_frames, ignore_index=True)
        if long_term_frames
        else pd.DataFrame(columns=["COMID", "qout"])
    )

    region_metadata = {
        "region_name": region_name,
        "source_file": path.name,
        "num_reaches": int(reach_ids.size),
        "available_years": sorted({int(y) for y in years}),
        "selected_years": selected_years,
        "discharge_chunking": chunking,
        "reach_chunk_size": int(reach_chunk_size),
        "gzip_level": int(gzip_level),
    }
    return summaries, long_term_frame, region_metadata


def write_metadata(
    output_dir: Path,
    *,
    input_dir: Path,
    pattern: str,
    start_year: int,
    end_year: int | None,
    region_files: list[Path],
    inventory_df: pd.DataFrame,
    region_metadata: list[dict[str, object]],
) -> None:
    """Write preprocessing metadata files."""
    if inventory_df.empty:
        actual_years: list[int] = []
    else:
        actual_years = sorted(inventory_df["year"].astype(int).unique().tolist())

    metadata = {
        "product": "GRDR",
        "variable": "discharge",
        "aggregation": "calendar_year_mean",
        "input_dir": str(input_dir),
        "pattern": pattern,
        "requested_start_year": start_year,
        "requested_end_year": end_year,
        "actual_years": actual_years,
        "num_region_files": len(region_files),
        "region_files": [path.name for path in region_files],
        "outputs": {
            "annual_tables_dir": "annual_tables",
            "global_long_term_mean_csv_gz": "GRDR_long_term_mean_selected_years.csv.gz",
            "inventory_csv": "GRDR_processing_inventory.csv",
            "regions_json": "GRDR_processing_regions.json",
        },
    }
    (output_dir / "GRDR_processing_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "GRDR_processing_regions.json").write_text(
        json.dumps(region_metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Run the GRDR annual-mean preprocessing workflow."""
    args = parse_args()
    try:
        region_files = iter_region_files(args.input_dir, args.pattern)
    except FileNotFoundError:
        if args.input_dir == DEFAULT_INPUT_DIR and standardized_grdr_bundle_exists(DEFAULT_OUTPUT_DIR):
            print(
                "No raw GRDR regional NetCDF files were found under data/. "
                "The standardized annual GRDR bundle already exists under "
                f"{DEFAULT_OUTPUT_DIR}; nothing to rebuild.",
                flush=True,
            )
            return
        raise
    ensure_clean_output_dir(args.output_dir, overwrite=args.overwrite)

    inventory_records: list[dict[str, object]] = []
    region_records: list[dict[str, object]] = []
    global_long_term_frames: list[pd.DataFrame] = []

    for index, region_file in enumerate(region_files, start=1):
        print(f"[{index}/{len(region_files)}] {region_file.name}", flush=True)
        summaries, long_term_frame, region_metadata = process_region_file(
            region_file,
            output_dir=args.output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            write_per_region=args.write_per_region,
            reach_chunk_size=args.reach_chunk_size,
            gzip_level=args.gzip_level,
        )
        inventory_records.extend(summary.__dict__ for summary in summaries)
        region_records.append(region_metadata)
        global_long_term_frames.append(long_term_frame)

    inventory_df = pd.DataFrame(inventory_records)
    inventory_path = args.output_dir / "GRDR_processing_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)

    if global_long_term_frames:
        global_long_term_df = pd.concat(global_long_term_frames, ignore_index=True)
        global_long_term_df = (
            global_long_term_df.groupby("COMID", as_index=False)["qout"].first()
            .sort_values("COMID")
            .reset_index(drop=True)
        )
    else:
        global_long_term_df = pd.DataFrame(columns=["COMID", "qout"])

    write_csv_gz(
        global_long_term_df,
        args.output_dir / "GRDR_long_term_mean_selected_years.csv.gz",
        compresslevel=args.gzip_level,
    )

    write_metadata(
        args.output_dir,
        input_dir=args.input_dir,
        pattern=args.pattern,
        start_year=args.start_year,
        end_year=args.end_year,
        region_files=region_files,
        inventory_df=inventory_df,
        region_metadata=region_records,
    )

    actual_years = (
        sorted(inventory_df["year"].astype(int).unique().tolist())
        if not inventory_df.empty
        else []
    )
    print("\nFinished GRDR preprocessing.", flush=True)
    print(f"Processed region files: {len(region_files)}", flush=True)
    print(f"Actual output years: {actual_years}", flush=True)
    print(f"Annual tables: {args.output_dir / 'annual_tables'}", flush=True)
    print(
        "Long-term mean: "
        f"{args.output_dir / 'GRDR_long_term_mean_selected_years.csv.gz'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
