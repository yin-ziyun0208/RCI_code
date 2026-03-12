"""Prepare standardized Figure 8 inputs from GRADES, GRDR, and MSWEP.

This script builds a single Figure 8 input bundle under ``data/figure8``:

- ``grades_annual/annual_tables/GRADES_annual_mean_<YEAR>.csv.gz``
- ``mswep_annual_<START>_<END>_clean.nc``
- ``figure8_input_manifest.json``

The script does not duplicate the preprocessed GRDR annual tables. Instead,
the manifest records the canonical GRDR directory and the common analysis years
shared by all three products.

This repository keeps the standardized Figure 8 inputs, not the full raw source
files. To rebuild the bundle from raw GRADES/MSWEP sources, pass
``--grades-pattern`` and ``--mswep-path`` explicitly.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_OUTPUT_DIR = Path("data/figure8")
DEFAULT_GRDR_DIR = Path("data/figure8/grdr")
DEFAULT_START_YEAR = 1985
DEFAULT_END_YEAR = 2018


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardize Figure 8 annual inputs from GRADES, GRDR, and MSWEP."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--grades-pattern",
        default=None,
        help="Glob pattern for raw GRADES annual-index NetCDF files. Required only when rebuilding.",
    )
    parser.add_argument("--grdr-dir", type=Path, default=DEFAULT_GRDR_DIR)
    parser.add_argument(
        "--mswep-path",
        type=Path,
        default=None,
        help="Path to the raw merged MSWEP annual file. Required only when rebuilding.",
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def list_grades_files(pattern: str | None) -> list[Path]:
    if not pattern:
        raise FileNotFoundError(
            "Raw GRADES files are not bundled locally. Provide --grades-pattern to rebuild Figure 8 inputs."
        )
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No GRADES annual-index files found with pattern: {pattern}")
    return files


def load_grdr_years(grdr_dir: Path) -> list[int]:
    metadata_path = grdr_dir / "GRDR_processing_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"GRDR metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    years = metadata.get("actual_years", [])
    if not years:
        raise ValueError(f"No GRDR years recorded in {metadata_path}")
    return [int(year) for year in years]


def select_mswep_annual_slices(mswep_path: Path, start_year: int, end_year: int) -> xr.Dataset:
    if mswep_path is None:
        raise FileNotFoundError(
            "Raw MSWEP annual file is not bundled locally. Provide --mswep-path to rebuild Figure 8 inputs."
        )
    ds = xr.open_dataset(mswep_path)
    time_values = pd.to_datetime(ds["time"].values)

    if "time_bnds" not in ds:
        raise ValueError(f"MSWEP file does not contain time_bnds: {mswep_path}")

    bounds_array = np.asarray(ds["time_bnds"].values)
    start_bounds = pd.to_datetime(bounds_array[:, 0])
    end_bounds = pd.to_datetime(bounds_array[:, 1])
    keep_indices: list[int] = []

    for index, timestamp in enumerate(time_values):
        start_bound = start_bounds[index]
        end_bound = end_bounds[index]
        year = int(timestamp.year)
        is_single_year = int(start_bound.year) == int(end_bound.year) == year
        if not is_single_year:
            continue
        if start_year <= year <= end_year:
            keep_indices.append(index)

    if not keep_indices:
        ds.close()
        raise ValueError("No valid annual MSWEP slices selected for the requested year range.")

    subset = ds.isel(time=keep_indices).copy()
    selected_years = pd.to_datetime(subset["time"].values).year.tolist()
    if len(selected_years) != len(set(selected_years)):
        ds.close()
        raise ValueError("Duplicate years remain in the cleaned MSWEP annual series.")

    subset.attrs["cleaning_rule"] = (
        "Retain only slices where time_bnds start/end fall within the same year as the "
        "time coordinate; this removes decade-mean slices accidentally merged into the file."
    )
    subset.attrs["selected_years"] = ",".join(str(year) for year in selected_years)
    ds.close()
    return subset


def write_mswep_clean_file(
    mswep_path: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> tuple[Path, list[int]]:
    output_path = output_dir / f"mswep_annual_{start_year}_{end_year}_clean.nc"
    if output_path.exists() and not overwrite:
        with xr.open_dataset(output_path) as ds:
            years = pd.to_datetime(ds["time"].values).year.tolist()
        return output_path, [int(year) for year in years]

    cleaned = select_mswep_annual_slices(mswep_path, start_year, end_year)
    years = pd.to_datetime(cleaned["time"].values).year.tolist()
    cleaned.to_netcdf(output_path)
    cleaned.close()
    return output_path, [int(year) for year in years]


def extract_grades_annual_tables(
    grades_files: list[Path],
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> tuple[Path, list[int]]:
    annual_dir = output_dir / "grades_annual" / "annual_tables"
    annual_dir.mkdir(parents=True, exist_ok=True)

    years_to_write = list(range(start_year, end_year + 1))
    years_requiring_write: list[int] = []
    written_years: list[int] = []
    frames_by_year: dict[int, list[pd.DataFrame]] = {year: [] for year in years_to_write}

    for year in years_to_write:
        output_path = annual_dir / f"GRADES_annual_mean_{year}.csv.gz"
        if output_path.exists() and not overwrite:
            written_years.append(year)
        else:
            years_requiring_write.append(year)

    if years_requiring_write:
        for path in grades_files:
            with xr.open_dataset(path) as ds:
                time_values = ds["time"].values.astype(int)
                year_to_index = {int(year): int(index) for index, year in enumerate(time_values)}
                missing_years = [year for year in years_requiring_write if year not in year_to_index]
                if missing_years:
                    missing_text = ", ".join(str(year) for year in missing_years)
                    raise ValueError(f"Missing GRADES years in {path.name}: {missing_text}")

                comids = ds["COMID"].values.astype(np.int64)
                qmean_all = np.asarray(ds["QMEAN"].values, dtype=np.float64)
                for year in years_requiring_write:
                    year_index = year_to_index[year]
                    frames_by_year[year].append(
                        pd.DataFrame({"COMID": comids, "qout": qmean_all[year_index, :]})
                    )

        for year in years_requiring_write:
            output_path = annual_dir / f"GRADES_annual_mean_{year}.csv.gz"
            year_df = pd.concat(frames_by_year[year], ignore_index=True)
            year_df = year_df.sort_values("COMID").reset_index(drop=True)
            with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
                year_df.to_csv(handle, index=False)
            written_years.append(year)
            print(f"Wrote GRADES annual table: {output_path}", flush=True)

    written_years = sorted(written_years)

    inventory_rows: list[dict[str, object]] = []
    for year in written_years:
        output_path = annual_dir / f"GRADES_annual_mean_{year}.csv.gz"
        inventory_rows.append(
            {
                "year": year,
                "path": str(output_path),
                "exists": output_path.exists(),
            }
        )
    inventory_path = output_dir / "grades_annual" / "GRADES_processing_inventory.csv"
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    return annual_dir, written_years


def write_manifest(
    output_dir: Path,
    grades_dir: Path,
    grades_years: list[int],
    grdr_dir: Path,
    grdr_years: list[int],
    mswep_clean_path: Path,
    mswep_years: list[int],
) -> Path:
    common_years = sorted(set(grades_years) & set(grdr_years) & set(mswep_years))
    manifest = {
        "figure": "Figure 8",
        "q_nat": {
            "product": "GRADES",
            "annual_tables_dir": str(grades_dir),
            "years": grades_years,
            "filename_pattern": "GRADES_annual_mean_<YEAR>.csv.gz",
        },
        "q_hum": {
            "product": "GRDR",
            "annual_tables_dir": str(grdr_dir / "annual_tables"),
            "metadata_json": str(grdr_dir / "GRDR_processing_metadata.json"),
            "years": grdr_years,
            "filename_pattern": "GRDR_annual_mean_<YEAR>.csv.gz",
        },
        "precipitation": {
            "product": "MSWEP",
            "annual_clean_nc": str(mswep_clean_path),
            "years": mswep_years,
            "variable": "precipitation",
            "units": "mm/d",
        },
        "analysis_common_years": common_years,
        "analysis_start_year": common_years[0] if common_years else None,
        "analysis_end_year": common_years[-1] if common_years else None,
    }
    manifest_path = output_dir / "figure8_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grdr_years = load_grdr_years(args.grdr_dir)

    mswep_clean_path = args.output_dir / f"mswep_annual_{DEFAULT_START_YEAR}_{DEFAULT_END_YEAR}_clean.nc"
    grades_dir = args.output_dir / "grades_annual" / "annual_tables"
    manifest_path = args.output_dir / "figure8_input_manifest.json"

    rebuild_needed = args.overwrite or (not mswep_clean_path.exists()) or (not grades_dir.exists())
    if rebuild_needed:
        grades_files = list_grades_files(args.grades_pattern)

        grades_year_min = max(args.start_year, 1980)
        grades_year_max = min(args.end_year, 2019)
        grdr_year_min = min(grdr_years)
        grdr_year_max = max(grdr_years)
        common_start = max(grades_year_min, grdr_year_min)
        common_end = min(grades_year_max, grdr_year_max)
        if common_start > common_end:
            raise ValueError("No overlapping years across GRADES and GRDR inputs.")

        mswep_clean_path, mswep_years = write_mswep_clean_file(
            mswep_path=args.mswep_path,
            output_dir=args.output_dir,
            start_year=common_start,
            end_year=common_end,
            overwrite=args.overwrite,
        )

        grades_dir, grades_years = extract_grades_annual_tables(
            grades_files=grades_files,
            output_dir=args.output_dir,
            start_year=common_start,
            end_year=common_end,
            overwrite=args.overwrite,
        )

        manifest_path = write_manifest(
            output_dir=args.output_dir,
            grades_dir=grades_dir,
            grades_years=grades_years,
            grdr_dir=args.grdr_dir,
            grdr_years=grdr_years,
            mswep_clean_path=mswep_clean_path,
            mswep_years=mswep_years,
        )
    else:
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Figure 8 standardized inputs exist but manifest is missing. Re-run with "
                "--grades-pattern and --mswep-path to rebuild."
            )

    print("\nFigure 8 standard inputs ready:")
    print(f"- GRADES annual tables: {grades_dir}")
    print(f"- MSWEP clean annual file: {mswep_clean_path}")
    print(f"- GRDR annual tables: {args.grdr_dir / 'annual_tables'}")
    print(f"- Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
