"""Gridded precipitation workflows backed by per-basin upstream-total CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyogrio

from .basin_io import BasinPaths, iter_global_basin_paths
from .continuous_source import (
    calculate_basin_centroid_from_tables,
    prepare_river_dataframe,
    read_river_attributes,
)
from .mainstem import DEFAULT_UP_COLS, build_river_maps
from .multilevel import (
    DEFAULT_CATCHMENT_MASS_SUPERSAMPLING_FACTOR,
    UpstreamMassResolver,
    compute_catchment_masses,
    load_gridded_grid,
)


PRECIPITATION_UPSTREAM_SCHEMA_VERSION = 2
PRECIPITATION_UPSTREAM_METHOD = "supersampled_area_weighted_v1"


def _infer_data_type(gridded_data_path: str | Path) -> str:
    suffix = Path(gridded_data_path).suffix.lower()
    if suffix == ".nc":
        return "nc"
    return "tif"


def _precipitation_upstream_csv_path(basin: BasinPaths) -> Path:
    return basin.basin_dir / "mswep_precipitation_upstream_totals.csv"


def _precipitation_upstream_meta_path(basin: BasinPaths) -> Path:
    return basin.basin_dir / "mswep_precipitation_upstream_totals.meta.json"


def _expected_precipitation_upstream_meta(
    gridded_data_path: str | Path,
    variable: str,
    reduction: str,
) -> dict:
    return {
        "schema_version": PRECIPITATION_UPSTREAM_SCHEMA_VERSION,
        "aggregation_method": PRECIPITATION_UPSTREAM_METHOD,
        "supersample_factor": DEFAULT_CATCHMENT_MASS_SUPERSAMPLING_FACTOR,
        "area_weighted": True,
        "source_file": str(Path(gridded_data_path).resolve()),
        "variable": variable,
        "reduction": reduction,
    }


def read_precipitation_upstream_dataframe(
    basin: BasinPaths,
    comid_col: str = "COMID",
) -> Optional[pd.DataFrame]:
    """Read a previously materialized COMID -> precipitation-total table if present."""
    csv_path = basin.precipitation_upstream_csv or _precipitation_upstream_csv_path(basin)
    if not csv_path.exists():
        return None

    precip_df = pd.read_csv(csv_path, usecols=[comid_col, "p_local", "p_upstream_total"])
    precip_df[comid_col] = pd.to_numeric(precip_df[comid_col], errors="coerce").astype(int)
    precip_df["p_local"] = pd.to_numeric(precip_df["p_local"], errors="coerce").fillna(0.0)
    precip_df["p_upstream_total"] = (
        pd.to_numeric(precip_df["p_upstream_total"], errors="coerce").fillna(0.0)
    )
    return precip_df


def write_precipitation_upstream_dataframe(
    basin: BasinPaths,
    precip_df: pd.DataFrame,
    gridded_data_path: str | Path,
    variable: str,
    reduction: str,
    comid_col: str = "COMID",
) -> Path:
    """Persist COMID -> precipitation totals next to the packaged basin inputs."""
    csv_path = _precipitation_upstream_csv_path(basin)
    meta_path = _precipitation_upstream_meta_path(basin)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    precip_df.loc[:, [comid_col, "p_local", "p_upstream_total"]].to_csv(csv_path, index=False)
    meta = {
        **_expected_precipitation_upstream_meta(
            gridded_data_path=gridded_data_path,
            variable=variable,
            reduction=reduction,
        ),
        "row_count": int(len(precip_df)),
        "local_mass_unit": f"{variable} * km2",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path


def build_precipitation_upstream_dataframe(
    basin: BasinPaths,
    gridded_data_path: str | Path,
    variable: str = "precipitation",
    reduction: str = "mean",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    grid: Optional[dict] = None,
    river_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a fresh in-memory reach table with local and upstream precipitation totals."""
    _, precip_df = prepare_basin_precipitation_tables(
        basin=basin,
        gridded_data_path=gridded_data_path,
        variable=variable,
        reduction=reduction,
        comid_col=comid_col,
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
        grid=grid,
        river_df=river_df,
    )
    return precip_df


def prepare_basin_precipitation_tables(
    basin: BasinPaths,
    gridded_data_path: str | Path,
    variable: str = "precipitation",
    reduction: str = "mean",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    grid: Optional[dict] = None,
    river_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare a normalized river table and COMID -> upstream precipitation totals."""
    if basin.catchments is None:
        raise FileNotFoundError(f"No catchment shapefile found for basin {basin.name}")
    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    if grid is None:
        grid = load_gridded_grid(
            gridded_data_path=gridded_data_path,
            variable=variable,
            reduction=reduction,
        )
    if river_df is None:
        river_df = read_river_attributes(
            basin.river_network,
            columns=[comid_col, length_col, uparea_col, *up_cols],
        )
    river_df = prepare_river_dataframe(
        river_df,
        comid_col=comid_col,
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
    )
    basin_comids = set(river_df[comid_col].astype(int))

    catchment_gdf = pyogrio.read_dataframe(basin.catchments, columns=[comid_col])
    catchment_gdf[comid_col] = pd.to_numeric(catchment_gdf[comid_col], errors="coerce").astype(int)
    catchment_gdf = catchment_gdf[catchment_gdf[comid_col].isin(basin_comids)].copy()

    local_masses = compute_catchment_masses(
        catchment_gdf,
        grid,
        supersample_factor=DEFAULT_CATCHMENT_MASS_SUPERSAMPLING_FACTOR,
    )
    _, upstream_map = build_river_maps(
        river_df,
        comid_col=comid_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
    )
    resolver = UpstreamMassResolver(upstream_map)

    rows = []
    for comid in sorted(basin_comids):
        p_local = float(local_masses.get(comid, 0.0))
        p_upstream_total = float(resolver.upstream_mass(comid, local_masses))
        rows.append(
            {
                comid_col: comid,
                "p_local": p_local,
                "p_upstream_total": p_upstream_total,
            }
        )

    precip_df = pd.DataFrame(rows)
    return river_df, precip_df


def ensure_precipitation_upstream_dataframe(
    basin: BasinPaths,
    gridded_data_path: str | Path,
    variable: str = "precipitation",
    reduction: str = "mean",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    grid: Optional[dict] = None,
    river_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return a normalized river table and a persisted precipitation-total table.

    The packaged CSV is treated as a first-class basin input. If it already
    exists, workflows use it directly. The gridded field is consulted only when
    the table is missing and needs to be materialized from catchments.
    """
    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    if river_df is None:
        river_df = read_river_attributes(
            basin.river_network,
            columns=[comid_col, length_col, uparea_col, *up_cols],
        )
    river_df = prepare_river_dataframe(
        river_df,
        comid_col=comid_col,
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
    )

    precip_df = read_precipitation_upstream_dataframe(basin=basin, comid_col=comid_col)

    if precip_df is None:
        _, precip_df = prepare_basin_precipitation_tables(
            basin=basin,
            gridded_data_path=gridded_data_path,
            variable=variable,
            reduction=reduction,
            comid_col=comid_col,
            length_col=length_col,
            uparea_col=uparea_col,
            up_cols=up_cols,
            grid=grid,
            river_df=river_df,
        )
        write_precipitation_upstream_dataframe(
            basin=basin,
            precip_df=precip_df,
            gridded_data_path=gridded_data_path,
            variable=variable,
            reduction=reduction,
            comid_col=comid_col,
        )

    return river_df, precip_df


def run_single_basin_precipitation_rci(
    basin: BasinPaths,
    gridded_data_path: str | Path,
    output_path: str | Path,
    variable: Optional[str] = "precipitation",
    data_type: Optional[str] = None,
    reduction: str = "mean",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    min_segments: int = 3,
    grid: Optional[dict] = None,
) -> pd.DataFrame:
    """Run basin-scale precipitation RCI using the persisted upstream-total CSV."""
    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    river_df, precip_df = ensure_precipitation_upstream_dataframe(
        basin=basin,
        gridded_data_path=gridded_data_path,
        variable=variable or "precipitation",
        reduction=reduction,
        comid_col=comid_col,
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
        grid=grid,
    )
    precip_df = precip_df[[comid_col, "p_upstream_total"]]

    result_df = calculate_basin_centroid_from_tables(
        river_df=river_df,
        q_df=precip_df,
        output_path=output_path,
        basin_name=basin.name,
        q_col="p_upstream_total",
        comid_col=comid_col,
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
        min_segments=min_segments,
        total_column_name="total_mass",
        min_total=0.0,
        print_header=True,
        river_prepared=True,
        q_prepared=True,
    )
    result_df.insert(1, "river_file", basin.river_network.name)
    result_df.insert(2, "source_variable", variable or "precipitation")
    result_df.insert(3, "data_type", data_type or _infer_data_type(gridded_data_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    return result_df


def run_batch_basin_precipitation_level0(
    global_dir: str = "data/basins/global",
    gridded_data_path: str = "data/climate/climatology/mswep_precipitation_mean.nc",
    output_dir: str = "outputs/batch/precipitation_level0",
    variable: Optional[str] = "precipitation",
    data_type: Optional[str] = None,
    reduction: str = "mean",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    min_segments: int = 3,
    basin_names: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run basin-scale precipitation RCI across packaged basins."""
    output_dir_path = Path(output_dir)
    per_basin_dir = output_dir_path / "per_basin"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    per_basin_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    grid = None
    for basin in iter_global_basin_paths(global_dir, basin_names=basin_names):
        output_path = per_basin_dir / f"{basin.name}_precipitation_rci.csv"
        try:
            if basin.precipitation_upstream_csv is None or not basin.precipitation_upstream_csv.exists():
                if grid is None:
                    grid = load_gridded_grid(
                        gridded_data_path=gridded_data_path,
                        variable=variable or "precipitation",
                        reduction=reduction,
                    )
            basin_df = run_single_basin_precipitation_rci(
                basin=basin,
                gridded_data_path=gridded_data_path,
                output_path=output_path,
                variable=variable,
                data_type=data_type,
                reduction=reduction,
                comid_col=comid_col,
                length_col=length_col,
                uparea_col=uparea_col,
                up_cols=up_cols,
                min_segments=min_segments,
                grid=grid,
            )
            basin_df.insert(1, "source_mode", basin.mode)
            results.append(basin_df)
        except Exception as exc:
            failures.append(
                {
                    "basin_name": basin.name,
                    "river_file": basin.river_network.name,
                    "error": str(exc),
                }
            )

    combined_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    combined_df.to_csv(output_dir_path / "global_precipitation_level0_results.csv", index=False)
    failures_df.to_csv(output_dir_path / "global_precipitation_level0_failures.csv", index=False)
    return combined_df, failures_df
