"""Build Figure 8 annual RCI time series for Q_nat, Q_hum, and precipitation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import pyogrio
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.basin_io import BasinPaths, iter_global_basin_paths
from core.centroid import calculate_midpoint_centroid, calculate_rci, find_centroid_comid
from core.continuous_source import prepare_river_dataframe, read_river_attributes
from core.mainstem import DEFAULT_UP_COLS, build_river_maps, find_basin_outlet, trace_main_stem_with_maps
from core.multilevel import (
    DEFAULT_CATCHMENT_MASS_SUPERSAMPLING_FACTOR,
    UpstreamMassResolver,
    _coord_edges,
    _reproject_catchments_to_grid,
    _row_pixel_areas_km2,
    _select_crop_indices,
    _supersampled_transform,
)


MANIFEST_PATH = Path("data/figure8/figure8_input_manifest.json")
RESULT_DIR = Path("results/figure8")

Q_NAT_CSV = RESULT_DIR / "figure8_global_qnat_annual_rci.csv"
Q_HUM_CSV = RESULT_DIR / "figure8_global_qhum_annual_rci.csv"
PRECIP_CSV = RESULT_DIR / "figure8_global_precipitation_annual_rci.csv"
COMBINED_CSV = RESULT_DIR / "figure8_global_annual_rci_all.csv"
FAILURES_CSV = RESULT_DIR / "figure8_global_annual_rci_failures.csv"


@dataclass
class BasinState:
    basin: BasinPaths
    river_df: pd.DataFrame
    comid_list: list[int]
    comid_set: set[int]
    length_dict: dict[int, float]
    upstream_map: dict[int, list[int]]
    outlet_comid: int
    outlet_uparea_km2: float
    mainstem_comids: list[int]
    segment_lengths: list[float]
    cumulative_lengths: list[float]


@dataclass
class BasinCatchmentWeights:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    unique_labels: np.ndarray
    inverse_indices: np.ndarray
    parent_flat_indices: np.ndarray
    subcell_areas_km2: np.ndarray


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _prepare_basin_states(global_dir: str = "data/basins/global") -> tuple[list[BasinState], dict[int, str]]:
    basin_states: list[BasinState] = []
    comid_to_basin: dict[int, str] = {}

    for basin in iter_global_basin_paths(global_dir):
        river_df = read_river_attributes(
            basin.river_network,
            columns=["COMID", "lengthkm", "uparea", *DEFAULT_UP_COLS],
        )
        river_df = prepare_river_dataframe(river_df)
        comid_list = river_df["COMID"].astype(int).tolist()
        comid_set = set(comid_list)
        length_dict = river_df.set_index("COMID")["lengthkm"].to_dict()
        uparea_map, upstream_map = build_river_maps(
            river_df,
            comid_col="COMID",
            uparea_col="uparea",
            up_cols=DEFAULT_UP_COLS,
        )
        outlet_comid = find_basin_outlet(river_df, comid_col="COMID", uparea_col="uparea")
        outlet_uparea = float(
            river_df.loc[river_df["COMID"] == outlet_comid, "uparea"].iloc[0]
        )
        mainstem_comids = trace_main_stem_with_maps(
            outlet_comid=outlet_comid,
            uparea_map=uparea_map,
            upstream_map=upstream_map,
        )
        segment_lengths = [float(length_dict[comid]) for comid in mainstem_comids if comid in length_dict]
        valid_mainstem = [comid for comid in mainstem_comids if comid in length_dict]
        cumulative_lengths: list[float] = []
        cumulative = 0.0
        for length in segment_lengths:
            cumulative += length
            cumulative_lengths.append(cumulative)

        basin_states.append(
            BasinState(
                basin=basin,
                river_df=river_df,
                comid_list=comid_list,
                comid_set=comid_set,
                length_dict=length_dict,
                upstream_map=upstream_map,
                outlet_comid=outlet_comid,
                outlet_uparea_km2=outlet_uparea,
                mainstem_comids=valid_mainstem,
                segment_lengths=segment_lengths,
                cumulative_lengths=cumulative_lengths,
            )
        )
        for comid in comid_list:
            comid_to_basin[comid] = basin.name

    return basin_states, comid_to_basin


def _compute_continuous_result(
    state: BasinState,
    value_df: pd.DataFrame,
    year: int,
    source_key: str,
    product: str,
    total_column_name: str,
    min_segments: int = 3,
) -> dict:
    q_dict = pd.Series(
        pd.to_numeric(value_df["qout"], errors="coerce").values,
        index=pd.to_numeric(value_df["COMID"], errors="coerce").astype(int),
    ).dropna().to_dict()

    valid_comids: list[int] = []
    segment_lengths: list[float] = []
    cumulative_lengths: list[float] = []
    q_values: list[float] = []
    cumulative = 0.0

    for comid in state.mainstem_comids:
        q_value = q_dict.get(comid)
        if q_value is None:
            continue
        valid_comids.append(comid)
        length = float(state.length_dict[comid])
        segment_lengths.append(length)
        cumulative += length
        cumulative_lengths.append(cumulative)
        q_values.append(float(q_value))

    if len(valid_comids) < min_segments:
        raise ValueError(
            f"Main stem has only {len(valid_comids)} effective segments for {source_key} {year}"
        )

    incremental = [q_values[0]]
    incremental.extend(
        float(current - previous) for previous, current in zip(q_values[:-1], q_values[1:])
    )
    total_value = float(np.sum(incremental))
    max_length, centroid = calculate_midpoint_centroid(
        segment_lengths,
        cumulative_lengths,
        incremental,
    )
    centroid_comid = find_centroid_comid(
        centroid,
        max_length,
        valid_comids,
        cumulative_lengths,
    )
    return {
        "basin_name": state.basin.name,
        "year": int(year),
        "source_key": source_key,
        "product": product,
        "outlet_COMID": state.outlet_comid,
        "centroid_COMID": centroid_comid,
        "centroid_distance_km": centroid,
        "mainstem_length_km": max_length,
        "rci": calculate_rci(centroid, max_length),
        "num_segments": len(valid_comids),
        total_column_name: total_value,
        "outlet_uparea_km2": state.outlet_uparea_km2,
    }


def _prepare_mswep_annual_grid_template(mswep_path: Path) -> tuple[list[int], dict, bool, bool]:
    with xr.open_dataset(mswep_path) as ds:
        data_var = ds["precipitation"]
        years = pd.to_datetime(ds["time"].values).year.astype(int).tolist()
        lat_values = np.asarray(data_var["lat"].values, dtype=float)
        lon_values = np.asarray(data_var["lon"].values, dtype=float)

    reverse_lat = bool(lat_values[0] < lat_values[-1])
    reverse_lon = bool(lon_values[0] > lon_values[-1])
    if reverse_lat:
        lat_values = lat_values[::-1]
    if reverse_lon:
        lon_values = lon_values[::-1]

    grid_template = {
        "kind": "nc",
        "crs": "EPSG:4326",
        "lat_edges": _coord_edges(lat_values),
        "lon_edges": _coord_edges(lon_values),
        "lat_descending": True,
        "lon_ascending": True,
    }
    return years, grid_template, reverse_lat, reverse_lon


def _compute_precipitation_result(
    state: BasinState,
    local_masses: dict[int, float],
    year: int,
    min_segments: int = 3,
) -> dict:
    resolver = UpstreamMassResolver(state.upstream_map)
    slice_masses: list[float] = []
    previous_total: float | None = None
    for comid in state.mainstem_comids:
        current_total = float(resolver.upstream_mass(comid, local_masses))
        if previous_total is None:
            slice_masses.append(current_total)
        else:
            slice_masses.append(current_total - previous_total)
        previous_total = current_total

    if len(state.mainstem_comids) < min_segments:
        raise ValueError(
            f"Main stem has only {len(state.mainstem_comids)} effective segments for precipitation {year}"
        )

    total_mass = float(np.sum(slice_masses))
    if total_mass <= 0:
        raise ValueError(f"Total precipitation mass is {total_mass:.6g} for {state.basin.name} {year}")

    max_length, centroid = calculate_midpoint_centroid(
        state.segment_lengths,
        state.cumulative_lengths,
        slice_masses,
    )
    centroid_comid = find_centroid_comid(
        centroid,
        max_length,
        state.mainstem_comids,
        state.cumulative_lengths,
    )
    return {
        "basin_name": state.basin.name,
        "year": int(year),
        "source_key": "p",
        "product": "MSWEP",
        "outlet_COMID": state.outlet_comid,
        "centroid_COMID": centroid_comid,
        "centroid_distance_km": centroid,
        "mainstem_length_km": max_length,
        "rci": calculate_rci(centroid, max_length),
        "num_segments": len(state.mainstem_comids),
        "total_mass": total_mass,
        "outlet_uparea_km2": state.outlet_uparea_km2,
    }


def _load_annual_discharge_table(path: Path, comid_to_basin: dict[int, str]) -> dict[str, pd.DataFrame]:
    q_df = pd.read_csv(path, usecols=["COMID", "qout"], compression="infer")
    q_df["COMID"] = pd.to_numeric(q_df["COMID"], errors="coerce").astype(int)
    q_df["qout"] = pd.to_numeric(q_df["qout"], errors="coerce")
    q_df["basin_name"] = q_df["COMID"].map(comid_to_basin)
    q_df = q_df.dropna(subset=["basin_name", "qout"])
    return {
        basin_name: group.loc[:, ["COMID", "qout"]].reset_index(drop=True)
        for basin_name, group in q_df.groupby("basin_name", sort=False)
    }


def _prepare_basin_catchment_weights(
    state: BasinState,
    grid_template: dict,
) -> BasinCatchmentWeights:
    catchment_gdf = pyogrio.read_dataframe(state.basin.catchments, columns=["COMID"])
    catchment_gdf["COMID"] = pd.to_numeric(catchment_gdf["COMID"], errors="coerce").astype(int)
    catchment_gdf = catchment_gdf[catchment_gdf["COMID"].isin(state.comid_set)].copy()
    catchment_gdf = _reproject_catchments_to_grid(catchment_gdf, grid_template)

    xmin, ymin, xmax, ymax = catchment_gdf.total_bounds
    lat_edges = grid_template["lat_edges"]
    lon_edges = grid_template["lon_edges"]
    row_start, row_end = _select_crop_indices(lat_edges, ymin, ymax, descending=True)
    col_start, col_end = _select_crop_indices(lon_edges, xmin, xmax, descending=False)
    crop_height = row_end - row_start
    crop_width = col_end - col_start
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError(f"Empty MSWEP crop for basin {state.basin.name}")

    transform = from_bounds(
        lon_edges[col_start],
        lat_edges[row_end],
        lon_edges[col_end],
        lat_edges[row_start],
        crop_width,
        crop_height,
    )

    shapes = [
        (geom, int(comid))
        for comid, geom in zip(catchment_gdf["COMID"], catchment_gdf.geometry)
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        raise ValueError(f"No valid catchment geometries for basin {state.basin.name}")

    factor = DEFAULT_CATCHMENT_MASS_SUPERSAMPLING_FACTOR
    sub_height = crop_height * factor
    sub_width = crop_width * factor
    sub_transform = _supersampled_transform(transform, factor)
    comid_raster = rasterize(
        shapes=shapes,
        out_shape=(sub_height, sub_width),
        transform=sub_transform,
        fill=0,
        dtype="int32",
    )
    labeled_indices = np.flatnonzero(comid_raster.ravel() > 0)
    if labeled_indices.size == 0:
        raise ValueError(f"No labeled MSWEP subcells for basin {state.basin.name}")

    sub_rows = labeled_indices // sub_width
    sub_cols = labeled_indices % sub_width
    parent_rows = sub_rows // factor
    parent_cols = sub_cols // factor
    labels = comid_raster.ravel()[labeled_indices].astype(np.int64)
    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    parent_flat_indices = (parent_rows * crop_width + parent_cols).astype(np.int64)
    row_areas = _row_pixel_areas_km2(sub_transform, sub_height, grid_template.get("crs"))
    subcell_areas = row_areas[sub_rows]

    return BasinCatchmentWeights(
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        unique_labels=unique_labels,
        inverse_indices=inverse_indices,
        parent_flat_indices=parent_flat_indices,
        subcell_areas_km2=subcell_areas,
    )


def _compute_local_masses_from_weights(
    annual_grid: np.ndarray,
    weights: BasinCatchmentWeights,
) -> dict[int, float]:
    crop = annual_grid[weights.row_start : weights.row_end, weights.col_start : weights.col_end]
    flat_crop = np.asarray(crop, dtype=float).ravel()
    parent_values = flat_crop[weights.parent_flat_indices]
    finite_mask = np.isfinite(parent_values)
    if not np.any(finite_mask):
        return {}

    sums = np.bincount(
        weights.inverse_indices[finite_mask],
        weights=parent_values[finite_mask] * weights.subcell_areas_km2[finite_mask],
        minlength=len(weights.unique_labels),
    )
    return dict(zip(weights.unique_labels.tolist(), sums.tolist()))


def build_figure8_annual_rci() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = _load_manifest()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    years: list[int] = [int(year) for year in manifest["analysis_common_years"]]
    grades_dir = Path(manifest["q_nat"]["annual_tables_dir"])
    grdr_dir = Path(manifest["q_hum"]["annual_tables_dir"])
    mswep_path = Path(manifest["precipitation"]["annual_clean_nc"])

    basin_states, comid_to_basin = _prepare_basin_states()
    state_by_name = {state.basin.name: state for state in basin_states}

    qnat_rows: list[dict] = []
    qhum_rows: list[dict] = []
    precip_rows: list[dict] = []
    failures: list[dict[str, object]] = []

    print("=" * 72, flush=True)
    print(
        f"Building Figure 8 annual RCI for {len(basin_states)} basins and {len(years)} years",
        flush=True,
    )
    print("=" * 72, flush=True)

    for year in years:
        print(f"\n[Q_nat] {year}", flush=True)
        annual_groups = _load_annual_discharge_table(
            grades_dir / f"GRADES_annual_mean_{year}.csv.gz",
            comid_to_basin=comid_to_basin,
        )
        for state in basin_states:
            try:
                qnat_rows.append(
                    _compute_continuous_result(
                        state=state,
                        value_df=annual_groups[state.basin.name],
                        year=year,
                        source_key="q_nat",
                        product="GRADES",
                        total_column_name="total_discharge",
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "basin_name": state.basin.name,
                        "year": year,
                        "source_key": "q_nat",
                        "error": str(exc),
                    }
                )

    for year in years:
        print(f"\n[Q_hum] {year}", flush=True)
        annual_groups = _load_annual_discharge_table(
            grdr_dir / f"GRDR_annual_mean_{year}.csv.gz",
            comid_to_basin=comid_to_basin,
        )
        for state in basin_states:
            try:
                qhum_rows.append(
                    _compute_continuous_result(
                        state=state,
                        value_df=annual_groups[state.basin.name],
                        year=year,
                        source_key="q_hum",
                        product="GRDR",
                        total_column_name="total_discharge",
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "basin_name": state.basin.name,
                        "year": year,
                        "source_key": "q_hum",
                        "error": str(exc),
                    }
                )

    mswep_years, grid_template, reverse_lat, reverse_lon = _prepare_mswep_annual_grid_template(
        mswep_path
    )
    year_to_index = {int(year): index for index, year in enumerate(mswep_years)}
    with xr.open_dataset(mswep_path) as mswep_ds:
        mswep_var = mswep_ds["precipitation"]

        for basin_index, state in enumerate(basin_states, start=1):
            print(f"\n[P {basin_index}/{len(basin_states)}] {state.basin.name}", flush=True)
            try:
                weights = _prepare_basin_catchment_weights(state, grid_template)
            except Exception as exc:
                for year in years:
                    failures.append(
                        {
                            "basin_name": state.basin.name,
                            "year": year,
                            "source_key": "p",
                            "error": f"Weight preparation failed: {exc}",
                        }
                    )
                continue

            for year in years:
                try:
                    annual_grid = np.asarray(mswep_var.isel(time=year_to_index[year]).values, dtype=float)
                    if reverse_lat:
                        annual_grid = annual_grid[::-1, :]
                    if reverse_lon:
                        annual_grid = annual_grid[:, ::-1]
                    local_masses = _compute_local_masses_from_weights(annual_grid, weights)
                    precip_rows.append(
                        _compute_precipitation_result(
                            state=state,
                            local_masses=local_masses,
                            year=year,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "basin_name": state.basin.name,
                            "year": year,
                            "source_key": "p",
                            "error": str(exc),
                        }
                    )

    qnat_df = pd.DataFrame(qnat_rows).sort_values(["basin_name", "year"]).reset_index(drop=True)
    qhum_df = pd.DataFrame(qhum_rows).sort_values(["basin_name", "year"]).reset_index(drop=True)
    precip_df = pd.DataFrame(precip_rows).sort_values(["basin_name", "year"]).reset_index(drop=True)
    failures_df = pd.DataFrame(failures, columns=["basin_name", "year", "source_key", "error"])

    combined_df = pd.concat(
        [
            qnat_df,
            qhum_df,
            precip_df,
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["source_key", "basin_name", "year"]).reset_index(drop=True)

    qnat_df.to_csv(Q_NAT_CSV, index=False)
    qhum_df.to_csv(Q_HUM_CSV, index=False)
    precip_df.to_csv(PRECIP_CSV, index=False)
    combined_df.to_csv(COMBINED_CSV, index=False)
    failures_df.to_csv(FAILURES_CSV, index=False)

    return qnat_df, qhum_df, precip_df, combined_df, failures_df


if __name__ == "__main__":
    build_figure8_annual_rci()
