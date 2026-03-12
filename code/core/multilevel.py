"""Multi-level centroid workflows for continuous and gridded packaged basins."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import xarray as xr
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from .basin_io import BasinPaths, iter_global_basin_paths
from .centroid import calculate_midpoint_centroid, calculate_rci, find_centroid_comid
from .continuous_source import (
    prepare_discharge_dataframe,
    prepare_river_dataframe,
    read_river_attributes,
)
from .mainstem import DEFAULT_UP_COLS, build_river_maps, trace_main_stem_with_maps
from .pfaf import compute_closed_prefix_status, summarize_unit_topology


COMPLETE_BASIN_DIGITS = {"2", "4", "6", "8"}
EARTH_RADIUS_M = 6_371_008.8
DEFAULT_CATCHMENT_MASS_SUPERSAMPLING_FACTOR = 8


def prepare_pfaf_dataframe(
    pfaf_df: pd.DataFrame,
    comid_col: str = "COMID",
    pfaf_col: str = "pfafstetter",
) -> pd.DataFrame:
    """Normalize a reach-level Pfaf code table."""
    if pfaf_col not in pfaf_df.columns:
        raise ValueError(f"Pfafstetter column '{pfaf_col}' not found. Available: {list(pfaf_df.columns)}")

    pfaf_df = pd.DataFrame(pfaf_df.loc[:, [comid_col, pfaf_col]]).copy()
    pfaf_df[comid_col] = pd.to_numeric(pfaf_df[comid_col], errors="coerce").astype(int)
    pfaf_df[pfaf_col] = pfaf_df[pfaf_col].astype(str)
    return pfaf_df.dropna()


def filter_subbasin_codes(
    merged_df: pd.DataFrame,
    full_codes: Optional[Iterable[str]] = None,
    filter_mode: str = "independent_basins",
) -> pd.DataFrame:
    """Filter Pfaf codes according to the selected basin-eligibility rule."""
    if filter_mode == "pure_even_path":
        return merged_df[
            merged_df["subbasin_id"].apply(
                lambda code: all(digit in COMPLETE_BASIN_DIGITS for digit in str(code))
            )
        ].copy()

    if filter_mode == "independent_basins":
        target_level = int(merged_df["subbasin_id"].astype(str).str.len().iloc[0])
        if full_codes is None:
            full_codes = merged_df["pfafstetter"].astype(str).tolist()
        closed_status = compute_closed_prefix_status(full_codes, max_level=target_level)
        eligible_codes = [
            prefix
            for prefix, is_closed in closed_status.items()
            if len(prefix) == target_level and is_closed
        ]
        return merged_df[merged_df["subbasin_id"].isin(eligible_codes)].copy()

    raise ValueError(
        f"Unsupported filter_mode '{filter_mode}'. Use 'pure_even_path' or 'independent_basins'."
    )


def find_subbasin_outlets(
    river_df: pd.DataFrame,
    pfaf_df: pd.DataFrame,
    level: int,
    uparea_col: str = "uparea",
    pfaf_col: str = "pfafstetter",
    min_subbasin_segments: int = 1,
    filter_mode: str = "independent_basins",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find eligible subbasins and their outlet reaches at one Pfaf level."""
    merged_all = river_df.merge(pfaf_df, on="COMID", how="inner")
    merged_all["pfaf_len"] = merged_all[pfaf_col].astype(str).str.len()
    full_codes = merged_all[pfaf_col].astype(str).tolist()

    merged_df = merged_all[merged_all["pfaf_len"] >= level].copy()
    merged_df["subbasin_id"] = merged_df[pfaf_col].astype(str).str[:level]
    merged_df["level_digit"] = merged_df["subbasin_id"].str[-1]

    if merged_df.empty:
        return merged_df, pd.DataFrame()

    merged_df = filter_subbasin_codes(
        merged_df,
        full_codes=full_codes,
        filter_mode=filter_mode,
    )
    if merged_df.empty:
        return merged_df, pd.DataFrame()

    if filter_mode == "independent_basins":
        _, upstream_map = build_river_maps(
            river_df,
            comid_col="COMID",
            uparea_col=uparea_col,
            up_cols=DEFAULT_UP_COLS,
        )
        downstream_map: Dict[int, Optional[int]] = {}
        for downstream_comid, upstreams in upstream_map.items():
            for upstream_comid in upstreams:
                downstream_map[upstream_comid] = downstream_comid

        eligible_ids: List[str] = []
        for subbasin_id, group in merged_df.groupby("subbasin_id"):
            topology = summarize_unit_topology(
                set(group["COMID"].astype(int)),
                upstream_map=upstream_map,
                downstream_map=downstream_map,
            )
            if topology["outlet_count"] == 1 and topology["external_inflow_count"] == 0:
                eligible_ids.append(str(subbasin_id))

        merged_df = merged_df[merged_df["subbasin_id"].isin(eligible_ids)].copy()
        if merged_df.empty:
            return merged_df, pd.DataFrame()

    subbasin_sizes = (
        merged_df.groupby("subbasin_id")
        .size()
        .rename("num_subbasin_segments")
        .reset_index()
    )
    merged_df = merged_df.merge(subbasin_sizes, on="subbasin_id", how="left")
    if min_subbasin_segments > 1:
        merged_df = merged_df[merged_df["num_subbasin_segments"] >= min_subbasin_segments].copy()
    if merged_df.empty:
        return merged_df, pd.DataFrame()

    outlets = merged_df.loc[merged_df.groupby("subbasin_id")[uparea_col].idxmax()].copy()
    outlets = (
        outlets[["subbasin_id", "COMID", uparea_col, "num_subbasin_segments"]]
        .rename(columns={"COMID": "outlet_COMID"})
        .sort_values("subbasin_id")
        .reset_index(drop=True)
    )
    return merged_df, outlets


def calculate_incremental_discharge(
    river_df: pd.DataFrame,
    mainstem_comids: List[int],
    q_df: pd.DataFrame,
    comid_col: str = "COMID",
    q_col: str = "qout",
    length_col: str = "lengthkm",
    length_dict: Optional[Dict[int, float]] = None,
    q_dict: Optional[Dict[int, float]] = None,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Compute effective mainstem reaches and their incremental discharge."""
    if length_dict is None:
        length_dict = river_df.set_index(comid_col)[length_col].to_dict()
    if q_dict is None:
        q_dict = q_df.set_index(comid_col)[q_col].to_dict()

    valid_mainstem_comids = []
    segment_lengths = []
    cum_lengths = []
    q_values = []
    cum_length = 0.0

    for comid in mainstem_comids:
        if comid not in length_dict or comid not in q_dict:
            continue
        valid_mainstem_comids.append(comid)
        segment_length = float(length_dict[comid])
        segment_lengths.append(segment_length)
        cum_length += segment_length
        cum_lengths.append(cum_length)
        q_values.append(float(q_dict[comid]))

    delta_q = []
    for index, q_value in enumerate(q_values):
        if index == 0:
            delta_q.append(q_value)
        else:
            delta_q.append(q_value - q_values[index - 1])

    return valid_mainstem_comids, segment_lengths, cum_lengths, delta_q


def calculate_basin_centroids_from_tables(
    river_df: pd.DataFrame,
    q_df: pd.DataFrame,
    pfaf_df: pd.DataFrame,
    output_path: str | Path,
    basin_name: str,
    levels: List[int],
    comid_col: str = "COMID",
    q_col: str = "qout",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[List[str]] = None,
    pfaf_col: str = "pfafstetter",
    min_segments: int = 1,
    filter_mode: str = "independent_basins",
    min_mainstem_segments: int = 4,
    total_column_name: str = "total_discharge",
    min_total: float | None = None,
    print_header: bool = True,
    q_lookup: Optional[Dict[int, float]] = None,
    river_prepared: bool = False,
    q_prepared: bool = False,
    pfaf_prepared: bool = False,
) -> pd.DataFrame:
    """Calculate multi-level continuous RCI from in-memory tables."""
    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    if river_prepared:
        river_df = pd.DataFrame(
            river_df.loc[:, [comid_col, length_col, uparea_col, *up_cols]]
        ).copy()
    else:
        river_df = prepare_river_dataframe(
            river_df,
            comid_col=comid_col,
            length_col=length_col,
            uparea_col=uparea_col,
            up_cols=up_cols,
        )

    if q_prepared:
        q_df = pd.DataFrame(q_df.loc[:, [comid_col, q_col]]).copy()
    else:
        q_df = prepare_discharge_dataframe(q_df, comid_col=comid_col, q_col=q_col)

    if pfaf_prepared:
        pfaf_df = pd.DataFrame(pfaf_df.loc[:, [comid_col, pfaf_col]]).copy()
    else:
        pfaf_df = prepare_pfaf_dataframe(pfaf_df, comid_col=comid_col, pfaf_col=pfaf_col)

    if print_header:
        print("=" * 60)
        print(f"Calculating flow centroids for basin: {basin_name}")
        print("=" * 60)

    all_results = []
    q_dict = q_lookup if q_lookup is not None else q_df.set_index(comid_col)[q_col].to_dict()
    length_dict = river_df.set_index(comid_col)[length_col].to_dict()

    for level in levels:
        print(f"\n[Step 2] Processing Pfafstetter level {level}...")
        level_df, outlets_df = find_subbasin_outlets(
            river_df,
            pfaf_df,
            level,
            uparea_col=uparea_col,
            pfaf_col=pfaf_col,
            min_subbasin_segments=min_segments,
            filter_mode=filter_mode,
        )
        if outlets_df.empty:
            print(f"  No eligible subbasins found for level {level}")
            continue

        print(f"  Found {len(outlets_df)} eligible subbasins at level {level}")
        level_uparea_map, level_upstream_map = build_river_maps(
            level_df,
            comid_col=comid_col,
            uparea_col=uparea_col,
            up_cols=up_cols,
        )
        subbasin_sets = (
            level_df.groupby("subbasin_id")[comid_col]
            .apply(lambda series: set(series.astype(int)))
            .to_dict()
        )
        level_results = []

        for _, outlet_row in outlets_df.iterrows():
            outlet_comid = int(outlet_row["outlet_COMID"])
            subbasin_id = outlet_row["subbasin_id"]
            subbasin_num_segments = int(outlet_row["num_subbasin_segments"])
            subbasin_comids = subbasin_sets[subbasin_id]

            mainstem_comids = trace_main_stem_with_maps(
                outlet_comid=outlet_comid,
                uparea_map=level_uparea_map,
                upstream_map=level_upstream_map,
                valid_comids=subbasin_comids,
            )
            if len(mainstem_comids) < min_mainstem_segments:
                continue

            valid_mainstem, segment_lengths, cum_lengths, delta_q = calculate_incremental_discharge(
                river_df,
                mainstem_comids,
                q_df,
                comid_col=comid_col,
                q_col=q_col,
                length_col=length_col,
                length_dict=length_dict,
                q_dict=q_dict,
            )
            if len(valid_mainstem) < min_mainstem_segments or len(cum_lengths) == 0:
                continue

            total_value = float(np.sum(delta_q))
            if min_total is not None and total_value <= min_total:
                continue

            max_length, centroid = calculate_midpoint_centroid(segment_lengths, cum_lengths, delta_q)
            centroid_comid = find_centroid_comid(centroid, max_length, valid_mainstem, cum_lengths)
            rci = calculate_rci(centroid, max_length)

            level_results.append(
                {
                    "basin_name": basin_name,
                    "level": level,
                    "subbasin_code": subbasin_id,
                    "outlet_COMID": outlet_comid,
                    "centroid_COMID": centroid_comid,
                    "centroid_distance_km": centroid,
                    "mainstem_length_km": max_length,
                    "rci": rci,
                    "num_segments": len(valid_mainstem),
                    "num_subbasin_segments": subbasin_num_segments,
                    total_column_name: total_value,
                }
            )

        if level_results:
            print(f"  Calculated centroids for {len(level_results)} subbasins")
            all_results.append(pd.DataFrame(level_results))

    if not all_results:
        raise ValueError("No results generated for any level")

    final_df = pd.concat(all_results, ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    return final_df


def run_single_basin_multilevel_continuous(
    basin: BasinPaths,
    output_path: str | Path,
    levels: list[int],
    q_col: str = "qout",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    pfaf_col: str = "pfafstetter",
    min_segments: int = 1,
    filter_mode: str = "independent_basins",
    min_mainstem_segments: int = 4,
) -> pd.DataFrame:
    """Run continuous multilevel RCI for one packaged basin."""
    if basin.discharge_csv is None:
        raise FileNotFoundError(f"No discharge CSV found for basin {basin.name}")
    if basin.pfaf_csv is None:
        raise FileNotFoundError(f"No Pfaf CSV found for basin {basin.name}")

    river_df = read_river_attributes(
        basin.river_network,
        columns=[comid_col, length_col, uparea_col, *(up_cols or DEFAULT_UP_COLS)],
    )
    q_df = pd.read_csv(basin.discharge_csv, usecols=[comid_col, q_col])
    pfaf_df = pd.read_csv(basin.pfaf_csv, usecols=[comid_col, pfaf_col])
    return calculate_basin_centroids_from_tables(
        river_df=river_df,
        q_df=q_df,
        pfaf_df=pfaf_df,
        output_path=output_path,
        basin_name=basin.name,
        levels=levels,
        comid_col=comid_col,
        q_col=q_col,
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
        pfaf_col=pfaf_col,
        min_segments=min_segments,
        filter_mode=filter_mode,
        min_mainstem_segments=min_mainstem_segments,
        total_column_name="total_discharge",
        print_header=True,
    )


def summarize_multilevel_results(
    result_df: pd.DataFrame,
    weight_column: str,
) -> pd.DataFrame:
    """Create a per-level summary table for multilevel outputs."""
    if result_df.empty:
        return pd.DataFrame(
            columns=[
                "level",
                "n",
                "mean_rci",
                "median_rci",
                "min_rci",
                "max_rci",
                "mean_mainstem_length_km",
                f"mean_{weight_column}",
            ]
        )

    return (
        result_df.groupby("level")
        .agg(
            n=("rci", "size"),
            mean_rci=("rci", "mean"),
            median_rci=("rci", "median"),
            min_rci=("rci", "min"),
            max_rci=("rci", "max"),
            mean_mainstem_length_km=("mainstem_length_km", "mean"),
            **{f"mean_{weight_column}": (weight_column, "mean")},
        )
        .reset_index()
    )


def run_batch_basin_multilevel_continuous(
    global_dir: str = "data/basins/global",
    output_dir: str = "outputs/batch/multilevel_continuous",
    levels: list[int] | None = None,
    q_col: str = "qout",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    pfaf_col: str = "pfafstetter",
    min_segments: int = 1,
    filter_mode: str = "independent_basins",
    min_mainstem_segments: int = 4,
    basin_names: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run continuous multilevel RCI across packaged global basins."""
    if levels is None:
        levels = [1, 2, 3, 4, 5]
    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    output_dir_path = Path(output_dir)
    per_basin_dir = output_dir_path / "per_basin"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    per_basin_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    failure_rows = []
    basins = list(iter_global_basin_paths(global_dir, basin_names=basin_names))

    print("=" * 72)
    print(f"Running global multi-level RCI for {len(basins)} basins")
    print("Input mode: packaged data/basins/global folders")
    print("=" * 72)

    for index, basin in enumerate(basins, start=1):
        print(f"\n[{index}/{len(basins)}] {basin.name}")
        output_path = per_basin_dir / f"{basin.name}_multilevel_rci.csv"
        try:
            basin_df = run_single_basin_multilevel_continuous(
                basin=basin,
                output_path=output_path,
                levels=levels,
                q_col=q_col,
                comid_col=comid_col,
                length_col=length_col,
                uparea_col=uparea_col,
                up_cols=up_cols,
                pfaf_col=pfaf_col,
                min_segments=min_segments,
                filter_mode=filter_mode,
                min_mainstem_segments=min_mainstem_segments,
            )
            basin_df.insert(1, "river_file", basin.river_network.name)
            basin_df.insert(2, "pfaf_file", basin.pfaf_csv.name if basin.pfaf_csv else "")
            basin_df.insert(3, "source_mode", basin.mode)
            result_rows.append(basin_df)
        except Exception as exc:
            failure_rows.append(
                {
                    "basin_name": basin.name,
                    "river_file": basin.river_network.name,
                    "pfaf_file": basin.pfaf_csv.name if basin.pfaf_csv else "",
                    "error": str(exc),
                }
            )

    combined_df = pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()
    failures_df = pd.DataFrame(failure_rows)
    combined_df.to_csv(output_dir_path / "global_multilevel_rci_results.csv", index=False)
    failures_df.to_csv(output_dir_path / "global_multilevel_rci_failures.csv", index=False)
    summarize_multilevel_results(combined_df, "total_discharge").to_csv(
        output_dir_path / "global_multilevel_rci_summary_by_level.csv",
        index=False,
    )
    return combined_df, failures_df


def _infer_netcdf_spatial_dims(data_var: xr.DataArray) -> Tuple[str, str]:
    lon_candidates = {"lon", "longitude", "x"}
    lat_candidates = {"lat", "latitude", "y"}
    lon_dim = next((dim for dim in data_var.dims if dim.lower() in lon_candidates), None)
    lat_dim = next((dim for dim in data_var.dims if dim.lower() in lat_candidates), None)
    if lon_dim is None or lat_dim is None:
        raise ValueError(
            f"Unable to infer spatial dimensions from NetCDF variable dims {data_var.dims}."
        )
    return lat_dim, lon_dim


def _reduce_netcdf_to_2d(
    data_var: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    reduction: str = "mean",
) -> xr.DataArray:
    extra_dims = [dim for dim in data_var.dims if dim not in {lat_dim, lon_dim}]
    if not extra_dims:
        return data_var
    if reduction == "mean":
        return data_var.mean(dim=extra_dims, skipna=True)
    if reduction == "sum":
        return data_var.sum(dim=extra_dims, skipna=True)
    if reduction == "median":
        return data_var.median(dim=extra_dims, skipna=True)
    if reduction == "first":
        return data_var.isel({dim: 0 for dim in extra_dims})
    raise ValueError(f"Unsupported NetCDF reduction '{reduction}'")


def _coord_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    diffs = np.diff(values)
    if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-6):
        raise ValueError("NetCDF spatial coordinates must be regularly spaced")
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = (values[:-1] + values[1:]) / 2
    edges[0] = values[0] - diffs[0] / 2
    edges[-1] = values[-1] + diffs[-1] / 2
    return edges


def load_gridded_grid(
    gridded_data_path: str | Path,
    variable: str = "precipitation",
    reduction: str = "mean",
) -> dict:
    """Load a gridded field into a lightweight extraction dictionary."""
    gridded_data_path = Path(gridded_data_path)
    suffix = gridded_data_path.suffix.lower()

    if suffix == ".nc":
        with xr.open_dataset(gridded_data_path) as ds:
            if variable not in ds.variables:
                raise ValueError(f"Variable '{variable}' not found in NetCDF. Available: {list(ds.variables)}")
            data_var = ds[variable]
            lat_dim, lon_dim = _infer_netcdf_spatial_dims(data_var)
            data_var = _reduce_netcdf_to_2d(data_var, lat_dim, lon_dim, reduction=reduction)
            data_var = data_var.transpose(lat_dim, lon_dim)

            lat_values = np.asarray(data_var[lat_dim].values, dtype=float)
            lon_values = np.asarray(data_var[lon_dim].values, dtype=float)
            data_array = np.asarray(data_var.values, dtype=float)

        if lat_values[0] < lat_values[-1]:
            lat_values = lat_values[::-1]
            data_array = data_array[::-1, :]
        if lon_values[0] > lon_values[-1]:
            lon_values = lon_values[::-1]
            data_array = data_array[:, ::-1]

        lat_edges = _coord_edges(lat_values)
        lon_edges = _coord_edges(lon_values)
        return {
            "data": data_array,
            "kind": "nc",
            "crs": "EPSG:4326",
            "lat_edges": lat_edges,
            "lon_edges": lon_edges,
            "lat_descending": True,
            "lon_ascending": True,
        }

    with rasterio.open(gridded_data_path) as src:
        data = src.read(1, masked=True)
        return {
            "data": np.asarray(data.filled(np.nan), dtype=float),
            "kind": "raster",
            "crs": src.crs,
            "transform": src.transform,
            "shape": (src.height, src.width),
        }


def _select_crop_indices(
    edges: np.ndarray,
    min_value: float,
    max_value: float,
    descending: bool = False,
) -> tuple[int, int]:
    centers = (edges[:-1] + edges[1:]) / 2
    if descending:
        valid = (centers <= max_value) & (centers >= min_value)
    else:
        valid = (centers >= min_value) & (centers <= max_value)
    if not np.any(valid):
        return 0, len(centers)
    indices = np.where(valid)[0]
    return int(indices.min()), int(indices.max()) + 1


def _crop_grid(grid: dict, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, object]:
    """Crop a loaded grid to catchment bounds and return data plus transform."""
    xmin, ymin, xmax, ymax = bounds
    if grid["kind"] == "nc":
        lon_edges = grid["lon_edges"]
        lat_edges = grid["lat_edges"]
        row_start, row_end = _select_crop_indices(lat_edges, ymin, ymax, descending=True)
        col_start, col_end = _select_crop_indices(lon_edges, xmin, xmax, descending=False)
        data_crop = grid["data"][row_start:row_end, col_start:col_end]
        transform = from_bounds(
            lon_edges[col_start],
            lat_edges[row_end],
            lon_edges[col_end],
            lat_edges[row_start],
            data_crop.shape[1],
            data_crop.shape[0],
        )
        return data_crop, transform

    return grid["data"], grid["transform"]


def _normalize_grid_crs(crs_value) -> Optional[CRS]:
    if crs_value is None:
        return None
    return CRS.from_user_input(crs_value)


def _reproject_catchments_to_grid(catchment_gdf, grid: dict):
    target_crs = _normalize_grid_crs(grid.get("crs"))
    if target_crs is None or getattr(catchment_gdf, "crs", None) is None:
        return catchment_gdf

    current_crs = _normalize_grid_crs(catchment_gdf.crs)
    if current_crs == target_crs:
        return catchment_gdf
    return catchment_gdf.to_crs(target_crs)


def _supersampled_transform(transform, factor: int):
    return transform * Affine.scale(1.0 / factor, 1.0 / factor)


def _row_pixel_areas_km2(transform, height: int, crs_value) -> np.ndarray:
    grid_crs = _normalize_grid_crs(crs_value)
    if grid_crs is None or grid_crs.is_geographic:
        lon_width_rad = np.deg2rad(abs(float(transform.a)))
        lat_edges = float(transform.f) + np.arange(height + 1, dtype=float) * float(transform.e)
        lat_edges_rad = np.deg2rad(lat_edges)
        row_areas_m2 = (
            (EARTH_RADIUS_M**2)
            * np.abs(np.sin(lat_edges_rad[:-1]) - np.sin(lat_edges_rad[1:]))
            * lon_width_rad
        )
        return row_areas_m2 / 1e6

    pixel_area_m2 = abs(float(transform.a) * float(transform.e))
    return np.full(height, pixel_area_m2 / 1e6, dtype=float)


def compute_catchment_masses(
    catchment_gdf,
    grid: dict,
    supersample_factor: int = DEFAULT_CATCHMENT_MASS_SUPERSAMPLING_FACTOR,
) -> dict[int, float]:
    """Approximate catchment precipitation masses by supersampled area weighting."""
    if catchment_gdf.empty:
        return {}
    if supersample_factor < 1:
        raise ValueError("supersample_factor must be >= 1")

    catchment_gdf = _reproject_catchments_to_grid(catchment_gdf, grid)
    data_crop, transform = _crop_grid(grid, catchment_gdf.total_bounds)
    if data_crop.size == 0:
        return {}

    shapes = [
        (geom, int(comid))
        for comid, geom in zip(catchment_gdf["COMID"], catchment_gdf.geometry)
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return {}

    sub_height = int(data_crop.shape[0]) * supersample_factor
    sub_width = int(data_crop.shape[1]) * supersample_factor
    sub_transform = _supersampled_transform(transform, supersample_factor)
    comid_raster = rasterize(
        shapes=shapes,
        out_shape=(sub_height, sub_width),
        transform=sub_transform,
        fill=0,
        dtype="int32",
    )
    labeled_indices = np.flatnonzero(comid_raster.ravel() > 0)
    if labeled_indices.size == 0:
        return {}

    sub_rows = labeled_indices // sub_width
    sub_cols = labeled_indices % sub_width
    parent_rows = sub_rows // supersample_factor
    parent_cols = sub_cols // supersample_factor
    parent_values = np.asarray(data_crop, dtype=np.float64)[parent_rows, parent_cols]
    finite_mask = np.isfinite(parent_values)
    if not np.any(finite_mask):
        return {}

    valid_labels = comid_raster.ravel()[labeled_indices[finite_mask]].astype(np.int64)
    valid_rows = sub_rows[finite_mask]
    valid_values = parent_values[finite_mask]
    subcell_areas_km2 = _row_pixel_areas_km2(sub_transform, sub_height, grid.get("crs"))
    weighted_values = valid_values * subcell_areas_km2[valid_rows]

    unique_labels, inverse = np.unique(valid_labels, return_inverse=True)
    sums = np.bincount(inverse, weights=weighted_values)
    return dict(zip(unique_labels.tolist(), sums.tolist()))


class UpstreamMassResolver:
    """Memoized recursive upstream-mass accumulator."""

    def __init__(self, upstream_map: Dict[int, list[int]]):
        self.upstream_map = upstream_map
        self.cache: Dict[int, float] = {}

    def upstream_mass(self, comid: int, catchment_masses: dict[int, float]) -> float:
        if comid in self.cache:
            return self.cache[comid]

        stack: list[tuple[int, bool]] = [(comid, False)]
        processing: set[int] = set()

        while stack:
            current, expanded = stack.pop()
            if current in self.cache:
                continue
            if not expanded:
                if current in processing:
                    raise ValueError(f"Cycle detected while tracing upstream closure of COMID {current}")
                processing.add(current)
                stack.append((current, True))
                for upstream in self.upstream_map.get(current, []):
                    if upstream not in self.cache:
                        stack.append((upstream, False))
                continue

            total_mass = float(catchment_masses.get(current, 0.0))
            for upstream in self.upstream_map.get(current, []):
                total_mass += float(self.cache.get(upstream, 0.0))
            self.cache[current] = total_mass
            processing.discard(current)

        return self.cache[comid]


def _prepare_mainstem_lengths(
    mainstem_comids: list[int],
    length_dict: dict[int, float],
) -> tuple[list[int], list[float], list[float]]:
    valid_comids = [comid for comid in mainstem_comids if comid in length_dict]
    segment_lengths = [float(length_dict[comid]) for comid in valid_comids]
    cum_lengths = []
    cumulative = 0.0
    for length in segment_lengths:
        cumulative += length
        cum_lengths.append(cumulative)
    return valid_comids, segment_lengths, cum_lengths


def _calculate_incremental_masses(
    mainstem_comids: list[int],
    mass_resolver: UpstreamMassResolver,
    catchment_masses: dict[int, float],
) -> list[float]:
    masses = []
    previous_total: Optional[float] = None
    for comid in mainstem_comids:
        current_total = mass_resolver.upstream_mass(comid, catchment_masses)
        if previous_total is None:
            masses.append(current_total)
        else:
            masses.append(current_total - previous_total)
        previous_total = current_total
    return masses


def build_level0_gridded_result(
    basin_name: str,
    root_river_df: pd.DataFrame,
    root_comids: set[int],
    catchment_masses: dict[int, float],
    min_mainstem_segments: int,
    comid_col: str,
    length_col: str,
    uparea_col: str,
    up_cols: list[str],
) -> Optional[dict]:
    """Build the basin-scale level-0 gridded result from reach catchment masses."""
    if root_river_df.empty:
        return None

    length_dict = root_river_df.set_index(comid_col)[length_col].to_dict()
    uparea_map, upstream_map = build_river_maps(
        root_river_df,
        comid_col=comid_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
    )
    mass_resolver = UpstreamMassResolver(upstream_map)

    outlet_idx = root_river_df[uparea_col].idxmax()
    outlet_comid = int(root_river_df.loc[outlet_idx, comid_col])
    mainstem_comids = trace_main_stem_with_maps(
        outlet_comid=outlet_comid,
        uparea_map=uparea_map,
        upstream_map=upstream_map,
        valid_comids=root_comids,
    )
    valid_comids, segment_lengths, cum_lengths = _prepare_mainstem_lengths(mainstem_comids, length_dict)
    if len(valid_comids) < min_mainstem_segments:
        return None

    slice_masses = _calculate_incremental_masses(valid_comids, mass_resolver, catchment_masses)
    total_mass = float(np.sum(slice_masses))
    if total_mass <= 0:
        return None

    max_length, centroid = calculate_midpoint_centroid(segment_lengths, cum_lengths, slice_masses)
    centroid_comid = find_centroid_comid(centroid, max_length, valid_comids, cum_lengths)
    rci = calculate_rci(centroid, max_length)
    return {
        "basin_name": basin_name,
        "level": 0,
        "subbasin_code": "ROOT",
        "outlet_COMID": outlet_comid,
        "centroid_COMID": centroid_comid,
        "centroid_distance_km": centroid,
        "mainstem_length_km": max_length,
        "rci": rci,
        "num_segments": len(valid_comids),
        "num_subbasin_segments": len(root_comids),
        "total_mass": total_mass,
    }


def build_multilevel_gridded_results(
    basin_name: str,
    root_river_df: pd.DataFrame,
    pfaf_df: pd.DataFrame,
    catchment_masses: dict[int, float],
    levels: list[int],
    min_segments: int,
    filter_mode: str,
    min_mainstem_segments: int,
    comid_col: str,
    length_col: str,
    uparea_col: str,
    up_cols: list[str],
    pfaf_col: str,
) -> list[dict]:
    """Build multilevel gridded RCI results from catchment masses."""
    if root_river_df.empty:
        return []

    length_dict = root_river_df.set_index(comid_col)[length_col].to_dict()
    uparea_map, upstream_map = build_river_maps(
        root_river_df,
        comid_col=comid_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
    )
    mass_resolver = UpstreamMassResolver(upstream_map)
    results: list[dict] = []

    for level in levels:
        print(f"\n[Step 2] Processing Pfafstetter level {level}...")
        level_df, outlets_df = find_subbasin_outlets(
            root_river_df,
            pfaf_df,
            level,
            uparea_col=uparea_col,
            pfaf_col=pfaf_col,
            min_subbasin_segments=min_segments,
            filter_mode=filter_mode,
        )
        if outlets_df.empty:
            print(f"  No eligible subbasins found for level {level}")
            continue

        print(f"  Found {len(outlets_df)} eligible subbasins at level {level}")
        level_uparea_map, level_upstream_map = build_river_maps(
            level_df,
            comid_col=comid_col,
            uparea_col=uparea_col,
            up_cols=up_cols,
        )
        subbasin_sets = (
            level_df.groupby("subbasin_id")[comid_col]
            .apply(lambda series: set(series.astype(int)))
            .to_dict()
        )
        level_result_count = 0

        for _, outlet_row in outlets_df.iterrows():
            outlet_comid = int(outlet_row["outlet_COMID"])
            subbasin_id = outlet_row["subbasin_id"]
            subbasin_set = subbasin_sets[subbasin_id]
            subbasin_num_segments = int(outlet_row["num_subbasin_segments"])

            mainstem_comids = trace_main_stem_with_maps(
                outlet_comid=outlet_comid,
                uparea_map=level_uparea_map,
                upstream_map=level_upstream_map,
                valid_comids=subbasin_set,
            )
            valid_comids, segment_lengths, cum_lengths = _prepare_mainstem_lengths(mainstem_comids, length_dict)
            if len(valid_comids) < min_mainstem_segments:
                continue

            slice_masses = _calculate_incremental_masses(valid_comids, mass_resolver, catchment_masses)
            total_mass = float(np.sum(slice_masses))
            if total_mass <= 0:
                continue

            max_length, centroid = calculate_midpoint_centroid(segment_lengths, cum_lengths, slice_masses)
            centroid_comid = find_centroid_comid(centroid, max_length, valid_comids, cum_lengths)
            rci = calculate_rci(centroid, max_length)
            results.append(
                {
                    "basin_name": basin_name,
                    "level": level,
                    "subbasin_code": subbasin_id,
                    "outlet_COMID": outlet_comid,
                    "centroid_COMID": centroid_comid,
                    "centroid_distance_km": centroid,
                    "mainstem_length_km": max_length,
                    "rci": rci,
                    "num_segments": len(valid_comids),
                    "num_subbasin_segments": subbasin_num_segments,
                    "total_mass": total_mass,
                }
            )
            level_result_count += 1

        print(f"  Calculated precipitation centroids for {level_result_count} subbasins")
    return results


def run_single_basin_multilevel_gridded(
    basin: BasinPaths,
    gridded_data_path: str | Path,
    output_path: str | Path,
    levels: list[int],
    variable: str = "precipitation",
    reduction: str = "mean",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    pfaf_col: str = "pfafstetter",
    min_segments: int = 1,
    filter_mode: str = "independent_basins",
    min_mainstem_segments: int = 4,
    grid: Optional[dict] = None,
) -> pd.DataFrame:
    """Run gridded multilevel RCI from the persisted COMID -> P_upstream_total table."""
    if basin.pfaf_csv is None:
        raise FileNotFoundError(f"No Pfaf CSV found for basin {basin.name}")

    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    from .gridded_source import ensure_precipitation_upstream_dataframe

    river_df, precip_df = ensure_precipitation_upstream_dataframe(
        basin=basin,
        gridded_data_path=gridded_data_path,
        variable=variable,
        reduction=reduction,
        comid_col=comid_col,
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
        grid=grid,
    )
    pfaf_df = pd.read_csv(basin.pfaf_csv, usecols=[comid_col, pfaf_col])
    result_df = calculate_basin_centroids_from_tables(
        river_df=river_df,
        q_df=precip_df[[comid_col, "p_upstream_total"]],
        pfaf_df=pfaf_df,
        output_path=output_path,
        basin_name=basin.name,
        levels=levels,
        comid_col=comid_col,
        q_col="p_upstream_total",
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
        pfaf_col=pfaf_col,
        min_segments=min_segments,
        filter_mode=filter_mode,
        min_mainstem_segments=min_mainstem_segments,
        total_column_name="total_mass",
        min_total=0.0,
        print_header=True,
        river_prepared=True,
        q_prepared=True,
    )
    result_df.insert(1, "river_file", basin.river_network.name)
    result_df.insert(2, "pfaf_file", basin.pfaf_csv.name)
    result_df.insert(3, "source_variable", variable)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    return result_df


def run_batch_basin_multilevel_gridded(
    global_dir: str = "data/basins/global",
    gridded_data_path: str = "data/climate/climatology/mswep_precipitation_mean.nc",
    output_dir: str = "outputs/batch/multilevel_gridded",
    levels: list[int] | None = None,
    variable: str = "precipitation",
    reduction: str = "mean",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    pfaf_col: str = "pfafstetter",
    min_segments: int = 1,
    filter_mode: str = "independent_basins",
    min_mainstem_segments: int = 4,
    basin_names: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run gridded multilevel RCI across packaged global basins."""
    if levels is None:
        levels = [1, 2, 3, 4]
    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    output_dir_path = Path(output_dir)
    per_basin_dir = output_dir_path / "per_basin"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    per_basin_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    basins = list(iter_global_basin_paths(global_dir, basin_names=basin_names))
    grid = None

    for basin in basins:
        output_path = per_basin_dir / f"{basin.name}_multilevel_gridded_rci.csv"
        try:
            if basin.precipitation_upstream_csv is None or not basin.precipitation_upstream_csv.exists():
                if grid is None:
                    grid = load_gridded_grid(
                        gridded_data_path=gridded_data_path,
                        variable=variable,
                        reduction=reduction,
                    )
            basin_df = run_single_basin_multilevel_gridded(
                basin=basin,
                gridded_data_path=gridded_data_path,
                output_path=output_path,
                levels=levels,
                variable=variable,
                reduction=reduction,
                comid_col=comid_col,
                length_col=length_col,
                uparea_col=uparea_col,
                up_cols=up_cols,
                pfaf_col=pfaf_col,
                min_segments=min_segments,
                filter_mode=filter_mode,
                min_mainstem_segments=min_mainstem_segments,
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
    combined_df.to_csv(output_dir_path / "global_multilevel_gridded_rci_results.csv", index=False)
    failures_df.to_csv(output_dir_path / "global_multilevel_gridded_rci_failures.csv", index=False)
    summarize_multilevel_results(combined_df, "total_mass").to_csv(
        output_dir_path / "global_multilevel_gridded_rci_summary_by_level.csv",
        index=False,
    )
    return combined_df, failures_df
