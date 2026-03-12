"""Build Figure 7c style aridity-group boxplots from seasonal runoff and precipitation RCIs."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.basin_io import iter_global_basin_paths
from core.continuous_source import (
    calculate_basin_centroid_from_tables,
    prepare_river_dataframe,
    read_river_attributes,
)
from core.mainstem import build_river_maps
from core.multilevel import (
    DEFAULT_CATCHMENT_MASS_SUPERSAMPLING_FACTOR,
    UpstreamMassResolver,
    _coord_edges,
    _infer_netcdf_spatial_dims,
    compute_catchment_masses,
)


RESULT_DIR = Path("results/figure7")
RUNOFF_LONG_CSV = RESULT_DIR / "figure7c_global_seasonal_runoff_centroids.csv"
PRECIP_LONG_CSV = RESULT_DIR / "figure7c_global_seasonal_precipitation_centroids.csv"
BOXPLOT_DATA_CSV = RESULT_DIR / "figure7c_global_aridity_boxplot_data.csv"
SUMMARY_CSV = RESULT_DIR / "figure7c_global_aridity_boxplot_summary.csv"
PLOT_PNG = RESULT_DIR / "figure7c_global_aridity_boxplot.png"
FAILURE_CSV = RESULT_DIR / "figure7c_global_aridity_failures.csv"

ARIDITY_CSV = Path("data/attributes/aridity_basin.csv")
SEASONAL_PRECIP_PATH = Path("data/climate/climatology/seasonal_climatology_P.nc")
RUNOFF_SEASON_FILES = {
    "DJF": Path("data/hydrology/grades/seasonal/GRADES_all_pfaf_19792019_mean_DJF.csv"),
    "MAM": Path("data/hydrology/grades/seasonal/GRADES_all_pfaf_19792019_mean_MAM.csv"),
    "JJA": Path("data/hydrology/grades/seasonal/GRADES_all_pfaf_19792019_mean_JJA.csv"),
    "SON": Path("data/hydrology/grades/seasonal/GRADES_all_pfaf_19792019_mean_SON.csv"),
}
SEASON_ORDER = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
ARIDITY_GROUP_ORDER = ["dry", "subhumid", "humid"]
ARIDITY_GROUP_LABELS = {"dry": "Dry", "subhumid": "Subhumid", "humid": "Humid"}
SOURCE_ORDER = ["runoff", "precipitation"]
SOURCE_LABELS = {"runoff": "Runoff", "precipitation": "Precipitation"}


def classify_aridity(ai: float) -> str:
    """Collapse the original five AI classes into three groups."""
    if ai < 0.5:
        return "dry"
    if ai < 0.75:
        return "subhumid"
    return "humid"


def load_aridity_table() -> pd.DataFrame:
    """Load basin aridity and derive the 3-group classification."""
    aridity_df = pd.read_csv(ARIDITY_CSV, usecols=["basinname", "aridity_mean"]).copy()
    aridity_df["ai"] = pd.to_numeric(aridity_df["aridity_mean"], errors="coerce") / 10000.0
    aridity_df["aridity_group"] = aridity_df["ai"].map(classify_aridity)
    return aridity_df.rename(columns={"basinname": "basin_name"})[
        ["basin_name", "ai", "aridity_group"]
    ]


def load_seasonal_precip_grids() -> dict[str, dict]:
    """Load the 4 seasonal MSWEP climatology grids as 2D extraction dictionaries."""
    with xr.open_dataset(SEASONAL_PRECIP_PATH) as ds:
        data_var = ds["precipitation"]
        lat_dim, lon_dim = _infer_netcdf_spatial_dims(data_var)

        lat_values = np.asarray(data_var[lat_dim].values, dtype=float)
        lon_values = np.asarray(data_var[lon_dim].values, dtype=float)
        if lat_values[0] < lat_values[-1]:
            lat_values = lat_values[::-1]
            lat_reversed = True
        else:
            lat_reversed = False
        if lon_values[0] > lon_values[-1]:
            lon_values = lon_values[::-1]
            lon_reversed = True
        else:
            lon_reversed = False

        lat_edges = _coord_edges(lat_values)
        lon_edges = _coord_edges(lon_values)

        grids: dict[str, dict] = {}
        season_names = ["DJF", "MAM", "JJA", "SON"]
        for index, season in enumerate(season_names):
            array = np.asarray(data_var.isel(time=index).transpose(lat_dim, lon_dim).values, dtype=float)
            if lat_reversed:
                array = array[::-1, :]
            if lon_reversed:
                array = array[:, ::-1]
            grids[season] = {
                "data": array,
                "kind": "nc",
                "crs": "EPSG:4326",
                "lat_edges": lat_edges,
                "lon_edges": lon_edges,
                "lat_descending": True,
                "lon_ascending": True,
            }
    return grids


def build_runoff_seasonal_centroids() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute seasonal runoff centroids for all packaged basins."""
    basins = list(iter_global_basin_paths("data/basins/global"))
    basin_rivers: dict[str, pd.DataFrame] = {}
    basin_comids: dict[str, list[int]] = {}

    for basin in basins:
        river_df = read_river_attributes(
            basin.river_network,
            columns=["COMID", "lengthkm", "uparea", "up1", "up2", "up3", "up4"],
        )
        river_df = prepare_river_dataframe(river_df)
        basin_rivers[basin.name] = river_df
        basin_comids[basin.name] = river_df["COMID"].astype(int).tolist()

    results: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="figure7c_runoff_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for season, csv_path in RUNOFF_SEASON_FILES.items():
            seasonal_q = pd.read_csv(csv_path, usecols=["COMID", "qout"])
            seasonal_q["COMID"] = pd.to_numeric(seasonal_q["COMID"], errors="coerce").astype(int)
            seasonal_q["qout"] = pd.to_numeric(seasonal_q["qout"], errors="coerce")
            seasonal_q = seasonal_q.set_index("COMID")

            for basin in basins:
                q_subset = (
                    seasonal_q.reindex(basin_comids[basin.name])
                    .dropna(subset=["qout"])
                    .reset_index()
                )
                output_path = tmpdir_path / f"{basin.name}_{season}_runoff.csv"
                try:
                    basin_df = calculate_basin_centroid_from_tables(
                        river_df=basin_rivers[basin.name],
                        q_df=q_subset,
                        output_path=output_path,
                        basin_name=basin.name,
                        q_col="qout",
                        comid_col="COMID",
                        length_col="lengthkm",
                        uparea_col="uparea",
                        up_cols=["up1", "up2", "up3", "up4"],
                        min_segments=3,
                        total_column_name="total_discharge",
                        print_header=False,
                        river_prepared=True,
                        q_prepared=False,
                    )
                    basin_df.insert(1, "season", season)
                    basin_df.insert(2, "season_order", SEASON_ORDER[season])
                    basin_df.insert(3, "source", "runoff")
                    basin_df.insert(4, "data_source", "GRADES")
                    basin_df.insert(5, "aggregation", "seasonal_mean_1979_2019")
                    results.append(basin_df)
                except Exception as exc:
                    failures.append(
                        {
                            "basin_name": basin.name,
                            "season": season,
                            "source": "runoff",
                            "error": str(exc),
                        }
                    )

    long_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not long_df.empty:
        long_df = long_df[
            [
                "basin_name",
                "season",
                "season_order",
                "source",
                "data_source",
                "aggregation",
                "outlet_COMID",
                "centroid_COMID",
                "centroid_distance_km",
                "mainstem_length_km",
                "rci",
                "num_segments",
                "total_discharge",
                "outlet_uparea_km2",
            ]
        ].sort_values(["basin_name", "season_order"]).reset_index(drop=True)
    failures_df = pd.DataFrame(failures, columns=["basin_name", "season", "source", "error"])
    return long_df, failures_df


def build_precipitation_upstream_table(
    basin,
    river_df: pd.DataFrame,
    catchment_gdf,
    grid: dict,
) -> pd.DataFrame:
    """Build an in-memory COMID -> upstream precipitation-total table for one season."""
    basin_comids = set(river_df["COMID"].astype(int))
    catchment_local = catchment_gdf[catchment_gdf["COMID"].isin(basin_comids)].copy()
    local_masses = compute_catchment_masses(
        catchment_local,
        grid,
        supersample_factor=DEFAULT_CATCHMENT_MASS_SUPERSAMPLING_FACTOR,
    )
    _, upstream_map = build_river_maps(
        river_df,
        comid_col="COMID",
        uparea_col="uparea",
        up_cols=["up1", "up2", "up3", "up4"],
    )
    resolver = UpstreamMassResolver(upstream_map)

    rows = []
    for comid in sorted(basin_comids):
        rows.append(
            {
                "COMID": comid,
                "p_local": float(local_masses.get(comid, 0.0)),
                "p_upstream_total": float(resolver.upstream_mass(comid, local_masses)),
            }
        )
    return pd.DataFrame(rows)


def build_precipitation_seasonal_centroids() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute seasonal precipitation centroids from the 4-season MSWEP climatology."""
    basins = list(iter_global_basin_paths("data/basins/global"))
    seasonal_grids = load_seasonal_precip_grids()
    results: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="figure7c_precip_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for basin in basins:
            river_df = read_river_attributes(
                basin.river_network,
                columns=["COMID", "lengthkm", "uparea", "up1", "up2", "up3", "up4"],
            )
            river_df = prepare_river_dataframe(river_df)
            catchment_gdf = pyogrio.read_dataframe(basin.catchments, columns=["COMID"])
            catchment_gdf["COMID"] = pd.to_numeric(catchment_gdf["COMID"], errors="coerce").astype(int)

            for season, grid in seasonal_grids.items():
                output_path = tmpdir_path / f"{basin.name}_{season}_precip.csv"
                try:
                    precip_df = build_precipitation_upstream_table(
                        basin=basin,
                        river_df=river_df,
                        catchment_gdf=catchment_gdf,
                        grid=grid,
                    )
                    basin_df = calculate_basin_centroid_from_tables(
                        river_df=river_df,
                        q_df=precip_df[["COMID", "p_upstream_total"]],
                        output_path=output_path,
                        basin_name=basin.name,
                        q_col="p_upstream_total",
                        comid_col="COMID",
                        length_col="lengthkm",
                        uparea_col="uparea",
                        up_cols=["up1", "up2", "up3", "up4"],
                        min_segments=3,
                        total_column_name="total_mass",
                        min_total=0.0,
                        print_header=False,
                        river_prepared=True,
                        q_prepared=True,
                    )
                    basin_df.insert(1, "season", season)
                    basin_df.insert(2, "season_order", SEASON_ORDER[season])
                    basin_df.insert(3, "source", "precipitation")
                    basin_df.insert(4, "data_source", "MSWEP")
                    basin_df.insert(5, "aggregation", "seasonal_climatology")
                    results.append(basin_df)
                except Exception as exc:
                    failures.append(
                        {
                            "basin_name": basin.name,
                            "season": season,
                            "source": "precipitation",
                            "error": str(exc),
                        }
                    )

    long_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not long_df.empty:
        long_df = long_df[
            [
                "basin_name",
                "season",
                "season_order",
                "source",
                "data_source",
                "aggregation",
                "outlet_COMID",
                "centroid_COMID",
                "centroid_distance_km",
                "mainstem_length_km",
                "rci",
                "num_segments",
                "total_mass",
                "outlet_uparea_km2",
            ]
        ].sort_values(["basin_name", "season_order"]).reset_index(drop=True)
    failures_df = pd.DataFrame(failures, columns=["basin_name", "season", "source", "error"])
    return long_df, failures_df


def build_boxplot_outputs(
    runoff_df: pd.DataFrame,
    precip_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge seasonal centroids with aridity groups and prepare boxplot tables."""
    aridity_df = load_aridity_table()
    combined = pd.concat(
        [
            runoff_df[["basin_name", "season", "season_order", "source", "rci", "centroid_distance_km", "mainstem_length_km"]],
            precip_df[["basin_name", "season", "season_order", "source", "rci", "centroid_distance_km", "mainstem_length_km"]],
        ],
        ignore_index=True,
    )
    combined = combined.merge(aridity_df, on="basin_name", how="left", validate="many_to_one")
    combined["rci_percent"] = combined["rci"] * 100.0
    combined["aridity_group"] = pd.Categorical(
        combined["aridity_group"],
        categories=ARIDITY_GROUP_ORDER,
        ordered=True,
    )
    combined["source"] = pd.Categorical(
        combined["source"],
        categories=SOURCE_ORDER,
        ordered=True,
    )
    combined = combined.sort_values(["aridity_group", "source", "basin_name", "season_order"]).reset_index(drop=True)

    summary = (
        combined.groupby(["aridity_group", "source"], observed=True)
        .agg(
            n=("rci", "size"),
            basin_n=("basin_name", "nunique"),
            mean_rci=("rci", "mean"),
            median_rci=("rci", "median"),
            mean_rci_percent=("rci_percent", "mean"),
            median_rci_percent=("rci_percent", "median"),
            mean_ai=("ai", "mean"),
        )
        .reset_index()
    )
    return combined, summary


def plot_boxplot(boxplot_df: pd.DataFrame) -> None:
    """Draw the 3-group aridity boxplot for seasonal runoff and precipitation RCI."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    colors = {"runoff": "#2b6cb0", "precipitation": "#d97706"}
    group_positions = np.arange(len(ARIDITY_GROUP_ORDER), dtype=float)
    width = 0.32

    legend_handles = []
    for offset, source in zip([-width / 2, width / 2], SOURCE_ORDER):
        data = []
        positions = []
        for idx, group in enumerate(ARIDITY_GROUP_ORDER):
            values = boxplot_df.loc[
                (boxplot_df["aridity_group"] == group) & (boxplot_df["source"] == source),
                "rci_percent",
            ].dropna()
            if values.empty:
                continue
            data.append(values.to_numpy())
            positions.append(group_positions[idx] + offset)

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=True,
            boxprops={"facecolor": colors[source], "alpha": 0.75},
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": colors[source], "linewidth": 1.0},
            capprops={"color": colors[source], "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markersize": 2.2,
                "markerfacecolor": colors[source],
                "markeredgecolor": colors[source],
                "alpha": 0.5,
            },
        )
        legend_handles.append(bp["boxes"][0])

    ax.set_xlabel("Aridity Group")
    ax.set_ylabel("Seasonal centroid RCI (%)")
    ax.set_xticks(group_positions)
    ax.set_xticklabels([ARIDITY_GROUP_LABELS[group] for group in ARIDITY_GROUP_ORDER])
    ax.legend(
        legend_handles,
        [SOURCE_LABELS[source] for source in SOURCE_ORDER],
        title="Source",
        frameon=False,
    )
    ax.set_title("Seasonal runoff and precipitation RCIs by aridity group")
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_figure7c_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full Figure 7c preprocessing workflow."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    runoff_df, runoff_failures = build_runoff_seasonal_centroids()
    precip_df, precip_failures = build_precipitation_seasonal_centroids()
    failures_df = pd.concat([runoff_failures, precip_failures], ignore_index=True)

    boxplot_df, summary_df = build_boxplot_outputs(runoff_df, precip_df)

    runoff_df.to_csv(RUNOFF_LONG_CSV, index=False)
    precip_df.to_csv(PRECIP_LONG_CSV, index=False)
    boxplot_df.to_csv(BOXPLOT_DATA_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    failures_df.to_csv(FAILURE_CSV, index=False)
    plot_boxplot(boxplot_df)
    return runoff_df, precip_df, boxplot_df, summary_df


if __name__ == "__main__":
    build_figure7c_outputs()
