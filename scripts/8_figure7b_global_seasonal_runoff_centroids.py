"""Build the Figure 7b seasonal runoff-centroid tables for 40 global basins."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd


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


RESULT_DIR = Path("results/figure7")
LONG_CSV = RESULT_DIR / "figure7b_global_seasonal_runoff_centroids.csv"
WIDE_CSV = RESULT_DIR / "figure7b_global_seasonal_runoff_centroids_wide.csv"
FAILURE_CSV = RESULT_DIR / "figure7b_global_seasonal_runoff_centroid_failures.csv"

SEASON_FILES = {
    "DJF": Path("data/hydrology/grades/seasonal/GRADES_all_pfaf_19792019_mean_DJF.csv"),
    "MAM": Path("data/hydrology/grades/seasonal/GRADES_all_pfaf_19792019_mean_MAM.csv"),
    "JJA": Path("data/hydrology/grades/seasonal/GRADES_all_pfaf_19792019_mean_JJA.csv"),
    "SON": Path("data/hydrology/grades/seasonal/GRADES_all_pfaf_19792019_mean_SON.csv"),
}
SEASON_ORDER = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}


def build_figure7b_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

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

    with tempfile.TemporaryDirectory(prefix="figure7b_runoff_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for season, csv_path in SEASON_FILES.items():
            print(f"Loading seasonal discharge table: {season} from {csv_path}")
            seasonal_q = pd.read_csv(csv_path, usecols=["COMID", "qout"])
            seasonal_q["COMID"] = pd.to_numeric(seasonal_q["COMID"], errors="coerce").astype(int)
            seasonal_q["qout"] = pd.to_numeric(seasonal_q["qout"], errors="coerce")
            seasonal_q = seasonal_q.set_index("COMID")

            print(f"Running {season} centroids for {len(basins)} basins")
            for index, basin in enumerate(basins, start=1):
                print(f"[{season} {index}/{len(basins)}] {basin.name}")
                q_subset = (
                    seasonal_q.reindex(basin_comids[basin.name])
                    .dropna(subset=["qout"])
                    .reset_index()
                )
                output_path = tmpdir_path / f"{basin.name}_{season}_centroid.csv"
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
                    basin_df.insert(3, "data_source", "GRADES")
                    basin_df.insert(4, "aggregation", "seasonal_mean_1979_2019")
                    results.append(basin_df)
                except Exception as exc:
                    failures.append(
                        {
                            "basin_name": basin.name,
                            "season": season,
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

    wide_df = (
        long_df.pivot(
            index="basin_name",
            columns="season",
            values=["centroid_distance_km", "rci", "centroid_COMID", "total_discharge"],
        )
        if not long_df.empty
        else pd.DataFrame()
    )
    if not wide_df.empty:
        wide_df.columns = [f"{metric}_{season}" for metric, season in wide_df.columns]
        wide_df = wide_df.reset_index()

    failures_df = pd.DataFrame(failures, columns=["basin_name", "season", "error"])

    long_df.to_csv(LONG_CSV, index=False)
    wide_df.to_csv(WIDE_CSV, index=False)
    failures_df.to_csv(FAILURE_CSV, index=False)
    return long_df, wide_df, failures_df


if __name__ == "__main__":
    build_figure7b_tables()
