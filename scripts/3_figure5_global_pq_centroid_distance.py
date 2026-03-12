"""Build the global precipitation-runoff centroid distance table for Figure 5."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.continuous_source import run_batch_basin_runoff_rci
from core.gridded_source import run_batch_basin_precipitation_level0


RESULT_DIR = Path("results/figure5")
PAIR_CSV = RESULT_DIR / "figure5_global_pq_centroid_distance.csv"


def _load_or_build_runoff_results() -> pd.DataFrame:
    path = Path("results/figure4/figure4_global_runoff_rci.csv")
    if path.exists():
        return pd.read_csv(path)

    result_df, _ = run_batch_basin_runoff_rci(
        global_dir="data/basins/global",
        output_dir="results/figure4",
        min_segments=3,
    )
    return result_df[
        [
            "basin_name",
            "outlet_COMID",
            "centroid_COMID",
            "centroid_distance_km",
            "mainstem_length_km",
            "rci",
            "num_segments",
            "total_discharge",
            "outlet_uparea_km2",
        ]
    ].copy()


def _load_or_build_precipitation_results() -> pd.DataFrame:
    path = Path("outputs/batch/precipitation_level0/global_precipitation_level0_results.csv")
    if path.exists():
        return pd.read_csv(path)

    result_df, _ = run_batch_basin_precipitation_level0(
        global_dir="data/basins/global",
        gridded_data_path="data/climate/climatology/mswep_precipitation_mean.nc",
        output_dir="outputs/batch/precipitation_level0",
        variable="precipitation",
        reduction="mean",
        min_segments=3,
    )
    return result_df


def build_figure5_pair_table() -> pd.DataFrame:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    runoff_df = _load_or_build_runoff_results().rename(
        columns={
            "outlet_COMID": "runoff_outlet_COMID",
            "centroid_COMID": "runoff_centroid_COMID",
            "centroid_distance_km": "runoff_centroid_distance_km",
            "mainstem_length_km": "runoff_mainstem_length_km",
            "rci": "runoff_rci",
            "num_segments": "runoff_num_segments",
            "total_discharge": "runoff_total_discharge",
            "outlet_uparea_km2": "outlet_uparea_km2",
        }
    )
    precip_df = _load_or_build_precipitation_results().rename(
        columns={
            "outlet_COMID": "precipitation_outlet_COMID",
            "centroid_COMID": "precipitation_centroid_COMID",
            "centroid_distance_km": "precipitation_centroid_distance_km",
            "mainstem_length_km": "precipitation_mainstem_length_km",
            "rci": "precipitation_rci",
            "num_segments": "precipitation_num_segments",
            "total_mass": "precipitation_total_mass",
        }
    )

    merged = runoff_df.merge(
        precip_df[
            [
                "basin_name",
                "precipitation_outlet_COMID",
                "precipitation_centroid_COMID",
                "precipitation_centroid_distance_km",
                "precipitation_mainstem_length_km",
                "precipitation_rci",
                "precipitation_num_segments",
                "precipitation_total_mass",
            ]
        ],
        on="basin_name",
        how="inner",
    )

    merged["mainstem_length_km"] = merged["runoff_mainstem_length_km"]
    merged["signed_q_minus_p_distance_km"] = (
        merged["runoff_centroid_distance_km"] - merged["precipitation_centroid_distance_km"]
    )
    merged["abs_pq_distance_km"] = merged["signed_q_minus_p_distance_km"].abs()
    merged["signed_q_minus_p_rci"] = merged["runoff_rci"] - merged["precipitation_rci"]
    merged["abs_pq_distance_rci"] = merged["signed_q_minus_p_rci"].abs()

    merged["downstream_centroid_type"] = merged.apply(
        lambda row: "runoff"
        if row["runoff_centroid_distance_km"] <= row["precipitation_centroid_distance_km"]
        else "precipitation",
        axis=1,
    )
    merged["upstream_centroid_type"] = merged.apply(
        lambda row: "runoff"
        if row["runoff_centroid_distance_km"] > row["precipitation_centroid_distance_km"]
        else "precipitation",
        axis=1,
    )
    merged["centroid_segment_downstream_distance_km"] = merged[
        ["runoff_centroid_distance_km", "precipitation_centroid_distance_km"]
    ].min(axis=1)
    merged["centroid_segment_upstream_distance_km"] = merged[
        ["runoff_centroid_distance_km", "precipitation_centroid_distance_km"]
    ].max(axis=1)
    merged["centroid_segment_length_km"] = (
        merged["centroid_segment_upstream_distance_km"]
        - merged["centroid_segment_downstream_distance_km"]
    )

    threshold = float(merged["abs_pq_distance_rci"].mean())
    merged["pq_distance_group_threshold_rci"] = threshold
    merged["pq_distance_group"] = merged["abs_pq_distance_rci"].apply(
        lambda value: "short" if value <= threshold else "long"
    )

    final_df = merged[
        [
            "basin_name",
            "mainstem_length_km",
            "runoff_centroid_COMID",
            "precipitation_centroid_COMID",
            "runoff_centroid_distance_km",
            "precipitation_centroid_distance_km",
            "runoff_rci",
            "precipitation_rci",
            "signed_q_minus_p_distance_km",
            "abs_pq_distance_km",
            "signed_q_minus_p_rci",
            "abs_pq_distance_rci",
            "downstream_centroid_type",
            "upstream_centroid_type",
            "centroid_segment_length_km",
            "pq_distance_group_threshold_rci",
            "pq_distance_group",
        ]
    ].sort_values("basin_name").reset_index(drop=True)

    final_df.to_csv(PAIR_CSV, index=False)
    return final_df


if __name__ == "__main__":
    build_figure5_pair_table()
