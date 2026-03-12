"""Build the Figure 5 slope table using the mainstem DEM longitudinal profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULT_DIR = Path("results/figure5")
PQ_DISTANCE_CSV = RESULT_DIR / "figure5_global_pq_centroid_distance.csv"
SLOPE_CSV = RESULT_DIR / "figure5_global_slope_between_pq.csv"
DEM_PROFILE_CSV = Path("data/terrain/merged_mainstream_dem_profile.csv")


def _pick_profile_row(
    profile_df: pd.DataFrame,
    centroid_comid: int,
    centroid_rci: float,
) -> tuple[pd.Series, str]:
    """Match a centroid to the DEM profile using exact COMID first, then nearest normalized position."""
    exact = profile_df[profile_df["COMID"] == int(centroid_comid)]
    if not exact.empty:
        return exact.iloc[0], "exact"

    target_normalized_length = 1.0 - float(centroid_rci)
    nearest_idx = (profile_df["normalized_length"] - target_normalized_length).abs().idxmin()
    return profile_df.loc[nearest_idx], "nearest_normalized"


def _compute_segment_slope_m_per_km(
    profile_segment: pd.DataFrame,
    fallback_drop_m: float,
    fallback_length_km: float,
) -> float:
    """
    Estimate the along-profile slope between P and Q centroids.

    The preferred metric is the absolute OLS slope of elevation against along-mainstem
    distance within the centroid segment, reported in ``m/km``. If too few DEM points
    are available, fall back to absolute end-point elevation drop divided by the
    centroid-segment length.
    """
    if len(profile_segment) >= 2:
        x = profile_segment["cumulative_value"].to_numpy(dtype=float)
        y = profile_segment["elev"].to_numpy(dtype=float)
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.unique(x).size >= 2:
            return float(abs(np.polyfit(x, y, 1)[0]))

    if fallback_length_km > 0:
        return float(abs(fallback_drop_m) / fallback_length_km)
    return float("nan")


def _slope_m_per_km_to_rad(slope_m_per_km: float) -> float:
    """Convert slope from ``m/km`` to radians."""
    if not np.isfinite(slope_m_per_km):
        return float("nan")
    return float(np.arctan(slope_m_per_km / 1000.0))


def build_slope_table() -> pd.DataFrame:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    pq_df = pd.read_csv(PQ_DISTANCE_CSV)
    dem_df = pd.read_csv(
        DEM_PROFILE_CSV,
        usecols=["COMID", "elev", "cumulative_value", "normalized_length", "rivername"],
    )
    dem_df["COMID"] = pd.to_numeric(dem_df["COMID"], errors="coerce").astype(int)
    dem_df["elev"] = pd.to_numeric(dem_df["elev"], errors="coerce")
    dem_df["cumulative_value"] = pd.to_numeric(dem_df["cumulative_value"], errors="coerce")
    dem_df["normalized_length"] = pd.to_numeric(dem_df["normalized_length"], errors="coerce")

    rows: list[dict[str, object]] = []

    for row in pq_df.itertuples(index=False):
        basin_profile = dem_df[dem_df["rivername"] == row.basin_name].copy()
        if basin_profile.empty:
            raise ValueError(f"No DEM longitudinal profile found for basin: {row.basin_name}")
        basin_profile = basin_profile.sort_values("normalized_length").reset_index(drop=True)

        runoff_profile_row, runoff_match = _pick_profile_row(
            basin_profile,
            centroid_comid=int(row.runoff_centroid_COMID),
            centroid_rci=float(row.runoff_rci),
        )
        precip_profile_row, precip_match = _pick_profile_row(
            basin_profile,
            centroid_comid=int(row.precipitation_centroid_COMID),
            centroid_rci=float(row.precipitation_rci),
        )

        runoff_norm = float(runoff_profile_row["normalized_length"])
        precip_norm = float(precip_profile_row["normalized_length"])
        lower_norm = min(runoff_norm, precip_norm)
        upper_norm = max(runoff_norm, precip_norm)
        profile_segment = basin_profile[
            (basin_profile["normalized_length"] >= lower_norm)
            & (basin_profile["normalized_length"] <= upper_norm)
        ].copy()

        if float(row.runoff_centroid_distance_km) > float(row.precipitation_centroid_distance_km):
            upstream_profile_row = runoff_profile_row
            downstream_profile_row = precip_profile_row
        else:
            upstream_profile_row = precip_profile_row
            downstream_profile_row = runoff_profile_row

        elevation_drop_m = abs(float(upstream_profile_row["elev"] - downstream_profile_row["elev"]))
        centroid_segment_length_km = float(row.centroid_segment_length_km)
        slope_m_per_km = _compute_segment_slope_m_per_km(
            profile_segment=profile_segment,
            fallback_drop_m=elevation_drop_m,
            fallback_length_km=centroid_segment_length_km,
        )
        slope_rad = _slope_m_per_km_to_rad(slope_m_per_km)

        rows.append(
            {
                "basin_name": row.basin_name,
                "runoff_rci": float(row.runoff_rci),
                "precipitation_rci": float(row.precipitation_rci),
                "abs_pq_distance_rci": float(row.abs_pq_distance_rci),
                "pq_distance_group": row.pq_distance_group,
                "runoff_centroid_distance_km": float(row.runoff_centroid_distance_km),
                "precipitation_centroid_distance_km": float(row.precipitation_centroid_distance_km),
                "centroid_segment_length_km": centroid_segment_length_km,
                "runoff_profile_match": runoff_match,
                "precipitation_profile_match": precip_match,
                "runoff_profile_COMID": int(runoff_profile_row["COMID"]),
                "precipitation_profile_COMID": int(precip_profile_row["COMID"]),
                "runoff_profile_elevation_m": float(runoff_profile_row["elev"]),
                "precipitation_profile_elevation_m": float(precip_profile_row["elev"]),
                "elevation_drop_m": elevation_drop_m,
                "centroid_segment_slope_m_per_km": slope_m_per_km,
                "centroid_segment_slope_rad": slope_rad,
            }
        )

    slope_df = pd.DataFrame(rows).sort_values("basin_name").reset_index(drop=True)
    slope_df.to_csv(SLOPE_CSV, index=False)
    return slope_df


if __name__ == "__main__":
    build_slope_table()
