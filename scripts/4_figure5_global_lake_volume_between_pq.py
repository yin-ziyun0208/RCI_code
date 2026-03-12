"""Summarize HydroLAKES storage between precipitation and runoff centroids for Figure 5."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pyogrio
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.pfaf import PfafstetterCoder


RESULT_DIR = Path("results/figure5")
PQ_DISTANCE_CSV = RESULT_DIR / "figure5_global_pq_centroid_distance.csv"
LAKE_SUMMARY_CSV = RESULT_DIR / "figure5_global_lake_volume_between_pq.csv"
HYDROLAKES_POINTS = Path("data/lakes/HydroLAKES_points_v10.shp")


def _build_upstream_closure_map(river_network_path: Path) -> tuple[pd.DataFrame, PfafstetterCoder]:
    river_df = pyogrio.read_dataframe(
        river_network_path,
        columns=["COMID", "uparea", "NextDownID", "up1", "up2", "up3", "up4", "lengthkm"],
        read_geometry=False,
    )
    river_df["COMID"] = pd.to_numeric(river_df["COMID"], errors="coerce").astype(int)
    river_df["uparea"] = pd.to_numeric(river_df["uparea"], errors="coerce").fillna(0.0)
    river_df["NextDownID"] = pd.to_numeric(river_df["NextDownID"], errors="coerce").fillna(0).astype(int)
    river_df["lengthkm"] = pd.to_numeric(river_df["lengthkm"], errors="coerce").fillna(0.0)
    for col in ["up1", "up2", "up3", "up4"]:
        river_df[col] = pd.to_numeric(river_df[col], errors="coerce").fillna(0).astype(int)
    coder = PfafstetterCoder(
        river_df=river_df,
        comid_col="COMID",
        uparea_col="uparea",
        down_col="NextDownID",
        up_cols=["up1", "up2", "up3", "up4"],
        max_level=1,
        min_unit_reaches=1,
    )
    return river_df, coder

def build_lake_volume_summary() -> pd.DataFrame:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pq_df = pd.read_csv(PQ_DISTANCE_CSV)
    print(f"Loading HydroLAKES points from {HYDROLAKES_POINTS} ...")
    all_lakes = pyogrio.read_dataframe(
        HYDROLAKES_POINTS,
        columns=["Hylak_id", "Lake_name", "Lake_area", "Vol_total", "Vol_res"],
    )
    all_lakes["Vol_total"] = pd.to_numeric(all_lakes["Vol_total"], errors="coerce").fillna(0.0)
    all_lakes["Vol_res"] = pd.to_numeric(all_lakes["Vol_res"], errors="coerce").fillna(0.0)
    lakes_sindex = all_lakes.sindex
    print(f"Loaded {len(all_lakes)} lake points")

    summary_rows: list[dict[str, object]] = []

    for index, row in enumerate(pq_df.itertuples(index=False), start=1):
        print(f"[{index}/{len(pq_df)}] {row.basin_name}")
        basin_dir = Path("data/basins/global") / str(row.basin_name)
        river_path = basin_dir / "river_network.shp"
        catchment_path = basin_dir / "catchments.shp"
        discharge_path = basin_dir / "grades_discharge.csv"

        river_df, coder = _build_upstream_closure_map(river_path)
        catchment_gdf = pyogrio.read_dataframe(catchment_path)
        catchment_gdf["COMID"] = pd.to_numeric(catchment_gdf["COMID"], errors="coerce").astype(int)
        discharge_df = pd.read_csv(discharge_path, usecols=["COMID", "qout"])
        discharge_df["COMID"] = pd.to_numeric(discharge_df["COMID"], errors="coerce").astype(int)
        discharge_df["qout"] = pd.to_numeric(discharge_df["qout"], errors="coerce")
        discharge_map = dict(zip(discharge_df["COMID"], discharge_df["qout"]))

        downstream_comid = (
            int(row.runoff_centroid_COMID)
            if float(row.runoff_centroid_distance_km) <= float(row.precipitation_centroid_distance_km)
            else int(row.precipitation_centroid_COMID)
        )
        upstream_comid = (
            int(row.runoff_centroid_COMID)
            if float(row.runoff_centroid_distance_km) > float(row.precipitation_centroid_distance_km)
            else int(row.precipitation_centroid_COMID)
        )

        downstream_closure = set(coder.upstream_closure(downstream_comid))
        upstream_closure = set(coder.upstream_closure(upstream_comid))
        pq_domain_comids = sorted(downstream_closure - upstream_closure)

        if pq_domain_comids:
            pq_domain_gdf = catchment_gdf[catchment_gdf["COMID"].isin(pq_domain_comids)].copy()
            domain_bounds = tuple(float(v) for v in pq_domain_gdf.total_bounds)

            candidate_idx = list(lakes_sindex.query(box(*domain_bounds), predicate="intersects"))
            lake_points = all_lakes.iloc[candidate_idx].copy() if candidate_idx else all_lakes.iloc[0:0].copy()
            if not lake_points.empty:
                joined = lake_points.sjoin(
                    pq_domain_gdf[["geometry"]],
                    how="inner",
                    predicate="within",
                )
                lake_points = lake_points.loc[joined.index.unique()].copy()
            lake_ids = sorted(pd.to_numeric(lake_points["Hylak_id"], errors="coerce").dropna().astype(int).tolist())
            lake_count = int(len(lake_points))
            lake_volume_total = float(lake_points["Vol_total"].sum())
            lake_volume_reservoir = float(lake_points["Vol_res"].sum())
        else:
            domain_bounds = (float("nan"), float("nan"), float("nan"), float("nan"))
            lake_ids = []
            lake_count = 0
            lake_volume_total = 0.0
            lake_volume_reservoir = 0.0

        centroid_segment_discharge = float(discharge_map.get(int(row.runoff_centroid_COMID), float("nan")))
        lake_volume_to_discharge_ratio = (
            lake_volume_total / centroid_segment_discharge
            if pd.notna(centroid_segment_discharge) and centroid_segment_discharge != 0
            else float("nan")
        )

        summary_rows.append(
            {
                "basin_name": row.basin_name,
                "mainstem_length_km": float(row.mainstem_length_km),
                "runoff_centroid_COMID": int(row.runoff_centroid_COMID),
                "precipitation_centroid_COMID": int(row.precipitation_centroid_COMID),
                "runoff_centroid_distance_km": float(row.runoff_centroid_distance_km),
                "precipitation_centroid_distance_km": float(row.precipitation_centroid_distance_km),
                "runoff_rci": float(row.runoff_rci),
                "precipitation_rci": float(row.precipitation_rci),
                "signed_q_minus_p_distance_km": float(row.signed_q_minus_p_distance_km),
                "abs_pq_distance_km": float(row.abs_pq_distance_km),
                "signed_q_minus_p_rci": float(row.signed_q_minus_p_rci),
                "abs_pq_distance_rci": float(row.abs_pq_distance_rci),
                "pq_distance_group": row.pq_distance_group,
                "pq_domain_downstream_COMID": downstream_comid,
                "pq_domain_upstream_COMID": upstream_comid,
                "pq_domain_num_reaches": len(pq_domain_comids),
                "centroid_segment_length_km": float(row.centroid_segment_length_km),
                "lake_count": lake_count,
                "hydrolakes_total_volume_sum": lake_volume_total,
                "hydrolakes_reservoir_volume_sum": lake_volume_reservoir,
                "runoff_centroid_discharge": centroid_segment_discharge,
                "lake_volume_to_discharge_ratio": lake_volume_to_discharge_ratio,
            }
        )
        pd.DataFrame(summary_rows).sort_values("basin_name").to_csv(LAKE_SUMMARY_CSV, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("basin_name").reset_index(drop=True)
    summary_df.to_csv(LAKE_SUMMARY_CSV, index=False)
    return summary_df


if __name__ == "__main__":
    build_lake_volume_summary()
