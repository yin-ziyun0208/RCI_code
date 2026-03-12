"""Runoff-based centroid workflows using packaged basin folders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyogrio

from .basin_io import BasinPaths, iter_global_basin_paths
from .centroid import calculate_midpoint_centroid, calculate_rci, find_centroid_comid
from .mainstem import DEFAULT_UP_COLS, build_river_maps, find_basin_outlet, trace_main_stem_with_maps


def read_river_attributes(
    river_network_path: str | Path,
    columns: list[str],
) -> pd.DataFrame:
    """Read only the required river-network attributes."""
    return pyogrio.read_dataframe(
        river_network_path,
        columns=columns,
        read_geometry=False,
    )


def prepare_river_dataframe(
    river_df: pd.DataFrame,
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Normalize river-network attributes into a lightweight table."""
    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    required_cols = [comid_col, length_col, uparea_col, *up_cols]
    missing = [col for col in required_cols if col not in river_df.columns]
    if missing:
        raise ValueError(f"Missing river-network columns: {missing}")

    river_df = pd.DataFrame(river_df.loc[:, required_cols]).copy()
    river_df[comid_col] = pd.to_numeric(river_df[comid_col], errors="coerce").astype(int)
    river_df[length_col] = pd.to_numeric(river_df[length_col], errors="coerce").fillna(0.0)
    river_df[uparea_col] = pd.to_numeric(river_df[uparea_col], errors="coerce").fillna(0.0)
    for up_col in up_cols:
        river_df[up_col] = pd.to_numeric(river_df[up_col], errors="coerce").fillna(0).astype(int)
    return river_df


def prepare_discharge_dataframe(
    q_df: pd.DataFrame,
    comid_col: str = "COMID",
    q_col: str = "qout",
) -> pd.DataFrame:
    """Normalize the discharge table into COMID/q pairs."""
    if q_col not in q_df.columns:
        raise ValueError(f"Discharge column '{q_col}' not found. Available: {list(q_df.columns)}")

    q_df = pd.DataFrame(q_df.loc[:, [comid_col, q_col]]).copy()
    q_df[comid_col] = pd.to_numeric(q_df[comid_col], errors="coerce").astype(int)
    q_df[q_col] = pd.to_numeric(q_df[q_col], errors="coerce")
    return q_df.dropna()


def calculate_incremental_discharge(
    river_df: pd.DataFrame,
    mainstem_comids: list[int],
    q_df: pd.DataFrame,
    comid_col: str = "COMID",
    q_col: str = "qout",
    length_col: str = "lengthkm",
    length_dict: Optional[dict[int, float]] = None,
    q_dict: Optional[dict[int, float]] = None,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Compute valid mainstem reaches, lengths, cumulative lengths, and delta_q."""
    if length_dict is None:
        length_dict = river_df.set_index(comid_col)[length_col].to_dict()
    if q_dict is None:
        q_dict = q_df.set_index(comid_col)[q_col].to_dict()

    valid_mainstem_comids: list[int] = []
    segment_lengths: list[float] = []
    cumulative_lengths: list[float] = []
    q_values: list[float] = []
    cumulative = 0.0

    for comid in mainstem_comids:
        if comid not in length_dict or comid not in q_dict:
            continue
        valid_mainstem_comids.append(comid)
        segment_length = float(length_dict[comid])
        segment_lengths.append(segment_length)
        cumulative += segment_length
        cumulative_lengths.append(cumulative)
        q_values.append(float(q_dict[comid]))

    delta_q: list[float] = []
    for idx, current_q in enumerate(q_values):
        if idx == 0:
            delta_q.append(current_q)
        else:
            delta_q.append(current_q - q_values[idx - 1])

    return valid_mainstem_comids, segment_lengths, cumulative_lengths, delta_q


def calculate_basin_centroid_from_tables(
    river_df: pd.DataFrame,
    q_df: pd.DataFrame,
    output_path: str | Path,
    basin_name: str,
    q_col: str = "qout",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    min_segments: int = 3,
    total_column_name: str = "total_discharge",
    min_total: float | None = None,
    print_header: bool = True,
    river_prepared: bool = False,
    q_prepared: bool = False,
) -> pd.DataFrame:
    """Calculate a basin-scale continuous RCI from in-memory tables."""
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

    if print_header:
        print("=" * 60)
        print(f"Calculating centroid for basin: {basin_name}")
        print("=" * 60)

    outlet_comid = find_basin_outlet(river_df, comid_col=comid_col, uparea_col=uparea_col)
    outlet_area = float(river_df.loc[river_df[comid_col] == outlet_comid, uparea_col].iloc[0])
    uparea_map, upstream_map = build_river_maps(
        river_df,
        comid_col=comid_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
    )
    mainstem_comids = trace_main_stem_with_maps(outlet_comid, uparea_map, upstream_map)
    valid_comids, segment_lengths, cumulative_lengths, delta_q = calculate_incremental_discharge(
        river_df,
        mainstem_comids,
        q_df,
        comid_col=comid_col,
        q_col=q_col,
        length_col=length_col,
    )

    if len(valid_comids) < min_segments:
        raise ValueError(
            f"Main stem has only {len(valid_comids)} effective segments "
            f"(minimum {min_segments} required)"
        )

    total_value = float(np.sum(delta_q))
    if min_total is not None and total_value <= min_total:
        raise ValueError(
            f"Total {total_column_name} is {total_value:.6g}, "
            f"which does not exceed the required threshold {min_total:.6g}"
        )

    max_length, centroid = calculate_midpoint_centroid(segment_lengths, cumulative_lengths, delta_q)
    centroid_comid = find_centroid_comid(centroid, max_length, valid_comids, cumulative_lengths)
    rci = calculate_rci(centroid, max_length)

    results = pd.DataFrame(
        [
            {
                "basin_name": basin_name,
                "outlet_COMID": outlet_comid,
                "centroid_COMID": centroid_comid,
                "centroid_distance_km": centroid,
                "mainstem_length_km": max_length,
                "rci": rci,
                "num_segments": len(valid_comids),
                total_column_name: total_value,
                "outlet_uparea_km2": outlet_area,
            }
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def run_single_basin_runoff_rci(
    basin: BasinPaths,
    output_path: str | Path,
    q_col: str = "qout",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    min_segments: int = 3,
) -> pd.DataFrame:
    """Run basin-scale runoff RCI for one packaged basin."""
    if basin.discharge_csv is None:
        raise FileNotFoundError(f"No discharge CSV found for basin {basin.name}")

    river_df = read_river_attributes(
        basin.river_network,
        columns=[comid_col, length_col, uparea_col, *(up_cols or DEFAULT_UP_COLS)],
    )
    q_df = pd.read_csv(basin.discharge_csv, usecols=[comid_col, q_col])
    return calculate_basin_centroid_from_tables(
        river_df=river_df,
        q_df=q_df,
        output_path=output_path,
        basin_name=basin.name,
        q_col=q_col,
        comid_col=comid_col,
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols,
        min_segments=min_segments,
        total_column_name="total_discharge",
        print_header=True,
    )


def run_batch_basin_runoff_rci(
    global_dir: str = "data/basins/global",
    output_dir: str = "outputs/batch/runoff",
    q_col: str = "qout",
    comid_col: str = "COMID",
    length_col: str = "lengthkm",
    uparea_col: str = "uparea",
    up_cols: Optional[list[str]] = None,
    min_segments: int = 3,
    basin_names: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run basin-scale runoff RCI across packaged global basins."""
    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    output_dir_path = Path(output_dir)
    per_basin_dir = output_dir_path / "per_basin"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    per_basin_dir.mkdir(parents=True, exist_ok=True)

    results: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    basins = list(iter_global_basin_paths(global_dir, basin_names=basin_names))

    print("=" * 72)
    print(f"Running runoff RCI for {len(basins)} packaged basins")
    print("=" * 72)

    for index, basin in enumerate(basins, start=1):
        print(f"\n[{index}/{len(basins)}] {basin.name}")
        output_path = per_basin_dir / f"{basin.name}_centroid_results.csv"
        try:
            basin_df = run_single_basin_runoff_rci(
                basin=basin,
                output_path=output_path,
                q_col=q_col,
                comid_col=comid_col,
                length_col=length_col,
                uparea_col=uparea_col,
                up_cols=up_cols,
                min_segments=min_segments,
            )
            basin_df.insert(1, "river_file", basin.river_network.name)
            basin_df.insert(2, "source_mode", basin.mode)
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
    combined_df.to_csv(output_dir_path / "global_centroid_results.csv", index=False)
    failures_df.to_csv(output_dir_path / "global_centroid_failures.csv", index=False)
    return combined_df, failures_df
