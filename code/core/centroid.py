"""Shared centroid helpers for continuous and gridded workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


def ensure_output_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def default_single_output_dir(workflow: str, basin_name: str) -> Path:
    output_dir = Path("outputs") / "single" / basin_name / workflow
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def default_batch_output_dir(workflow: str) -> Path:
    output_dir = Path("outputs") / "batch" / workflow
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def keep_multilevel_rows(result_df: pd.DataFrame, include_level0: bool = False) -> pd.DataFrame:
    if result_df.empty or "level" not in result_df.columns:
        return result_df.copy()
    if include_level0:
        return result_df.copy()
    return result_df[result_df["level"] > 0].copy()


def calculate_midpoint_centroid(
    segment_lengths: list[float],
    cumulative_lengths: list[float],
    delta_values: list[float],
) -> tuple[float, float]:
    """Compute a midpoint-based one-dimensional centroid along the main stem."""
    if len(segment_lengths) == 0 or len(cumulative_lengths) == 0 or len(delta_values) == 0:
        return 0.0, 0.0

    segment_array = np.asarray(segment_lengths, dtype=float)
    cumulative_array = np.asarray(cumulative_lengths, dtype=float)
    delta_array = np.asarray(delta_values, dtype=float)

    max_length = float(cumulative_array[-1])
    total_delta = float(delta_array.sum())
    if total_delta == 0:
        return max_length, max_length / 2.0

    midpoint_array = cumulative_array - segment_array / 2.0
    centroid_from_source = float(np.sum(midpoint_array * delta_array) / total_delta)
    centroid_from_outlet = max_length - centroid_from_source
    return max_length, centroid_from_outlet


def find_centroid_comid(
    centroid: float,
    max_length: float,
    comid_list: list[int],
    cumulative_lengths: list[float],
) -> int:
    """Find the reach interval that contains the centroid position."""
    if len(comid_list) == 0:
        return -1

    previous_cum_length = 0.0
    for comid, current_cum_length in zip(comid_list, cumulative_lengths):
        downstream_distance = max_length - current_cum_length
        upstream_distance = max_length - previous_cum_length
        if downstream_distance - 1e-9 <= centroid <= upstream_distance + 1e-9:
            return int(comid)
        previous_cum_length = current_cum_length

    distances_from_outlet = [max_length - cum for cum in cumulative_lengths]
    nearest_idx = int(np.argmin(np.abs(np.asarray(distances_from_outlet) - centroid)))
    return int(comid_list[nearest_idx])


def calculate_rci(centroid: float, max_length: float) -> float:
    """Compute the relative centroid index."""
    return float(centroid / max_length) if max_length > 0 else 0.0
