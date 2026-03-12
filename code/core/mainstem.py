"""Shared river-network topology helpers used by all active workflows."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_UP_COLS = ["up1", "up2", "up3", "up4"]


def build_river_maps(
    river_df: pd.DataFrame,
    comid_col: str = "COMID",
    uparea_col: str = "uparea",
    up_cols: Optional[List[str]] = None,
) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    """Build lookup maps for upstream area and upstream connectivity."""
    if up_cols is None:
        up_cols = DEFAULT_UP_COLS

    comids = set(pd.to_numeric(river_df[comid_col], errors="coerce").dropna().astype(int))
    uparea_map = dict(
        zip(
            pd.to_numeric(river_df[comid_col], errors="coerce").astype(int),
            pd.to_numeric(river_df[uparea_col], errors="coerce").fillna(0.0),
        )
    )
    upstream_map: Dict[int, List[int]] = {}

    topology_cols = [comid_col, *up_cols]
    for row in river_df[topology_cols].itertuples(index=False, name=None):
        comid = int(row[0])
        upstreams = []
        for upstream_id in row[1:]:
            upstream_id = int(upstream_id)
            if upstream_id > 0 and upstream_id in comids:
                upstreams.append(upstream_id)
        upstream_map[comid] = upstreams

    return uparea_map, upstream_map


def trace_main_stem_with_maps(
    outlet_comid: int,
    uparea_map: Dict[int, float],
    upstream_map: Dict[int, List[int]],
    valid_comids: Optional[set[int]] = None,
) -> List[int]:
    """Trace the main stem from outlet to source, following the largest-uparea branch."""
    if outlet_comid not in upstream_map:
        return []

    main_stem_down_to_up = [outlet_comid]
    current = outlet_comid

    while True:
        candidates = upstream_map.get(current, [])
        if valid_comids is not None:
            candidates = [comid for comid in candidates if comid in valid_comids]

        if not candidates:
            break

        current = max(candidates, key=lambda comid: uparea_map.get(comid, -1.0))
        main_stem_down_to_up.append(current)

    return main_stem_down_to_up[::-1]


def trace_main_stem_from_outlet(
    river_df: pd.DataFrame,
    outlet_comid: int,
    comid_col: str = "COMID",
    uparea_col: str = "uparea",
    up_cols: Optional[List[str]] = None,
    uparea_map: Optional[Dict[int, float]] = None,
    upstream_map: Optional[Dict[int, List[int]]] = None,
    valid_comids: Optional[set[int]] = None,
) -> List[int]:
    """Trace the main stem using either cached or freshly built topology maps."""
    if uparea_map is None or upstream_map is None:
        uparea_map, upstream_map = build_river_maps(
            river_df,
            comid_col=comid_col,
            uparea_col=uparea_col,
            up_cols=up_cols,
        )
    return trace_main_stem_with_maps(
        outlet_comid=outlet_comid,
        uparea_map=uparea_map,
        upstream_map=upstream_map,
        valid_comids=valid_comids,
    )


def find_basin_outlet(
    river_df: pd.DataFrame,
    comid_col: str = "COMID",
    uparea_col: str = "uparea",
) -> int:
    """Select the basin outlet as the reach with the largest upstream area."""
    outlet_idx = river_df[uparea_col].idxmax()
    return int(river_df.loc[outlet_idx, comid_col])
