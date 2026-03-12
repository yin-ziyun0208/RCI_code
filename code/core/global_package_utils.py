"""Helpers for packaged global basin folders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

def _normalize_requested_names(basin_names: Optional[Iterable[str]]) -> Optional[set[str]]:
    if basin_names is None:
        return None
    return {name.strip() for name in basin_names}


def iter_packaged_global_basins(global_dir: str | Path, basin_names: Optional[Iterable[str]] = None):
    requested = _normalize_requested_names(basin_names)
    global_dir_path = Path(global_dir)

    if not global_dir_path.exists():
        return

    for basin_dir in sorted(global_dir_path.iterdir()):
        if not basin_dir.is_dir():
            continue

        basin_name = basin_dir.name
        if requested is not None and basin_name not in requested:
            continue

        river_shp = basin_dir / "river_network.shp"
        if not river_shp.exists():
            continue

        yield {
            "basin_name": basin_name,
            "folder": basin_dir,
            "river_shp": river_shp,
            "catchment_shp": basin_dir / "catchments.shp",
            "discharge_csv": basin_dir / "grades_discharge.csv",
            "pfaf_csv": basin_dir / "pfaf_codes.csv",
            "pfaf_report_json": basin_dir / "pfaf_report.json",
            "precip_csv": basin_dir / "mswep_precipitation_upstream_totals.csv",
            "metadata_json": basin_dir / "metadata.json",
        }


def has_packaged_global_basins(global_dir: str | Path) -> bool:
    return any(iter_packaged_global_basins(global_dir))
