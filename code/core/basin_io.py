"""Path resolution helpers for single-basin and packaged global basin data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class BasinPaths:
    """Resolved input files for one basin."""

    name: str
    basin_dir: Path
    river_network: Path
    catchments: Optional[Path]
    discharge_csv: Optional[Path]
    precipitation_upstream_csv: Optional[Path]
    precipitation_upstream_meta_json: Optional[Path]
    pfaf_csv: Optional[Path]
    pfaf_report_json: Optional[Path]
    mode: str


def _pick_single(directory: Path, patterns: list[str], label: str, required: bool) -> Optional[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(directory.glob(pattern)))
    matches = sorted(set(matches))

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        if required:
            raise FileNotFoundError(f"Unable to find {label} in {directory}")
        return None
    raise ValueError(f"Expected exactly one {label} in {directory}, found {len(matches)}")


def resolve_single_basin_paths(basin_dir: str | Path, basin_name: Optional[str] = None) -> BasinPaths:
    """
    Resolve the standard file layout for a single basin directory.

    This supports both:
    - the local example layout under ``data/basins/poyang/``, and
    - the packaged global layout under ``data/basins/global/<BASIN_NAME>/``.
    """
    basin_dir_path = Path(basin_dir)
    if not basin_dir_path.is_dir():
        raise FileNotFoundError(f"Basin directory not found: {basin_dir_path}")

    river_network = _pick_single(
        basin_dir_path,
        ["river_network.shp"],
        label="river-network shapefile",
        required=True,
    )
    catchments = _pick_single(
        basin_dir_path,
        ["catchments.shp"],
        label="catchment shapefile",
        required=False,
    )
    discharge_csv = _pick_single(
        basin_dir_path,
        ["grades_discharge.csv"],
        label="discharge CSV",
        required=False,
    )
    precipitation_upstream_csv = _pick_single(
        basin_dir_path,
        ["mswep_precipitation_upstream_totals.csv"],
        label="precipitation upstream-total CSV",
        required=False,
    )
    precipitation_upstream_meta_json = _pick_single(
        basin_dir_path,
        ["mswep_precipitation_upstream_totals.meta.json"],
        label="precipitation upstream-total metadata JSON",
        required=False,
    )
    pfaf_csv = _pick_single(
        basin_dir_path,
        ["pfaf_codes.csv"],
        label="Pfaf CSV",
        required=False,
    )
    pfaf_report_json = _pick_single(
        basin_dir_path,
        ["pfaf_report.json"],
        label="Pfaf report JSON",
        required=False,
    )
    inferred_name = basin_name or basin_dir_path.name
    mode = "local" if basin_dir_path.name == "poyang" else "packaged"

    return BasinPaths(
        name=inferred_name,
        basin_dir=basin_dir_path,
        river_network=river_network,
        catchments=catchments,
        discharge_csv=discharge_csv,
        precipitation_upstream_csv=precipitation_upstream_csv,
        precipitation_upstream_meta_json=precipitation_upstream_meta_json,
        pfaf_csv=pfaf_csv,
        pfaf_report_json=pfaf_report_json,
        mode=mode,
    )


def iter_global_basin_paths(global_dir: str | Path, basin_names: Optional[Iterable[str]] = None):
    """Yield packaged global basin directories under ``data/basins/global``."""
    global_dir_path = Path(global_dir)
    if not global_dir_path.is_dir():
        raise FileNotFoundError(f"Global basin directory not found: {global_dir_path}")

    requested = None if basin_names is None else {name.strip() for name in basin_names}
    for basin_dir in sorted(global_dir_path.iterdir()):
        if not basin_dir.is_dir():
            continue
        if requested is not None and basin_dir.name not in requested:
            continue
        if not (basin_dir / "river_network.shp").exists():
            continue
        yield resolve_single_basin_paths(basin_dir, basin_name=basin_dir.name)
