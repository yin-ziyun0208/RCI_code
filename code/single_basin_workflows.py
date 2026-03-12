"""
Unified command-line entry point for single-basin workflows.
"""

from __future__ import annotations

import argparse

from core.basin_io import resolve_single_basin_paths
from core.centroid import default_single_output_dir
from core.continuous_source import run_single_basin_runoff_rci
from core.gridded_source import run_single_basin_precipitation_rci
from core.multilevel import (
    run_single_basin_multilevel_continuous,
    run_single_basin_multilevel_gridded,
)
from core.pfaf import generate_single_basin_pfaf


def resolve_levels(explicit_levels, max_level: int):
    if explicit_levels is not None:
        return explicit_levels
    return list(range(1, max_level + 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unified workflows for one basin directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_pfaf = subparsers.add_parser("pfaf", help="Generate Pfafstetter codes for one basin")
    parser_pfaf.add_argument("--basin-dir", default="data/basins/poyang", help="Single-basin data directory")
    parser_pfaf.add_argument("--basin-name", default=None, help="Optional explicit basin name")
    parser_pfaf.add_argument(
        "--max-level",
        type=int,
        default=None,
        help=(
            "Optional maximum Pfaf depth. Default is unlimited until a unit reaches "
            "max_level, falls below min_unit_reaches, or has no tributary left to subdivide."
        ),
    )
    parser_pfaf.add_argument("--min-unit-reaches", type=int, default=3)

    parser_runoff = subparsers.add_parser("runoff", help="Run basin-scale runoff RCI")
    parser_runoff.add_argument("--basin-dir", default="data/basins/poyang", help="Single-basin data directory")
    parser_runoff.add_argument("--basin-name", default=None, help="Optional explicit basin name")
    parser_runoff.add_argument("--output", default=None, help="Optional output CSV path")
    parser_runoff.add_argument("--min-segments", type=int, default=3)

    parser_precip = subparsers.add_parser("precipitation", help="Run basin-scale gridded precipitation RCI")
    parser_precip.add_argument("--basin-dir", default="data/basins/poyang", help="Single-basin data directory")
    parser_precip.add_argument("--basin-name", default=None, help="Optional explicit basin name")
    parser_precip.add_argument("--gridded-data", default="data/climate/climatology/mswep_precipitation_mean.nc")
    parser_precip.add_argument("--variable", default="precipitation")
    parser_precip.add_argument("--reduction", default="mean")
    parser_precip.add_argument("--output", default=None, help="Optional output CSV path")
    parser_precip.add_argument("--min-segments", type=int, default=3)

    parser_multi = subparsers.add_parser("multilevel", help="Run multilevel RCI for one basin")
    parser_multi.add_argument("--basin-dir", default="data/basins/poyang", help="Single-basin data directory")
    parser_multi.add_argument("--basin-name", default=None, help="Optional explicit basin name")
    parser_multi.add_argument("--source", choices=["continuous", "gridded"], default="continuous")
    parser_multi.add_argument("--levels", nargs="+", type=int, default=None, help="Explicit Pfaf levels to calculate")
    parser_multi.add_argument("--max-level", type=int, default=5, help="Calculate levels 1..N when --levels is not provided")
    parser_multi.add_argument("--output", default=None, help="Optional output CSV path")
    parser_multi.add_argument("--min-segments", type=int, default=1)
    parser_multi.add_argument("--min-mainstem-segments", type=int, default=4)
    parser_multi.add_argument("--filter-mode", default="independent_basins")
    parser_multi.add_argument("--gridded-data", default="data/climate/climatology/mswep_precipitation_mean.nc")
    parser_multi.add_argument("--variable", default="precipitation")
    parser_multi.add_argument("--reduction", default="mean")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    basin = resolve_single_basin_paths(args.basin_dir, basin_name=getattr(args, "basin_name", None))

    if args.command == "pfaf":
        generate_single_basin_pfaf(
            basin=basin,
            max_level=args.max_level,
            min_unit_reaches=args.min_unit_reaches,
        )
        return

    if args.command == "runoff":
        output = args.output or default_single_output_dir("runoff", basin.name) / "centroid_results.csv"
        run_single_basin_runoff_rci(
            basin=basin,
            output_path=output,
            min_segments=args.min_segments,
        )
        return

    if args.command == "precipitation":
        output = args.output or default_single_output_dir("precipitation", basin.name) / "centroid_results_gridded.csv"
        run_single_basin_precipitation_rci(
            basin=basin,
            gridded_data_path=args.gridded_data,
            output_path=output,
            variable=args.variable,
            reduction=args.reduction,
            min_segments=args.min_segments,
        )
        return

    if args.command == "multilevel":
        levels = resolve_levels(args.levels, args.max_level)
        workflow_name = f"multilevel_{args.source}"
        output = args.output or default_single_output_dir(workflow_name, basin.name) / "centroid_results.csv"
        if args.source == "continuous":
            run_single_basin_multilevel_continuous(
                basin=basin,
                output_path=output,
                levels=levels,
                min_segments=args.min_segments,
                filter_mode=args.filter_mode,
                min_mainstem_segments=args.min_mainstem_segments,
            )
        else:
            run_single_basin_multilevel_gridded(
                basin=basin,
                gridded_data_path=args.gridded_data,
                output_path=output,
                levels=levels,
                variable=args.variable,
                reduction=args.reduction,
                min_segments=args.min_segments,
                filter_mode=args.filter_mode,
                min_mainstem_segments=args.min_mainstem_segments,
            )


if __name__ == "__main__":
    main()
