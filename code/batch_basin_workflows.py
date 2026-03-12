"""Unified command-line entry point for packaged global basin workflows."""

from __future__ import annotations

import argparse

from core.continuous_source import run_batch_basin_runoff_rci
from core.gridded_source import run_batch_basin_precipitation_level0
from core.multilevel import (
    run_batch_basin_multilevel_continuous,
    run_batch_basin_multilevel_gridded,
)
from core.pfaf import generate_batch_pfaf


def resolve_levels(explicit_levels, max_level: int):
    if explicit_levels is not None:
        return explicit_levels
    return list(range(1, max_level + 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unified workflows across packaged basin folders under data/basins/global."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_pfaf = subparsers.add_parser("pfaf", help="Generate Pfaf codes across packaged basins")
    parser_pfaf.add_argument("--global-dir", default="data/basins/global")
    parser_pfaf.add_argument("--output-dir", default="data/basins/global")
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
    parser_pfaf.add_argument("--basins", nargs="*", default=None)

    parser_runoff = subparsers.add_parser("runoff", help="Run basin-scale runoff RCI across multiple basins")
    parser_runoff.add_argument("--global-dir", default="data/basins/global")
    parser_runoff.add_argument("--output-dir", default="outputs/batch/runoff")
    parser_runoff.add_argument("--min-segments", type=int, default=3)
    parser_runoff.add_argument("--basins", nargs="*", default=None)

    parser_precip = subparsers.add_parser("precipitation", help="Run basin-scale gridded precipitation RCI across packaged basins")
    parser_precip.add_argument("--global-dir", default="data/basins/global")
    parser_precip.add_argument("--gridded-data", default="data/climate/climatology/mswep_precipitation_mean.nc")
    parser_precip.add_argument("--variable", default="precipitation")
    parser_precip.add_argument("--reduction", default="mean")
    parser_precip.add_argument("--output-dir", default="outputs/batch/precipitation_level0")
    parser_precip.add_argument("--min-segments", type=int, default=3)
    parser_precip.add_argument("--basins", nargs="*", default=None)

    parser_multi = subparsers.add_parser("multilevel", help="Run multilevel RCI across multiple basins")
    parser_multi.add_argument("--source", choices=["continuous", "gridded"], default="continuous")
    parser_multi.add_argument("--global-dir", default="data/basins/global")
    parser_multi.add_argument("--gridded-data", default="data/climate/climatology/mswep_precipitation_mean.nc")
    parser_multi.add_argument("--variable", default="precipitation")
    parser_multi.add_argument("--reduction", default="mean")
    parser_multi.add_argument("--output-dir", default=None)
    parser_multi.add_argument("--levels", nargs="+", type=int, default=None, help="Explicit Pfaf levels to calculate")
    parser_multi.add_argument("--max-level", type=int, default=5, help="Calculate levels 1..N when --levels is not provided")
    parser_multi.add_argument("--min-segments", type=int, default=1)
    parser_multi.add_argument("--min-mainstem-segments", type=int, default=4)
    parser_multi.add_argument("--filter-mode", default="independent_basins")
    parser_multi.add_argument("--basins", nargs="*", default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pfaf":
        generate_batch_pfaf(
            global_dir=args.global_dir,
            output_dir=args.output_dir,
            max_level=args.max_level,
            min_unit_reaches=args.min_unit_reaches,
            basin_names=args.basins,
        )
        return

    if args.command == "runoff":
        run_batch_basin_runoff_rci(
            global_dir=args.global_dir,
            output_dir=args.output_dir,
            min_segments=args.min_segments,
            basin_names=args.basins,
        )
        return

    if args.command == "precipitation":
        run_batch_basin_precipitation_level0(
            global_dir=args.global_dir,
            gridded_data_path=args.gridded_data,
            output_dir=args.output_dir,
            variable=args.variable,
            reduction=args.reduction,
            min_segments=args.min_segments,
            basin_names=args.basins,
        )
        return

    if args.command == "multilevel":
        levels = resolve_levels(args.levels, args.max_level)
        if args.source == "continuous":
            output_dir = args.output_dir or "outputs/batch/multilevel_continuous"
            run_batch_basin_multilevel_continuous(
                global_dir=args.global_dir,
                output_dir=output_dir,
                levels=levels,
                min_segments=args.min_segments,
                filter_mode=args.filter_mode,
                min_mainstem_segments=args.min_mainstem_segments,
                basin_names=args.basins,
            )
        else:
            output_dir = args.output_dir or "outputs/batch/multilevel_gridded"
            run_batch_basin_multilevel_gridded(
                global_dir=args.global_dir,
                gridded_data_path=args.gridded_data,
                output_dir=output_dir,
                levels=levels,
                variable=args.variable,
                reduction=args.reduction,
                min_segments=args.min_segments,
                filter_mode=args.filter_mode,
                min_mainstem_segments=args.min_mainstem_segments,
                basin_names=args.basins,
            )


if __name__ == "__main__":
    main()
