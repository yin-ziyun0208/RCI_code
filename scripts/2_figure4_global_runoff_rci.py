"""Generate the global 40-basin runoff RCI table for Figure 4."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.continuous_source import run_batch_basin_runoff_rci


RESULT_DIR = Path("results/figure4")
RESULT_CSV = RESULT_DIR / "figure4_global_runoff_rci.csv"
FAILURE_CSV = RESULT_DIR / "figure4_global_runoff_rci_failures.csv"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    result_df, failures_df = run_batch_basin_runoff_rci(
        global_dir="data/basins/global",
        output_dir=RESULT_DIR,
        min_segments=3,
        basin_names=None,
    )

    figure_df = result_df[
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
    figure_df = figure_df.sort_values("basin_name").reset_index(drop=True)
    figure_df.to_csv(RESULT_CSV, index=False)
    if failures_df.empty:
        failures_df = failures_df.reindex(columns=["basin_name", "river_file", "error"])
    failures_df.to_csv(FAILURE_CSV, index=False)


if __name__ == "__main__":
    main()
