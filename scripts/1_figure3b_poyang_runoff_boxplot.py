"""Generate Figure 3b boxplot inputs and plot for Poyang runoff RCI."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.basin_io import resolve_single_basin_paths
from core.multilevel import run_single_basin_multilevel_continuous, summarize_multilevel_results
from core.pfaf import generate_single_basin_pfaf


LEVELS = [1, 2, 3, 4]
RESULT_DIR = Path("results/figure3b")
BOXPLOT_DATA_CSV = RESULT_DIR / "poyang_runoff_rci_boxplot_data_level1_4.csv"
SUMMARY_CSV = RESULT_DIR / "poyang_runoff_rci_boxplot_summary_level1_4.csv"
BOXPLOT_PNG = RESULT_DIR / "poyang_runoff_rci_boxplot_level1_4.png"


def build_boxplot() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    basin = resolve_single_basin_paths("data/basins/poyang", basin_name="poyang")

    # Regenerate Pfaf codes before the multilevel experiment so the workflow is self-contained.
    generate_single_basin_pfaf(
        basin=basin,
        max_level=max(LEVELS),
        min_unit_reaches=3,
    )

    with tempfile.NamedTemporaryFile(
        prefix="poyang_multilevel_runoff_rci_level1_4_",
        suffix=".csv",
        delete=False,
    ) as handle:
        temp_output_path = Path(handle.name)

    try:
        result_df = run_single_basin_multilevel_continuous(
            basin=basin,
            output_path=temp_output_path,
            levels=LEVELS,
            min_segments=1,
            filter_mode="independent_basins",
            min_mainstem_segments=4,
        )
    finally:
        if temp_output_path.exists():
            os.unlink(temp_output_path)

    boxplot_df = result_df[
        [
            "basin_name",
            "level",
            "subbasin_code",
            "rci",
            "num_segments",
            "num_subbasin_segments",
            "centroid_distance_km",
            "mainstem_length_km",
            "total_discharge",
        ]
    ].copy()
    boxplot_df.to_csv(BOXPLOT_DATA_CSV, index=False)

    summary_df = summarize_multilevel_results(result_df, "total_discharge").copy()
    summary_df.to_csv(SUMMARY_CSV, index=False)

    level_to_values = [
        boxplot_df.loc[boxplot_df["level"] == level, "rci"].astype(float).to_numpy()
        for level in LEVELS
    ]
    level_to_summary = {
        int(row.level): row
        for row in summary_df.itertuples(index=False)
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
    bp = ax.boxplot(
        level_to_values,
        tick_labels=[f"Level {level}\n(n={len(values)})" for level, values in zip(LEVELS, level_to_values)],
        patch_artist=True,
        widths=0.56,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#c0392b",
            "markeredgecolor": "#7b241c",
            "markersize": 5,
        },
        medianprops={"color": "#1f1f1f", "linewidth": 1.6},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#7f8c8d",
            "markeredgecolor": "#7f8c8d",
            "markersize": 3,
            "alpha": 0.55,
        },
    )

    colors = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#355c7d")
        patch.set_alpha(0.72)
        patch.set_linewidth(1.0)

    all_values = np.concatenate(level_to_values) if level_to_values else np.array([0.0])
    y_min = float(np.nanmin(all_values))
    y_max = float(np.nanmax(all_values))
    y_pad = max(0.03, (y_max - y_min) * 0.18)
    ax.set_ylim(y_min - 0.02, y_max + y_pad)

    for idx, level in enumerate(LEVELS, start=1):
        values = level_to_values[idx - 1]
        summary_row = level_to_summary[level]
        top_value = float(np.nanmax(values))
        ax.text(
            idx,
            top_value + y_pad * 0.18,
            f"mean={summary_row.mean_rci:.3f}\nmedian={summary_row.median_rci:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )

    ax.set_title("Poyang Runoff RCI by Pfaf Level", fontsize=12)
    ax.set_xlabel("Pfaf Level", fontsize=10)
    ax.set_ylabel("RCI", fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(BOXPLOT_PNG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_boxplot()
