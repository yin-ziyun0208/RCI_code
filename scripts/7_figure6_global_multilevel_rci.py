"""Build Figure 6 data tables and boxplot for global level 0-4 precipitation and runoff RCI."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("results/figure6")

RUNOFF_LEVEL0_CSV = Path("outputs/batch/runoff/global_centroid_results.csv")
PRECIP_LEVEL0_CSV = Path("outputs/batch/precipitation_level0/global_precipitation_level0_results.csv")
RUNOFF_MULTI_CSV = Path("outputs/batch/multilevel_continuous/global_multilevel_rci_results.csv")
PRECIP_MULTI_CSV = Path("outputs/batch/multilevel_gridded/global_multilevel_gridded_rci_results.csv")

RUNOFF_TABLE_CSV = RESULT_DIR / "figure6_global_runoff_rci_level0_4.csv"
PRECIP_TABLE_CSV = RESULT_DIR / "figure6_global_precipitation_rci_level0_4.csv"
BOXPLOT_DATA_CSV = RESULT_DIR / "figure6_global_rci_boxplot_data_level0_4.csv"
SUMMARY_CSV = RESULT_DIR / "figure6_global_rci_boxplot_summary_level0_4.csv"
FIGURE_PNG = RESULT_DIR / "figure6_global_rci_boxplot_level0_4.png"


def _prepare_level0_table(
    source_df: pd.DataFrame,
    variable: str,
    total_column: str,
) -> pd.DataFrame:
    table = source_df.copy()
    table["level"] = 0
    table["variable"] = variable
    keep_cols = [
        "basin_name",
        "level",
        "variable",
        "outlet_COMID",
        "centroid_COMID",
        "centroid_distance_km",
        "mainstem_length_km",
        "rci",
        "num_segments",
        total_column,
    ]
    return table.loc[:, keep_cols].copy()


def _prepare_multilevel_table(
    source_df: pd.DataFrame,
    variable: str,
    total_column: str,
) -> pd.DataFrame:
    table = source_df[source_df["level"].between(1, 4)].copy()
    table["variable"] = variable
    keep_cols = [
        "basin_name",
        "level",
        "variable",
        "subbasin_code",
        "outlet_COMID",
        "centroid_COMID",
        "centroid_distance_km",
        "mainstem_length_km",
        "rci",
        "num_segments",
        "num_subbasin_segments",
        total_column,
    ]
    return table.loc[:, keep_cols].copy()


def _build_summary(boxplot_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        boxplot_df.groupby(["variable", "level"], as_index=False)
        .agg(
            n=("rci", "size"),
            mean_rci=("rci", "mean"),
            median_rci=("rci", "median"),
            min_rci=("rci", "min"),
            max_rci=("rci", "max"),
        )
        .sort_values(["variable", "level"])
        .reset_index(drop=True)
    )
    return summary


def _draw_boxplot(boxplot_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    variable_order = ["precipitation", "runoff"]
    titles = {
        "precipitation": "Precipitation RCI",
        "runoff": "Runoff RCI",
    }

    for ax, variable in zip(axes, variable_order):
        subset = boxplot_df[boxplot_df["variable"] == variable].copy()
        level_data = [
            subset.loc[subset["level"] == level, "rci"].dropna().tolist()
            for level in range(0, 5)
        ]
        ax.boxplot(
            level_data,
            positions=list(range(1, 6)),
            widths=0.6,
            patch_artist=True,
            boxprops={"facecolor": "#cfd8dc", "edgecolor": "#37474f"},
            medianprops={"color": "#d32f2f", "linewidth": 1.8},
            whiskerprops={"color": "#546e7a"},
            capprops={"color": "#546e7a"},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "#78909c",
                "markeredgecolor": "#78909c",
                "alpha": 0.5,
            },
        )
        ax.set_title(titles[variable])
        ax.set_xlabel("Pfaf Level")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f"{level}" for level in range(0, 5)])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

        variable_summary = summary_df[summary_df["variable"] == variable]
        for _, row in variable_summary.iterrows():
            xpos = int(row["level"]) + 1
            ax.text(
                xpos,
                0.98,
                f"n={int(row['n'])}",
                ha="center",
                va="top",
                fontsize=8,
                color="#263238",
            )
            ax.text(
                xpos,
                0.06,
                f"mean={row['mean_rci']:.3f}\nmedian={row['median_rci']:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#263238",
            )

    axes[0].set_ylabel("RCI")
    fig.suptitle("Global Precipitation and Runoff RCI Across Levels 0-4", y=0.98)
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_figure6_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    runoff_level0 = pd.read_csv(RUNOFF_LEVEL0_CSV)
    precip_level0 = pd.read_csv(PRECIP_LEVEL0_CSV)
    runoff_multi = pd.read_csv(RUNOFF_MULTI_CSV)
    precip_multi = pd.read_csv(PRECIP_MULTI_CSV)

    runoff_table = pd.concat(
        [
            _prepare_level0_table(runoff_level0, "runoff", "total_discharge"),
            _prepare_multilevel_table(runoff_multi, "runoff", "total_discharge"),
        ],
        ignore_index=True,
    ).sort_values(["level", "basin_name", "centroid_COMID"]).reset_index(drop=True)

    precip_table = pd.concat(
        [
            _prepare_level0_table(precip_level0, "precipitation", "total_mass"),
            _prepare_multilevel_table(precip_multi, "precipitation", "total_mass"),
        ],
        ignore_index=True,
    ).sort_values(["level", "basin_name", "centroid_COMID"]).reset_index(drop=True)

    boxplot_df = pd.concat(
        [
            runoff_table.loc[:, ["basin_name", "level", "variable", "rci"]],
            precip_table.loc[:, ["basin_name", "level", "variable", "rci"]],
        ],
        ignore_index=True,
    ).sort_values(["variable", "level", "basin_name"]).reset_index(drop=True)

    summary_df = _build_summary(boxplot_df)

    runoff_table.to_csv(RUNOFF_TABLE_CSV, index=False)
    precip_table.to_csv(PRECIP_TABLE_CSV, index=False)
    boxplot_df.to_csv(BOXPLOT_DATA_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    _draw_boxplot(boxplot_df, summary_df)

    return runoff_table, precip_table, boxplot_df, summary_df


if __name__ == "__main__":
    build_figure6_outputs()
