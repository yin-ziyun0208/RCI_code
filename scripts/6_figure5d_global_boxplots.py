"""Build Figure 5d boxplot data and plot from existing Figure 5 summary tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = Path("results/figure5")

PQ_TABLE_CSV = RESULT_DIR / "figure5_global_pq_centroid_distance.csv"
LAKE_TABLE_CSV = RESULT_DIR / "figure5_global_lake_volume_between_pq.csv"
SLOPE_TABLE_CSV = RESULT_DIR / "figure5_global_slope_between_pq.csv"

BOXPLOT_DATA_CSV = RESULT_DIR / "figure5d_global_boxplot_data.csv"
SUMMARY_CSV = RESULT_DIR / "figure5d_global_boxplot_summary.csv"
FIGURE_PNG = RESULT_DIR / "figure5d_global_boxplot.png"

GROUP_ORDER = ["short", "long"]
GROUP_COLORS = {"short": "#4c78a8", "long": "#f58518"}
METRIC_ORDER = [
    ("centroid_segment_slope_m_per_km", "Slope", "Slope (m/km)", False),
    (
        "lake_volume_to_discharge_ratio",
        "Lake Volume (relative)",
        "Lake Volume (relative)",
        True,
    ),
]


def build_figure5d() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create Figure 5d long-format table, summary table, and boxplot."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    pq_df = pd.read_csv(PQ_TABLE_CSV)
    lake_df = pd.read_csv(LAKE_TABLE_CSV)
    slope_df = pd.read_csv(SLOPE_TABLE_CSV)

    merged = (
        pq_df.loc[:, ["basin_name", "pq_distance_group", "abs_pq_distance_rci"]]
        .merge(
            slope_df.loc[:, ["basin_name", "centroid_segment_slope_m_per_km"]],
            on="basin_name",
            how="inner",
        )
        .merge(
            lake_df.loc[:, ["basin_name", "lake_volume_to_discharge_ratio"]],
            on="basin_name",
            how="inner",
        )
        .sort_values("basin_name")
        .reset_index(drop=True)
    )

    boxplot_df = merged.melt(
        id_vars=["basin_name", "pq_distance_group", "abs_pq_distance_rci"],
        value_vars=[item[0] for item in METRIC_ORDER],
        var_name="metric_key",
        value_name="metric_value",
    )
    metric_labels = {item[0]: item[1] for item in METRIC_ORDER}
    y_labels = {item[0]: item[2] for item in METRIC_ORDER}
    log_flags = {item[0]: item[3] for item in METRIC_ORDER}
    boxplot_df["metric_label"] = boxplot_df["metric_key"].map(metric_labels)
    boxplot_df["y_label"] = boxplot_df["metric_key"].map(y_labels)
    boxplot_df["use_log_scale"] = boxplot_df["metric_key"].map(log_flags)
    boxplot_df["pq_distance_group"] = pd.Categorical(
        boxplot_df["pq_distance_group"],
        categories=GROUP_ORDER,
        ordered=True,
    )
    boxplot_df = boxplot_df.sort_values(
        ["metric_key", "pq_distance_group", "basin_name"]
    ).reset_index(drop=True)

    summary_df = (
        boxplot_df.groupby(["metric_key", "metric_label", "pq_distance_group"], as_index=False)
        .agg(
            n=("metric_value", "size"),
            mean_value=("metric_value", "mean"),
            median_value=("metric_value", "median"),
            min_value=("metric_value", "min"),
            max_value=("metric_value", "max"),
        )
        .sort_values(["metric_key", "pq_distance_group"])
        .reset_index(drop=True)
    )

    boxplot_df.to_csv(BOXPLOT_DATA_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    _draw_figure(boxplot_df, summary_df)
    return boxplot_df, summary_df


def _draw_figure(boxplot_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Draw the Figure 5d two-panel boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    for ax, (metric_key, metric_label, y_label, use_log_scale) in zip(axes, METRIC_ORDER):
        subset = boxplot_df[boxplot_df["metric_key"] == metric_key].copy()
        values_by_group = [
            subset.loc[subset["pq_distance_group"] == group, "metric_value"].to_numpy(dtype=float)
            for group in GROUP_ORDER
        ]
        positions = np.arange(1, len(GROUP_ORDER) + 1)

        bp = ax.boxplot(
            values_by_group,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            whis=1.5,
            showfliers=True,
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#444444", "linewidth": 1.1},
            capprops={"color": "#444444", "linewidth": 1.1},
            boxprops={"edgecolor": "#444444", "linewidth": 1.1},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#666666",
                "markeredgecolor": "#666666",
                "markersize": 3.5,
                "alpha": 0.55,
            },
        )
        for patch, group in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_alpha(0.68)

        summary_subset = summary_df[summary_df["metric_key"] == metric_key].copy()
        xticklabels = []
        for idx, group in enumerate(GROUP_ORDER, start=1):
            row = summary_subset[summary_subset["pq_distance_group"] == group].iloc[0]
            xticklabels.append(f"{group}\n(n={int(row['n'])})")

        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("P-Q centroid distance group")
        ax.set_ylabel(y_label)
        ax.set_title(metric_label)
        ax.grid(axis="y", color="#dddddd", linestyle="--", linewidth=0.7, alpha=0.8)

        if use_log_scale:
            positive_values = subset.loc[subset["metric_value"] > 0, "metric_value"]
            if not positive_values.empty:
                ax.set_yscale("log")
                ymin = positive_values.min() * 0.8
                ymax = subset["metric_value"].max() * 1.4
                ax.set_ylim(ymin, ymax)
        else:
            ymax = subset["metric_value"].max() * 1.18 if subset["metric_value"].max() > 0 else 1.0
            ax.set_ylim(0, ymax)

        for idx, group in enumerate(GROUP_ORDER, start=1):
            row = summary_subset[summary_subset["pq_distance_group"] == group].iloc[0]
            if use_log_scale:
                y_text = row["max_value"] * 1.12 if row["max_value"] > 0 else 1.0
                text = f"median={row['median_value']:.3g}\nmean={row['mean_value']:.3g}"
            else:
                y_text = row["max_value"] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                text = f"median={row['median_value']:.3f}\nmean={row['mean_value']:.3f}"
            ax.text(idx, y_text, text, ha="center", va="bottom", fontsize=8.5, color="#222222")

    fig.suptitle("Figure 5d: Slope and relative lake volume by P-Q centroid distance group")
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_figure5d()
