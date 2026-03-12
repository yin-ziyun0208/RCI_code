"""Build Figure 8 trend tables and plots from annual RCI time series."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = Path("results/figure8")

Q_NAT_CSV = RESULT_DIR / "figure8_global_qnat_annual_rci.csv"
Q_HUM_CSV = RESULT_DIR / "figure8_global_qhum_annual_rci.csv"
P_CSV = RESULT_DIR / "figure8_global_precipitation_annual_rci.csv"

TREND_TABLE_CSV = RESULT_DIR / "figure8_global_trend_table.csv"
SCATTER_CSV = RESULT_DIR / "figure8_global_trend_scatter_data.csv"
BOXPLOT_CSV = RESULT_DIR / "figure8_global_trend_boxplot_data.csv"
GROUP_SUMMARY_CSV = RESULT_DIR / "figure8_global_trend_group_summary.csv"
FIGURE_PNG = RESULT_DIR / "figure8_global_trend_analysis.png"

TREND_THRESHOLD_PCT_PER_YEAR = 0.02

SOURCE_LABELS = {
    "p": "P",
    "q_nat": "Q_nat",
    "q_hum": "Q_hum",
}

GROUP_LABELS = {
    1: "Group 1",
    2: "Group 2",
    3: "Group 3",
    4: "Group 4",
}


def sen_slope_pct_per_year(years: np.ndarray, values: np.ndarray) -> float:
    """Return Sen's slope in %/year from an annual RCI series."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float) * 100.0
    if len(years) < 2:
        raise ValueError("At least two annual points are required for Sen's slope")

    slopes = []
    for start in range(len(years) - 1):
        year_diffs = years[start + 1 :] - years[start]
        value_diffs = values[start + 1 :] - values[start]
        slopes.extend((value_diffs / year_diffs).tolist())
    return float(np.median(np.asarray(slopes, dtype=float)))


def _load_source_table(path: Path, source_key: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    keep_cols = ["basin_name", "year", "rci"]
    table = table.loc[:, keep_cols].copy()
    table["source_key"] = source_key
    table["source_label"] = SOURCE_LABELS[source_key]
    return table


def _classify_group(
    p_trend: float,
    qnat_trend: float,
    qhum_trend: float,
    threshold: float,
) -> tuple[int, str]:
    """Classify Figure 8 groups using the fuller paper logic rather than the caption shorthand.

    Group 1:
        P and Q_nat show trends in the same direction and both exceed the
        threshold, while Q_hum shows a more obvious trend than both.
    Group 2:
        P and Q_nat are both near-zero, but Q_hum shows an apparent trend.
    Group 4:
        Special cases where Q_hum is closer to P than Q_nat while P and Q_nat
        still migrate in the same direction. These are retained in the boxplot
        for completeness even though the figure logic treats them separately.
    Group 3:
        Remaining cases with inconsistent or mixed trend behavior.
    """
    p_small = abs(p_trend) < threshold
    qnat_small = abs(qnat_trend) < threshold
    qhum_small = abs(qhum_trend) < threshold
    same_direction = (p_trend * qnat_trend) > 0
    both_significant_same_direction = same_direction and (abs(p_trend) >= threshold) and (
        abs(qnat_trend) >= threshold
    )
    qnat_distance = abs(qnat_trend - p_trend)
    qhum_distance = abs(qhum_trend - p_trend)
    qhum_more_obvious = abs(qhum_trend) > max(abs(p_trend), abs(qnat_trend))

    if both_significant_same_direction and qhum_more_obvious:
        return 1, GROUP_LABELS[1]
    if p_small and qnat_small and not qhum_small:
        return 2, GROUP_LABELS[2]
    if same_direction and qhum_distance < qnat_distance:
        return 4, GROUP_LABELS[4]
    return 3, GROUP_LABELS[3]


def build_figure8_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_df = pd.concat(
        [
            _load_source_table(Q_NAT_CSV, "q_nat"),
            _load_source_table(Q_HUM_CSV, "q_hum"),
            _load_source_table(P_CSV, "p"),
        ],
        ignore_index=True,
    ).sort_values(["source_key", "basin_name", "year"]).reset_index(drop=True)

    trend_rows: list[dict[str, object]] = []
    for basin_name, basin_df in all_df.groupby("basin_name", sort=True):
        trend_record = {"basin_name": basin_name}
        years = None
        for source_key in ["p", "q_nat", "q_hum"]:
            source_df = basin_df[basin_df["source_key"] == source_key].copy()
            source_df = source_df.sort_values("year").reset_index(drop=True)
            if years is None:
                years = source_df["year"].to_numpy(dtype=float)
                trend_record["start_year"] = int(source_df["year"].min())
                trend_record["end_year"] = int(source_df["year"].max())
                trend_record["n_years"] = int(source_df["year"].nunique())
            slope = sen_slope_pct_per_year(
                source_df["year"].to_numpy(dtype=float),
                source_df["rci"].to_numpy(dtype=float),
            )
            trend_record[f"{source_key}_sen_slope_pct_per_year"] = slope

        p_trend = float(trend_record["p_sen_slope_pct_per_year"])
        qnat_trend = float(trend_record["q_nat_sen_slope_pct_per_year"])
        qhum_trend = float(trend_record["q_hum_sen_slope_pct_per_year"])
        group_id, group_label = _classify_group(
            p_trend,
            qnat_trend,
            qhum_trend,
            threshold=TREND_THRESHOLD_PCT_PER_YEAR,
        )
        trend_record["group_id"] = group_id
        trend_record["group_label"] = group_label
        trend_record["qnat_minus_p_pct_per_year"] = qnat_trend - p_trend
        trend_record["qhum_minus_p_pct_per_year"] = qhum_trend - p_trend
        trend_record["abs_qnat_minus_p_pct_per_year"] = abs(qnat_trend - p_trend)
        trend_record["abs_qhum_minus_p_pct_per_year"] = abs(qhum_trend - p_trend)
        trend_rows.append(trend_record)

    trend_df = pd.DataFrame(trend_rows).sort_values("basin_name").reset_index(drop=True)

    scatter_df = pd.concat(
        [
            trend_df.loc[
                :,
                [
                    "basin_name",
                    "group_id",
                    "group_label",
                    "p_sen_slope_pct_per_year",
                    "q_nat_sen_slope_pct_per_year",
                ],
            ].rename(columns={"q_nat_sen_slope_pct_per_year": "y_sen_slope_pct_per_year"}),
            trend_df.loc[
                :,
                [
                    "basin_name",
                    "group_id",
                    "group_label",
                    "p_sen_slope_pct_per_year",
                    "q_hum_sen_slope_pct_per_year",
                ],
            ].rename(columns={"q_hum_sen_slope_pct_per_year": "y_sen_slope_pct_per_year"}),
        ],
        keys=["q_nat", "q_hum"],
        names=["source_key"],
    ).reset_index(level=0).reset_index(drop=True)
    scatter_df["source_label"] = scatter_df["source_key"].map(SOURCE_LABELS)

    boxplot_rows: list[dict[str, object]] = []
    for _, row in trend_df.iterrows():
        for source_key in ["p", "q_nat", "q_hum"]:
            boxplot_rows.append(
                {
                    "basin_name": row["basin_name"],
                    "group_id": row["group_id"],
                    "group_label": row["group_label"],
                    "source_key": source_key,
                    "source_label": SOURCE_LABELS[source_key],
                    "sen_slope_pct_per_year": row[f"{source_key}_sen_slope_pct_per_year"],
                }
            )
    boxplot_df = pd.DataFrame(boxplot_rows).sort_values(
        ["group_id", "source_key", "basin_name"]
    ).reset_index(drop=True)

    group_summary = (
        boxplot_df.groupby(["group_id", "group_label", "source_key", "source_label"], as_index=False)
        .agg(
            n=("sen_slope_pct_per_year", "size"),
            mean_slope_pct_per_year=("sen_slope_pct_per_year", "mean"),
            median_slope_pct_per_year=("sen_slope_pct_per_year", "median"),
        )
        .sort_values(["group_id", "source_key"])
        .reset_index(drop=True)
    )
    basin_group_counts = trend_df.groupby(["group_id", "group_label"], as_index=False).agg(
        n_basins=("basin_name", "size")
    )
    group_summary = group_summary.merge(
        basin_group_counts,
        on=["group_id", "group_label"],
        how="left",
    )

    trend_df.to_csv(TREND_TABLE_CSV, index=False)
    scatter_df.to_csv(SCATTER_CSV, index=False)
    boxplot_df.to_csv(BOXPLOT_CSV, index=False)
    group_summary.to_csv(GROUP_SUMMARY_CSV, index=False)

    _draw_figure8(trend_df, scatter_df, boxplot_df, basin_group_counts)
    return trend_df, scatter_df, boxplot_df, group_summary


def _draw_figure8(
    trend_df: pd.DataFrame,
    scatter_df: pd.DataFrame,
    boxplot_df: pd.DataFrame,
    basin_group_counts: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.28)

    ax_scatter = fig.add_subplot(gs[0, :])
    colors = {
        1: "#0d47a1",
        2: "#2e7d32",
        3: "#ef6c00",
        4: "#6a1b9a",
    }
    markers = {"q_nat": "o", "q_hum": "o"}
    fills = {"q_nat": True, "q_hum": False}

    x_values = scatter_df["p_sen_slope_pct_per_year"].to_numpy(dtype=float)
    y_values = scatter_df["y_sen_slope_pct_per_year"].to_numpy(dtype=float)
    bound = max(np.max(np.abs(x_values)), np.max(np.abs(y_values)), TREND_THRESHOLD_PCT_PER_YEAR) * 1.1

    for source_key in ["q_nat", "q_hum"]:
        source_df = scatter_df[scatter_df["source_key"] == source_key]
        for group_id in sorted(source_df["group_id"].unique()):
            group_df = source_df[source_df["group_id"] == group_id]
            facecolor = colors[group_id] if fills[source_key] else "none"
            ax_scatter.scatter(
                group_df["p_sen_slope_pct_per_year"],
                group_df["y_sen_slope_pct_per_year"],
                s=54,
                marker=markers[source_key],
                facecolors=facecolor,
                edgecolors=colors[group_id],
                linewidths=1.2,
                alpha=0.9,
                label=f"{SOURCE_LABELS[source_key]} - {GROUP_LABELS[group_id]}",
            )

    ax_scatter.axline((0, 0), slope=1.0, color="#212121", linestyle="--", linewidth=1.0)
    ax_scatter.axhline(0, color="#9e9e9e", linestyle=":", linewidth=0.9)
    ax_scatter.axvline(0, color="#9e9e9e", linestyle=":", linewidth=0.9)
    ax_scatter.set_xlim(-bound, bound)
    ax_scatter.set_ylim(-bound, bound)
    ax_scatter.set_xlabel("P trend (Sen's slope, %/year)")
    ax_scatter.set_ylabel("Runoff trend (Sen's slope, %/year)")
    ax_scatter.set_title("(a) Trend comparison across 40 global basins")
    ax_scatter.grid(alpha=0.2, linewidth=0.6)

    handles, labels = ax_scatter.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_scatter.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        fontsize=7,
        ncol=2,
        frameon=False,
    )

    positions = {
        1: gs[1, 0],
        2: gs[1, 1],
        3: gs[2, 0],
        4: gs[2, 1],
    }
    source_order = ["p", "q_nat", "q_hum"]
    source_titles = ["P", "Q_nat", "Q_hum"]
    box_colors = ["#90caf9", "#80cbc4", "#f48fb1"]

    for group_id in [1, 2, 3, 4]:
        ax = fig.add_subplot(positions[group_id])
        group_df = boxplot_df[boxplot_df["group_id"] == group_id]
        level_data = [
            group_df.loc[group_df["source_key"] == source_key, "sen_slope_pct_per_year"].dropna().tolist()
            for source_key in source_order
        ]
        box = ax.boxplot(
            level_data,
            positions=[1, 2, 3],
            widths=0.6,
            patch_artist=True,
            boxprops={"edgecolor": "#37474f"},
            medianprops={"color": "#d32f2f", "linewidth": 1.8},
            whiskerprops={"color": "#546e7a"},
            capprops={"color": "#546e7a"},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "#78909c",
                "markeredgecolor": "#78909c",
                "alpha": 0.45,
            },
        )
        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)

        n_basins = int(
            basin_group_counts.loc[basin_group_counts["group_id"] == group_id, "n_basins"].iloc[0]
        )
        ax.axhline(0, color="#616161", linestyle="--", linewidth=0.9)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(source_titles)
        ax.set_ylabel("Trend (%/year)")
        ax.set_title(f"({chr(97 + group_id)}) {GROUP_LABELS[group_id]} (n={n_basins})")
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)

    fig.suptitle("Figure 8 trend experiment: P, Q_nat, and Q_hum annual RCI migration", y=0.98)
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_figure8_outputs()
