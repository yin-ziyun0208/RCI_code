# Analysis Scripts

This directory contains figure-oriented analysis scripts built on top of the
core workflow outputs.

The scripts are ordered by the paper workflow rather than by the historical
order in which they were written.

## General Usage

Run analysis scripts from the repository root:

```bash
cd /path/to/RCI_code-main
mamba activate rci
python scripts/<SCRIPT_NAME>.py
```

Most scripts assume that the packaged basin data under `data/` already exists.
Some scripts also assume that earlier figure-specific intermediate tables are
already available in `results/`.

## Script Index

### 1. Figure 3b

- [1_figure3b_poyang_runoff_boxplot.py](1_figure3b_poyang_runoff_boxplot.py)

Purpose:

- compute Poyang runoff multilevel RCI for levels 1-4
- write boxplot data, summary statistics, and the Figure 3b plot

Outputs:

- `results/figure3b/`

### 2. Figure 4

- [2_figure4_global_runoff_rci.py](2_figure4_global_runoff_rci.py)

Purpose:

- compute basin-scale runoff RCI for the 40 packaged global basins

Outputs:

- `results/figure4/`

### 3. Figure 5

- [3_figure5_global_pq_centroid_distance.py](3_figure5_global_pq_centroid_distance.py)
- [4_figure5_global_lake_volume_between_pq.py](4_figure5_global_lake_volume_between_pq.py)
- [5_figure5_global_slope_between_pq.py](5_figure5_global_slope_between_pq.py)
- [6_figure5d_global_boxplots.py](6_figure5d_global_boxplots.py)

Purposes:

- build the global `P-Q` centroid distance table
- summarize lake volume in the `P-Q` contributing domain
- summarize main-stem slope along the `P-Q` segment
- build the Figure 5d boxplots

Outputs:

- `results/figure5/`

### 4. Figure 6

- [7_figure6_global_multilevel_rci.py](7_figure6_global_multilevel_rci.py)

Purpose:

- combine level 0 basin-scale results with levels 1-4 multilevel results
- build global runoff and precipitation RCI tables and the Figure 6 boxplot

Outputs:

- `results/figure6/`

### 5. Figure 7

- [8_figure7b_global_seasonal_runoff_centroids.py](8_figure7b_global_seasonal_runoff_centroids.py)
- [9_figure7c_global_aridity_boxplot.py](9_figure7c_global_aridity_boxplot.py)

Purposes:

- compute seasonal runoff centroids for Figure 7b
- group seasonal runoff and precipitation RCI values by aridity class for
  Figure 7c

Outputs:

- `results/figure7/`

### 6. Figure 8

- [10_prepare_grdr_annual_reach_flow.py](10_prepare_grdr_annual_reach_flow.py)
- [11_prepare_figure8_standard_inputs.py](11_prepare_figure8_standard_inputs.py)
- [12_figure8_global_annual_rci_timeseries.py](12_figure8_global_annual_rci_timeseries.py)
- [13_figure8_trend_analysis.py](13_figure8_trend_analysis.py)

Purposes:

- preprocess annual GRDR reach discharge tables
- standardize annual `GRADES`, `GRDR`, and `MSWEP` inputs
- build annual RCI time series for `Q_nat`, `Q_hum`, and `P`
- run the Figure 8 trend grouping analysis

Outputs:

- `results/figure8/`

## Practical Notes

- Scripts `10` to `13` form one pipeline for Figure 8.
- Script `10` is designed for large external annual GRDR inputs and can be run
  outside the main repository if needed.
- Most scripts write both:
  - a primary figure table
  - one or more plotting tables or summaries
