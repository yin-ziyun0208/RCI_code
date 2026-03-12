# RCI Code Main

This repository calculates the **Relative Centroid Index (RCI)** along river
main stems for runoff and precipitation, both at the whole-basin scale and at
Pfafstetter-derived subbasin scales.

The project is organized around **two active entry scripts**:

- [single_basin_workflows.py](code/single_basin_workflows.py)
- [batch_basin_workflows.py](code/batch_basin_workflows.py)

Both scripts operate on **packaged basin folders**. The code no longer depends
on the former raw global fallback layout.

## What The Repository Does

The workflows support five main tasks:

- generate reach-level **Pfafstetter codes**
- compute basin-scale **runoff RCI**
- compute basin-scale **precipitation RCI**
- compute **multi-level runoff RCI**
- compute **multi-level precipitation RCI**

The same logic is exposed through:

- a **single-basin** CLI for one packaged basin folder
- a **batch** CLI for many packaged basin folders under one directory

## Repository Structure

```text
.
├── code/
│   ├── single_basin_workflows.py
│   ├── batch_basin_workflows.py
│   └── core/
├── data/
├── examples/
├── scripts/
├── results/
├── outputs/
├── reference/
├── environment.yml
└── requirements.txt
```

See:

- [data/README.md](data/README.md)
- [examples/README.md](examples/README.md)
- [scripts/README.md](scripts/README.md)
- [results/README.md](results/README.md)

## Environment

Recommended setup:

```bash
mamba env create -f environment.yml
mamba activate rci
```

If the environment already exists:

```bash
mamba env update -f environment.yml --prune
mamba activate rci
```

Run all commands from the repository root:

```bash
cd /path/to/RCI_code-main
```

The repository is tested against the dependency set pinned in
[`environment.yml`](environment.yml). For reproducible paper workflows, prefer
that file over ad hoc package installation.

## Data Model

### Single Basin

The default local example basin is:

- [data/basins/poyang](data/basins/poyang)

### Global Basins

The default global basin directory is:

- [data/basins/global](data/basins/global)

Each packaged basin folder uses the same file names:

```text
<BASIN_DIR>/
├── river_network.*
├── catchments.*
├── grades_discharge.csv
├── mswep_precipitation_upstream_totals.csv
├── mswep_precipitation_upstream_totals.meta.json
├── pfaf_codes.csv
├── pfaf_report.json
└── metadata.json
```

### Required Vs Optional Files

- `river_network.*`
  Required for all workflows.
- `catchments.*`
  Required for precipitation workflows only when
  `mswep_precipitation_upstream_totals.csv` is missing and must be rebuilt from
  a gridded field.
- `grades_discharge.csv`
  Required for runoff workflows.
- `mswep_precipitation_upstream_totals.csv`
  Recommended for all precipitation workflows. If present, the code reads it
  directly and does not need to touch the large gridded field.
- `pfaf_codes.csv`
  Required for multilevel workflows unless you run `pfaf` first.

## Quick Start

### Single Basin

Generate Pfaf codes for the default Poyang package:

```bash
python code/single_basin_workflows.py pfaf
```

Run basin-scale runoff RCI:

```bash
python code/single_basin_workflows.py runoff
```

Run basin-scale precipitation RCI:

```bash
python code/single_basin_workflows.py precipitation
```

Run multilevel runoff RCI:

```bash
python code/single_basin_workflows.py multilevel --source continuous
```

Run multilevel precipitation RCI:

```bash
python code/single_basin_workflows.py multilevel --source gridded
```

### Batch

Generate Pfaf codes across all packaged global basins:

```bash
python code/batch_basin_workflows.py pfaf
```

Run basin-scale runoff RCI across all packaged global basins:

```bash
python code/batch_basin_workflows.py runoff
```

Run basin-scale precipitation RCI across all packaged global basins:

```bash
python code/batch_basin_workflows.py precipitation
```

Run multilevel runoff RCI across all packaged global basins:

```bash
python code/batch_basin_workflows.py multilevel --source continuous
```

Run multilevel precipitation RCI across all packaged global basins:

```bash
python code/batch_basin_workflows.py multilevel --source gridded
```

## Minimal Reproducibility Path

This section summarizes the smallest command set needed to verify that the
repository works and to locate the analysis scripts used for the paper figures.

### 1. Single-Basin Reproducibility

Use the packaged Poyang basin:

```bash
python code/single_basin_workflows.py pfaf
python code/single_basin_workflows.py runoff
python code/single_basin_workflows.py precipitation
python code/single_basin_workflows.py multilevel --source continuous
python code/single_basin_workflows.py multilevel --source gridded
```

Default input:

- [`data/basins/poyang`](data/basins/poyang)

Main outputs:

- `outputs/single/poyang/runoff/`
- `outputs/single/poyang/precipitation/`
- `outputs/single/poyang/multilevel_continuous/`
- `outputs/single/poyang/multilevel_gridded/`

### 2. Batch Reproducibility

Use the packaged global basin collection:

```bash
python code/batch_basin_workflows.py pfaf
python code/batch_basin_workflows.py runoff
python code/batch_basin_workflows.py precipitation
python code/batch_basin_workflows.py multilevel --source continuous
python code/batch_basin_workflows.py multilevel --source gridded
```

Default input:

- [`data/basins/global`](data/basins/global)

Main outputs:

- `outputs/batch/runoff/`
- `outputs/batch/precipitation_level0/`
- `outputs/batch/multilevel_continuous/`
- `outputs/batch/multilevel_gridded/`

### 3. Figure Script Mapping

Figure-oriented scripts are stored under [`scripts/`](scripts/):

- Figure 3b:
  [`scripts/1_figure3b_poyang_runoff_boxplot.py`](scripts/1_figure3b_poyang_runoff_boxplot.py)
- Figure 4:
  [`scripts/2_figure4_global_runoff_rci.py`](scripts/2_figure4_global_runoff_rci.py)
- Figure 5:
  [`scripts/3_figure5_global_pq_centroid_distance.py`](scripts/3_figure5_global_pq_centroid_distance.py),
  [`scripts/4_figure5_global_lake_volume_between_pq.py`](scripts/4_figure5_global_lake_volume_between_pq.py),
  [`scripts/5_figure5_global_slope_between_pq.py`](scripts/5_figure5_global_slope_between_pq.py),
  [`scripts/6_figure5d_global_boxplots.py`](scripts/6_figure5d_global_boxplots.py)
- Figure 6:
  [`scripts/7_figure6_global_multilevel_rci.py`](scripts/7_figure6_global_multilevel_rci.py)
- Figure 7:
  [`scripts/8_figure7b_global_seasonal_runoff_centroids.py`](scripts/8_figure7b_global_seasonal_runoff_centroids.py),
  [`scripts/9_figure7c_global_aridity_boxplot.py`](scripts/9_figure7c_global_aridity_boxplot.py)
- Figure 8:
  [`scripts/10_prepare_grdr_annual_reach_flow.py`](scripts/10_prepare_grdr_annual_reach_flow.py),
  [`scripts/11_prepare_figure8_standard_inputs.py`](scripts/11_prepare_figure8_standard_inputs.py),
  [`scripts/12_figure8_global_annual_rci_timeseries.py`](scripts/12_figure8_global_annual_rci_timeseries.py),
  [`scripts/13_figure8_trend_analysis.py`](scripts/13_figure8_trend_analysis.py)

Figure outputs are written under [`results/`](results). See
[`results/README.md`](results/README.md) for file-level details.

## Command Reference

### Single-Basin CLI

Use:

```bash
python code/single_basin_workflows.py <command> [options]
```

Commands:

- `pfaf`
- `runoff`
- `precipitation`
- `multilevel`

#### `pfaf`

Example:

```bash
python code/single_basin_workflows.py pfaf \
  --basin-dir data/basins/poyang \
  --max-level 4
```

Parameters:

- `--basin-dir`
  Basin package directory. Default: `data/basins/poyang`
- `--basin-name`
  Optional label override for outputs and logs.
- `--max-level`
  Optional maximum Pfaf depth. If omitted, subdivision continues until no
  further valid split is possible.
- `--min-unit-reaches`
  Minimum number of reaches required to keep subdividing a Pfaf unit.

#### `runoff`

Example:

```bash
python code/single_basin_workflows.py runoff \
  --basin-dir data/basins/poyang \
  --output outputs/single/poyang/runoff/centroid_results.csv
```

Parameters:

- `--basin-dir`
  Basin package directory.
- `--basin-name`
  Optional label override.
- `--output`
  Output CSV path. If omitted, the default single-basin output folder is used.
- `--min-segments`
  Minimum number of valid main-stem segments required to keep a basin result.

#### `precipitation`

Example:

```bash
python code/single_basin_workflows.py precipitation \
  --basin-dir data/basins/poyang \
  --gridded-data data/climate/climatology/mswep_precipitation_mean.nc
```

Parameters:

- `--basin-dir`
  Basin package directory.
- `--basin-name`
  Optional label override.
- `--gridded-data`
  Path to the gridded precipitation product. This is only needed when the basin
  package does not already contain `mswep_precipitation_upstream_totals.csv`.
- `--variable`
  Variable name in the gridded file. Default: `precipitation`
- `--reduction`
  Reduction applied when non-spatial dimensions exist. Default: `mean`
- `--output`
  Output CSV path.
- `--min-segments`
  Minimum number of valid main-stem segments required to keep a basin result.

#### `multilevel`

Examples:

```bash
python code/single_basin_workflows.py multilevel \
  --source continuous \
  --max-level 4
```

```bash
python code/single_basin_workflows.py multilevel \
  --source gridded \
  --levels 1 2 4
```

Parameters:

- `--basin-dir`
  Basin package directory.
- `--basin-name`
  Optional label override.
- `--source`
  `continuous` for runoff, `gridded` for precipitation.
- `--levels`
  Explicit Pfaf levels to calculate.
- `--max-level`
  Highest Pfaf level to calculate when `--levels` is not supplied.
- `--output`
  Output CSV path.
- `--min-segments`
  Minimum number of reaches required for a Pfaf subbasin to be eligible.
- `--min-mainstem-segments`
  Minimum number of valid main-stem segments required for a subbasin result.
- `--filter-mode`
  Eligibility rule for Pfaf units. Default: `independent_basins`
- `--gridded-data`
  Gridded precipitation field, used only when `--source gridded` and no
  packaged precipitation totals are present.
- `--variable`
  Variable name for gridded precipitation workflows.
- `--reduction`
  Reduction used for non-spatial dimensions in the gridded field.

### Batch CLI

Use:

```bash
python code/batch_basin_workflows.py <command> [options]
```

Commands:

- `pfaf`
- `runoff`
- `precipitation`
- `multilevel`

#### Shared Batch Parameters

- `--global-dir`
  Directory containing packaged basin folders. Default:
  `data/basins/global`
- `--basins`
  Optional list of basin folder names to process instead of the entire global
  set.

#### Batch `pfaf`

Example:

```bash
python code/batch_basin_workflows.py pfaf \
  --global-dir data/basins/global \
  --basins Yangtze Rhine
```

Additional parameters:

- `--output-dir`
  Directory where regenerated `pfaf_codes.csv` and `pfaf_report.json` files are
  written. Default: `data/basins/global`
- `--max-level`
  Optional maximum Pfaf depth.
- `--min-unit-reaches`
  Minimum reaches required to keep subdividing a Pfaf unit.

#### Batch `runoff`

Example:

```bash
python code/batch_basin_workflows.py runoff \
  --basins Yangtze Rhine \
  --output-dir outputs/batch/runoff_subset
```

Additional parameters:

- `--output-dir`
  Output directory for the combined table, failure table, and per-basin tables.
- `--min-segments`
  Minimum valid main-stem segments per basin.

#### Batch `precipitation`

Example:

```bash
python code/batch_basin_workflows.py precipitation \
  --basins Yangtze Rhine \
  --gridded-data data/climate/climatology/mswep_precipitation_mean.nc
```

Additional parameters:

- `--gridded-data`
  Used only for basins that do not already contain
  `mswep_precipitation_upstream_totals.csv`.
- `--variable`
  Variable name in the gridded product.
- `--reduction`
  Reduction for non-spatial dimensions.
- `--output-dir`
  Output directory for combined and per-basin precipitation results.
- `--min-segments`
  Minimum valid main-stem segments per basin.

#### Batch `multilevel`

Example:

```bash
python code/batch_basin_workflows.py multilevel \
  --source continuous \
  --max-level 4
```

```bash
python code/batch_basin_workflows.py multilevel \
  --source gridded \
  --levels 1 2 4 \
  --basins Yangtze Rhine
```

Additional parameters:

- `--source`
  `continuous` or `gridded`
- `--output-dir`
  Output directory. If omitted, the code uses:
  - `outputs/batch/multilevel_continuous`
  - `outputs/batch/multilevel_gridded`
- `--levels`
  Explicit Pfaf levels.
- `--max-level`
  Highest Pfaf level when `--levels` is not supplied.
- `--min-segments`
  Minimum reaches required for a Pfaf subbasin.
- `--min-mainstem-segments`
  Minimum valid main-stem segments required per result.
- `--filter-mode`
  Pfaf unit eligibility rule. Default: `independent_basins`
- `--gridded-data`
  Gridded precipitation field used only when `--source gridded` and packaged
  upstream totals are missing.
- `--variable`
  Variable name in the gridded file.
- `--reduction`
  Reduction for non-spatial dimensions.

## Outputs

### Operational Workflow Outputs

Routine CLI outputs are written under:

- [outputs](outputs)

Typical locations:

- single-basin outputs:
  - `outputs/single/<BASIN>/runoff/`
  - `outputs/single/<BASIN>/precipitation/`
  - `outputs/single/<BASIN>/multilevel_continuous/`
  - `outputs/single/<BASIN>/multilevel_gridded/`
- batch outputs:
  - `outputs/batch/runoff/`
  - `outputs/batch/precipitation_level0/`
  - `outputs/batch/multilevel_continuous/`
  - `outputs/batch/multilevel_gridded/`

### Figure-Specific Analysis Outputs

Figure-specific tables and plots are written under:

- [results](results)

See [results/README.md](results/README.md)
for the mapping from figures to result folders.

## Analysis Scripts

The `scripts/` directory contains figure-specific processing code used after
the core workflows have produced basin-scale and multilevel outputs.

See:

- [scripts/README.md](scripts/README.md)

## Example Cases

Small packaged datasets for smoke testing and GitHub distribution are stored
under:

- [examples](examples)

These cases can be run directly with the same Python entry scripts as the full
project data.

## Reference Material

The manuscript draft and related explanatory material are stored under:

- [reference](reference)

See:

- [reference/README.md](reference/README.md)
