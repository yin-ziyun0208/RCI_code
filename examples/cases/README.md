# Example Cases

This folder stores minimal packaged basin datasets that can be run directly
with the repository's two active workflow CLIs.

## Included Cases

### `single_demo_basin`

- Source basin: packaged Poyang data
- Use case: single-basin workflow smoke testing
- Current size: 63 reaches

Recommended commands:

```bash
python code/single_basin_workflows.py runoff \
  --basin-dir examples/cases/single_demo_basin
```

```bash
python code/single_basin_workflows.py multilevel \
  --source gridded \
  --basin-dir examples/cases/single_demo_basin \
  --gridded-data missing_example_file.nc \
  --max-level 2
```

### `batch_demo_global`

Contains two compact basin packages:

- `yangtze_demo_basin`
- `rhine_demo_basin`

Use case: batch workflow smoke testing

Recommended commands:

```bash
python code/batch_basin_workflows.py runoff \
  --global-dir examples/cases/batch_demo_global
```

```bash
python code/batch_basin_workflows.py multilevel \
  --source continuous \
  --global-dir examples/cases/batch_demo_global \
  --max-level 2
```

## Why The Example Cases Can Run Without Large Grids

Each example basin already includes:

- `grades_discharge.csv`
- `mswep_precipitation_upstream_totals.csv`
- `pfaf_codes.csv`

That means both precipitation and runoff workflows can run directly from the
packaged example folders.
