# Examples

This directory contains **small runnable basin packages** for demonstration,
smoke testing, and repository distribution.

Unlike the full `data/` bundle, these cases are intentionally compact while
still preserving the standard basin package layout.

## Included Example Sets

- [examples/cases/single_demo_basin](cases/single_demo_basin)
  One compact basin package for the single-basin CLI.
- [examples/cases/batch_demo_global](cases/batch_demo_global)
  Two compact basin packages for the batch CLI.

See:

- [examples/cases/README.md](cases/README.md)

## Why These Cases Exist

The repository may be shared without the full production `data/` directory.
These small packages let a user immediately test:

- Pfaf generation
- basin-scale runoff RCI
- basin-scale precipitation RCI
- multilevel runoff RCI
- multilevel precipitation RCI

## Single-Basin Demo Commands

Run from the repository root:

```bash
python code/single_basin_workflows.py pfaf \
  --basin-dir examples/cases/single_demo_basin
```

```bash
python code/single_basin_workflows.py runoff \
  --basin-dir examples/cases/single_demo_basin
```

```bash
python code/single_basin_workflows.py precipitation \
  --basin-dir examples/cases/single_demo_basin \
  --gridded-data missing_example_file.nc
```

```bash
python code/single_basin_workflows.py multilevel \
  --source continuous \
  --basin-dir examples/cases/single_demo_basin \
  --max-level 2
```

```bash
python code/single_basin_workflows.py multilevel \
  --source gridded \
  --basin-dir examples/cases/single_demo_basin \
  --gridded-data missing_example_file.nc \
  --max-level 2
```

The precipitation examples intentionally use a non-existent `--gridded-data`
path to show that the packaged
`mswep_precipitation_upstream_totals.csv` file is sufficient.

## Batch Demo Commands

```bash
python code/batch_basin_workflows.py pfaf \
  --global-dir examples/cases/batch_demo_global \
  --output-dir examples/cases/batch_demo_global
```

```bash
python code/batch_basin_workflows.py runoff \
  --global-dir examples/cases/batch_demo_global
```

```bash
python code/batch_basin_workflows.py precipitation \
  --global-dir examples/cases/batch_demo_global \
  --gridded-data missing_example_file.nc
```

```bash
python code/batch_basin_workflows.py multilevel \
  --source continuous \
  --global-dir examples/cases/batch_demo_global \
  --max-level 2
```

```bash
python code/batch_basin_workflows.py multilevel \
  --source gridded \
  --global-dir examples/cases/batch_demo_global \
  --gridded-data missing_example_file.nc \
  --max-level 2
```
