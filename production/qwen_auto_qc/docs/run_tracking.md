# Run Tracking

This pipeline now stores run tracking files for each run.

## Files added per run

### `run_config.json`

Exact config used for that run.

### `run_summary.json`

Main run result summary.

### `metrics.json`

Main numeric metrics.

### `run_manifest.json`

Extra run metadata.

It includes:

- run id
- created time
- config hash
- previous run id
- changed config keys
- config diff vs previous run
- metrics
- artifact paths

## Global run tracking

### `runs/run_index.json`

This is a simple history file.

It stores one entry per run:

- run id
- run dir
- inference mode
- total samples
- flagged samples
- error rate
- mean latency

## Why this helps

Now you can:

- compare one run vs previous run
- see which config changed
- see how metrics changed
- track run history without opening every folder manually

For Azure DAG runs, `evaluation_summary.json` also includes:

- `quality_gate.total_samples`
- gate fails when `total_samples == 0`
- per-mode comparisons are easiest when each mode uses a distinct `display_name` pattern
