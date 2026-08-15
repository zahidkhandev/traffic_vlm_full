# Operations

## Normal local workflow

1. activate env
2. set dataset env vars
3. check config
4. run small smoke test
5. optionally enable MLflow with a central tracking URI
6. run larger sample

## Good first run

Use:

- `max_samples: 1`
- `mlflow.enabled: false`

Command:

```bash
python run_cli.py run --config configs/local.yaml
```

## Good second run

Use:

- `max_samples: 2`
- keep `mlflow.enabled: false` unless you have a central MLflow backend URI
- if enabling MLflow, set `MLFLOW_TRACKING_URI` to a non-`file:` URI

Then open MLflow UI for that backend.

## What to check after each run

Check:

- `runs/run_.../run_summary.json`
- `runs/run_.../metrics.json`
- `runs/run_.../all_samples.parquet`
- `runs/run_.../flagged_samples.parquet`

If MLflow is enabled, also check:

- experiment `qwen_auto_qc` in your configured backend
- local `mlruns/` only when using a local backend (and `mlflow.enabled: false`)

Important:

- local `file:./mlruns` URIs are rejected when `mlflow.enabled: true`
- central tracking URI is required by config validation

## Common problems

### Model does not load

Check:

- model path is correct
- transformers version supports Qwen3-VL
- GPU/CPU config is correct

### `qwen_vl_utils` import fails

This package is required for real inference.
Install the correct Qwen runtime dependencies first.

### No outputs written

Check:

- image path exists
- label path exists
- `max_samples` is not zero
- categories in labels match supported class list

Current behavior:

- if parsing yields zero usable samples, the run fails with skip diagnostics instead of silently succeeding with zero metrics.

### MLflow run missing

Check:

- `mlflow.enabled: true`
- tracking uri is valid
- UI is pointed to same backend store

## Production checklist

Before calling this production ready:

- local smoke test passes
- MLflow run is visible
- ruff passes
- mypy passes
- tests pass
- Docker build works
- Podman build works
- one container smoke run works

## Azure submission checklist

- required env vars are set (`AUTOQC_AZUREML_*`, `MLFLOW_TRACKING_URI`)
- workspace identifiers are passed at submit time (`<workspace>`, `<resource-group>`)
- pipeline job YAML references assets/compute/environment through `${...}` placeholders
- for multi-mode DAG submissions, submit one job per mode (`inputs.inference_mode`)
