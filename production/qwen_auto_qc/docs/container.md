# Container Guide

This section is step by step.
Use this when you want to run the pipeline inside Docker or Podman.

## Build

```bash
docker build -t qwen-auto-qc:local .
```

## Docker Run

### 1. Make sure config is ready

Check:

- `configs/local.yaml`
- model path
- image path
- label path

If your container paths are different from local paths, update the config first.

### 2. Build image

```bash
docker build -t qwen-auto-qc:local .
```

### 3. Run container

Mount data and model paths that match the config inside the container.

Example:

```bash
docker run --rm \
  -v <host-data-path>:/app/data \
  -v <host-model-path>:/app/qwen-vl-4b \
  qwen-auto-qc:local
```

PowerShell equivalent:

```powershell
docker run --rm `
  -v <host-data-path>:/app/data `
  -v <host-model-path>:/app/qwen-vl-4b `
  qwen-auto-qc:local
```

### 4. What to check after run

Check:

- `runs/`
- `run_summary.json`
- `metrics.json`
- `run_manifest.json`

If MLflow is enabled, also check:

- `mlruns/`

If you want Azure-style config values in container runs, pass env vars with `-e`
for the same keys used in `.env.example` (`AUTOQC_MODEL_PATH`, `AUTOQC_IMAGES_PATH`,
`AUTOQC_LABELS_PATH`, `MLFLOW_TRACKING_URI`, and optional `AUTOQC_AZUREML_*`).

## Podman Build

```bash
podman build -f Containerfile -t qwen-auto-qc:local .
```

## Podman Run

### 1. Build image

```bash
podman build -f Containerfile -t qwen-auto-qc:local .
```

### 2. Run container

```bash
podman run --rm \
  -v <host-data-path>:/app/data \
  -v <host-model-path>:/app/qwen-vl-4b \
  qwen-auto-qc:local
```

Both container files are local to this isolated project root:

- `Dockerfile`
- `Containerfile`
