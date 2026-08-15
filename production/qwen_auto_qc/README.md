# Qwen AutoQC

This is the Qwen AutoQC pipeline.

The goal is simple:

- take image + label data
- run Qwen on each labeled object
- find labels that look suspicious
- save the results in a clean format
- compare different prompt modes

## What this does

- reads image + label data
- runs Qwen on each object box
- checks if label looks wrong
- saves outputs
- can log to MLflow

## Pipeline flow

1. load config
2. read images and labels
3. extract valid object boxes
4. run Qwen on each object
5. compute QC scores
6. flag possible label issues
7. save outputs
8. log to MLflow if enabled

## Folder structure

Main folders:

- `qwen_auto_qc/`
- `configs/`
- `scripts/`
- `tests/`
- `docs/`
- `models/`
- `data/`
- `runs/`

Main code areas:

- `qwen_auto_qc/vlm/` : Qwen model inference
- `qwen_auto_qc/pipeline/` : end-to-end processing
- `qwen_auto_qc/analysis/` : scoring logic
- `qwen_auto_qc/viz/` : visualization hooks

Main entrypoints:

- `run_cli.py`
- `scripts/run_autoqc.py` (compatibility wrapper; prefer `run_cli.py`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

PowerShell:

```powershell
.\scripts\bootstrap_local.ps1
```

Linux/macOS:

```bash
sh scripts/bootstrap_local.sh
```

Keep your dataset inside `data/raw/`.
Keep your model inside `models/`.
Use Python 3.11 for local and Azure parity.

Do not commit personal runtime paths.
Pass these through env vars or CLI args.

Example env:

```bash
AUTOQC_MODEL_PATH=models/qwen-vl-4b
AUTOQC_IMAGES_PATH=data/raw/mini/images/test
AUTOQC_LABELS_PATH=data/raw/mini/labels/test
```

Azure env (for `configs/azureml*.yaml` and submit scripts):

```bash
AUTOQC_AZUREML_IMAGES_PATH=azureml://datastores/<datastore>/paths/<images-path>
AUTOQC_AZUREML_LABELS_PATH=azureml://datastores/<datastore>/paths/<labels-path>
AUTOQC_AZUREML_IMAGES_DATA_ASSET=<images-data-asset-name>
AUTOQC_AZUREML_LABELS_DATA_ASSET=<labels-data-asset-name>
AUTOQC_AZUREML_ENVIRONMENT=<azureml-environment-name>
AUTOQC_AZUREML_COMPUTE=<azureml-compute-name>
MLFLOW_TRACKING_URI=<azureml-mlflow-tracking-uri>
```

## Run

```bash
python run_cli.py run --config configs/local.yaml
```

Or pass the paths directly:

```bash
python run_cli.py run --config configs/local.yaml --model-path models/qwen-vl-4b --images-path data/raw/mini/images/test --labels-path data/raw/mini/labels/test
```

Main commands:

- run pipeline

```bash
python run_cli.py run --config configs/local.yaml
```

- benchmark

```bash
python run_cli.py benchmark --config configs/local.yaml
```

- evaluate (runs pipeline and prints summary line)

```bash
python run_cli.py evaluate --config configs/local.yaml
```

- azureml-spec (prints Azure ML job spec JSON from config)

```bash
python run_cli.py azureml-spec --config configs/azureml_job.yaml
```

- run experiment matrix

```bash
python run_cli.py experiment --config configs/experiment.yaml
```

## Azure ML DAG pipeline

Three-step Azure ML pipeline (Data Prep -> Inference -> Evaluation):

- Pipeline job spec: `configs/azureml_pipeline_job.yaml`
- Prep step: `scripts/azure/prepare_pipeline_inputs.py`
- Inference step: `scripts/azure/run_pipeline_inference.py`
- Evaluation step: `scripts/azure/evaluate_pipeline_run.py`

Important runtime notes:

- `prepare_pipeline_inputs.py` scans images/labels recursively, so nested asset layouts are supported.
- Inference command uses `PYTHONPATH=. python ...` in pipeline YAML so `qwen_auto_qc` imports work in Azure job context.
- Evaluation fails quality gate when `total_samples == 0` (zero-sample runs do not pass).

Submit from PowerShell:

```powershell
.\scripts\azure\azureml_submit_pipeline_job.ps1 -WorkspaceName <workspace> -ResourceGroup <resource-group>
```

Submit from bash:

```bash
sh scripts/azure/azureml_submit_pipeline_job.sh <workspace> <resource-group>
```

## Benchmark

```bash
python run_cli.py benchmark --config configs/local.yaml
```

## Tests

```bash
pytest tests --basetemp=pytest_tmp -p no:cacheprovider --cov=qwen_auto_qc --cov-report=term-missing --cov-fail-under=70
```

## Lint

```bash
ruff check qwen_auto_qc tests
ruff format --check qwen_auto_qc tests
```

## Type check

```bash
mypy qwen_auto_qc
```

PowerShell test helper:

```powershell
.\scripts\test_local.ps1
```

Linux/macOS test helper:

```bash
sh scripts/test_local.sh
```

## Containers

Docker build:

```bash
docker build -t qwen-auto-qc:local .
```

Podman build:

```bash
podman build -f Containerfile -t qwen-auto-qc:local .
```

## Outputs

Each run creates a timestamped folder in `runs/`.

Typical files:

- `run_config.json`
- `run_manifest.json`
- `metrics.json`
- `run_summary.json`
- `run.log`
- `all_samples.parquet`
- `flagged_samples.parquet`

If parquet is not available, CSV fallback is used.

Monitoring files:

- `run_manifest.json` : current run info + config diff vs previous run
- `run_index.json` : simple history of all runs in `runs/`

## Current status

What is already done:

- package split into smaller modules
- local CLI works
- tests are passing
- Dockerfile and Containerfile are present
- MLflow logging code is present

What still needs real smoke validation:

- actual Qwen inference on local machine
- MLflow end-to-end local run
- Docker/Podman runtime test with real model + data

## Docs

Suggested reading order:

- `docs/folder_structure.md` (repo map and study sequence)
- `docs/mermaid_charts.md` (visual flow by section)
- `docs/local_run.md` (first run)
- `docs/operations.md` (operational checks and troubleshooting)
- `docs/architecture.md` (design overview)
- `docs/config_reference.md` (field-by-field config)
- `docs/testing.md` (quality gates and coverage)
- `docs/experimentation.md` (prompt mode matrix)
- `docs/container.md` (containerized run)
- `docs/run_tracking.md` (run history and manifests)
- `docs/github_azure_manual_run.md` (manual Azure execution path)

- `docs/architecture.md`
- `docs/folder_structure.md`
- `docs/mermaid_charts.md`
- `docs/config_reference.md`
- `docs/local_run.md`
- `docs/container.md`
- `docs/testing.md`
- `docs/operations.md`
- `docs/experimentation.md`
- `docs/github_azure_manual_run.md`
