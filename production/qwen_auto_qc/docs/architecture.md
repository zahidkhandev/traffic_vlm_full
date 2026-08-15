# Architecture

## Purpose

This pipeline checks object labels using Qwen.

Input:

- images
- label json files

Output:

- per-object predictions
- QC scores
- flagged label issues
- run summary

## End-to-end flow

1. config is loaded
2. dataset parser reads label files
3. valid object boxes are extracted
4. Qwen gets the full image plus grounding box prompt
5. model returns class probabilities and embedding
6. scoring logic compares predicted confidence vs given label
7. suspicious labels are flagged
8. outputs are saved
9. MLflow logs run metadata if enabled

## Main modules

### `qwen_auto_qc/config.py`

Handles config loading.

Important things it does:

- loads yaml/json config
- expands env vars like `${AUTOQC_IMAGES_PATH}`
- holds runtime settings in `RunConfig`

### `qwen_auto_qc/dataset.py`

Reads BDD-style labels and creates detection samples.

Important things it does:

- scans label/image files recursively (nested folders supported)
- matches image files by stem across nested image folders
- filters categories
- removes invalid or tiny boxes
- normalizes category text to lowercase before class matching
- fails fast with diagnostic counters if zero usable samples remain

### `qwen_auto_qc/vlm/classifier.py`

This is the Qwen inference layer.

Important things it does:

- loads processor and model
- builds grounding prompt using normalized box coordinates
- runs generation
- parses class from model response
- returns class probabilities + embedding + latency

### `qwen_auto_qc/analysis/scoring.py`

This is the QC decision logic.

Important things it does:

- compute self-confidence for the given label
- compute confidence margin
- derive class thresholds
- decide whether a sample is a possible label error

### `qwen_auto_qc/pipeline/processor.py`

This is the main orchestration layer.

Important things it does:

- load samples
- call inference for each sample
- save checkpoints
- run scoring
- persist outputs
- call MLflow logging

### `qwen_auto_qc/results.py`

Writes artifacts for each run.

Important things it does:

- creates run folder
- writes parquet/csv tables
- writes metrics and summary json
- saves the config used for the run

### `qwen_auto_qc/mlflow_tracking.py`

Handles MLflow logging.

Important things it does:

- set tracking uri
- set experiment
- log params
- log metrics
- log thresholds
- log artifacts

## Data model

Main objects:

- `DetectionSample`
- `InferenceResult`
- `QCDecision`
- `RunSummary`

These are defined in `qwen_auto_qc/types.py`.

## Current inference method

Current method is:

- full image
- grounding box token in prompt
- choose one class from fixed class list

This matches the production direction better than crop-only inference.

## Current limits

- true Qwen runtime still depends on external `qwen_vl_utils`
- no real batch inference optimization yet
- no final dashboard yet
- no final drift logic yet

## Azure pipeline behavior

- `prepare_pipeline_inputs.py` validates mounted inputs and counts recursive `*.json` labels and `*.jpg` images.
- `run_pipeline_inference.py` runs under `PYTHONPATH=.` in the pipeline spec to keep imports stable in Azure execution.
- `evaluate_pipeline_run.py` fails quality gate when `total_samples` is zero.
- Azure job specs in `configs/azureml_job.yaml` and `configs/azureml_pipeline_job.yaml` are env-driven (`${AUTOQC_AZUREML_*}` placeholders).
- Submit scripts expand placeholders before `az ml job create`, so no workspace-specific names are committed in YAML.
