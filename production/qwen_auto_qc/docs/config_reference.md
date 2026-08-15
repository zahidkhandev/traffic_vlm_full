# Config Reference

This project mainly uses `configs/local.yaml`.
Baseline runtime is Python 3.11 (`conda.yaml` and `pyproject.toml` are pinned).

## Main config fields

### `model_path`

Required at runtime.
Pass through env var or CLI.
Path or model id for Qwen.

Example:

```yaml
model_path: models/qwen-vl-4b
```

### `images_path`

Required at runtime.
Pass through env var or CLI.
Path to images (local path or mounted Azure input path).

Use a relative path inside this project.

Example:

```yaml
images_path: data/raw/mini/images/test
```

### `labels_path`

Required at runtime.
Pass through env var or CLI.
Path to labels (local path or mounted Azure input path).

Use a relative path inside this project.

Example:

```yaml
labels_path: data/raw/mini/labels/test
```

### `output_root`

Where run outputs are written.

Example:

```yaml
output_root: runs
```

### `checkpoint_root`

Where checkpoints are written.

Example:

```yaml
checkpoint_root: checkpoints
```

### `device`

Usually:

- `auto`
- `cpu`
- `cuda`

### `max_samples`

Use this for smoke tests first.

Example:

```yaml
max_samples: 2
```

### `checkpoint_every`

Save progress after this many samples.

### `min_box_size`

Boxes smaller than this are ignored.

### `class_names`

Allowed object categories.
Category text from labels is normalized to lowercase before matching.

### `console_log_each_object`

When true, prints one JSON line per object for inference and one JSON line per
object for final decision (`is_error`, threshold, confidence, latency).

### `use_grounding`

Current production path should stay `true`.

## MLflow block

Example:

```yaml
mlflow:
  enabled: true
  tracking_uri: ${MLFLOW_TRACKING_URI}
  experiment_name: qwen_auto_qc
  tags:
    algorithm: qwen_full_image_grounding
    platform: local
```

## Environment variables

Defined in `.env.example`:

```env
AUTOQC_MODEL_PATH=models/qwen-vl-4b
AUTOQC_IMAGES_PATH=data/raw/mini/images/test
AUTOQC_LABELS_PATH=data/raw/mini/labels/test
MLFLOW_TRACKING_URI=file:./mlruns
AUTOQC_AZUREML_IMAGES_PATH=azureml://datastores/<datastore>/paths/<images-path>
AUTOQC_AZUREML_LABELS_PATH=azureml://datastores/<datastore>/paths/<labels-path>
AUTOQC_AZUREML_IMAGES_DATA_ASSET=<images-data-asset-name>
AUTOQC_AZUREML_LABELS_DATA_ASSET=<labels-data-asset-name>
AUTOQC_AZUREML_ENVIRONMENT=<azureml-environment-name>
AUTOQC_AZUREML_COMPUTE=<azureml-compute-name>
```

Note:

- `file:...` tracking URIs are local-only and are rejected by `validate_run_config`.
- For local smoke tests, keep `mlflow.enabled: false`.
- Azure job YAML files use `${...}` placeholders and are expanded by submit scripts at runtime.

## Recommended local smoke config

For first run:

```yaml
max_samples: 1
mlflow:
  enabled: false
```

After that:

```yaml
max_samples: 2
mlflow:
  enabled: true
```

## Zero-sample behavior

If no usable objects are produced after parsing/filtering, the run now fails with diagnostics:

- label file count
- image file count
- total objects seen
- skip counters (missing image, unknown category, missing box, invalid/small box)
