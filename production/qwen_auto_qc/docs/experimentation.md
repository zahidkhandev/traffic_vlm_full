# Experimentation

This is the place to test different prompt styles properly.

## Why this is needed

This is an ML project.
We should not keep only one prompt path and assume it is best forever.

We want to test things like:

- full image without red rectangle
- full image with red rectangle
- full image + plain coordinates in text
- crop only

Each mode should run separately and save separate outputs.

## Supported modes

### `without_red_rectangle`

This is the no-red-box version.

Prompt style:

- full image
- box tokens only

### `with_red_rectangle`

This draws the red rectangle on the image and asks the model to look inside it.

### `coordinates_text`

Uses the full image but gives x, y, width, height in plain text.

### `crop_only`

Uses only the cropped object.

## How to run experiment matrix

Use:

```bash
python run_cli.py experiment --config configs/experiment.yaml
```

Azure DAG note:

- keep `configs/azureml_pipeline_job.yaml` env-driven
- submit multiple modes with:
  - `scripts/azure/azureml_submit_pipeline_job.ps1`
  - `scripts/azure/azureml_submit_pipeline_job.sh`
  - or your notebook that writes per-mode temporary YAMLs

## What it does

It runs each mode separately.

For each mode it creates separate runs under:

- `runs/experiments/without_red_rectangle/`
- `runs/experiments/with_red_rectangle/`
- `runs/experiments/coordinates_text/`
- `runs/experiments/crop_only/`

It also writes:

- `runs/experiments/experiment_summary.json`

## What to compare

For each mode compare:

- error rate
- flagged sample count
- latency
- failure cases
- run manifest config changes

## Recommended way to use it

Start small:

- `max_samples: 2`

Main 4-way comparison:

- without red rectangle
- with red rectangle
- coordinates text
- crop only

After that:

- keep best mode as main production mode
- keep experiment config for future testing
