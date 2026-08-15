# GitHub + Azure Manual Run (No Auto-Deploy Required)

Use this when your repository is private and you cannot grant RBAC for GitHub Actions yet.

## Repo

Current repository: your own Git remote for this project.

## Goal

Run AutoQC in Azure ML while still using GitHub as your source of truth.

Use a Python 3.11-based Azure ML environment for consistency with project pins.

Before submission, export Azure env vars used by job YAML:

- `AUTOQC_AZUREML_IMAGES_PATH`
- `AUTOQC_AZUREML_LABELS_PATH`
- `AUTOQC_AZUREML_IMAGES_DATA_ASSET`
- `AUTOQC_AZUREML_LABELS_DATA_ASSET`
- `AUTOQC_AZUREML_ENVIRONMENT`
- `AUTOQC_AZUREML_COMPUTE`
- `MLFLOW_TRACKING_URI`

## Option A (Recommended): Upload code snapshot from local clone

1. Pull latest code from GitHub locally.
2. In Azure ML job creation, choose **Code source: Local upload**.
3. Upload the project root folder (must contain `run_cli.py` and `configs/azureml.yaml`).
4. Use command:

```bash
python run_cli.py run --config configs/azureml.yaml --images-path ${{inputs.images_path}} --labels-path ${{inputs.labels_path}} --model-path ${{inputs.model_path}}
```

5. Set inputs:

- `images_path`: your Azure ML image data asset (`<images-data-asset>:<version>` or `latest`)
- `labels_path`: your Azure ML label data asset (`<labels-data-asset>:<version>` or `latest`)
- `model_path`: `qwen-vl-4b`

6. Compute: your Azure ML compute target name
7. Environment variable:

- `MLFLOW_TRACKING_URI=<your-workspace-mlflow-uri>`

This gives you reproducible runs without requiring GitHub Actions RBAC changes.

For DAG submissions (`configs/azureml_pipeline_job.yaml`), keep inference command as:
`PYTHONPATH=. python scripts/azure/run_pipeline_inference.py ...`
so package imports resolve correctly in Azure step execution.

### Run multiple DAG jobs for different inference modes

Supported modes:

- `without_red_rectangle`
- `with_red_rectangle`
- `coordinates_text`
- `crop_only`

PowerShell example:

```powershell
$modes = @("without_red_rectangle","with_red_rectangle","coordinates_text","crop_only")
foreach ($mode in $modes) {
  az ml job create `
    --file configs/azureml_pipeline_job.yaml `
    --workspace-name <workspace> `
    --resource-group <resource-group> `
    --set inputs.inference_mode="$mode" `
    --set display_name="qwen-auto-qc-$mode"
}
```

Bash example:

```bash
for mode in without_red_rectangle with_red_rectangle coordinates_text crop_only; do
  az ml job create \
    --file configs/azureml_pipeline_job.yaml \
    --workspace-name <workspace> \
    --resource-group <resource-group> \
    --set inputs.inference_mode="$mode" \
    --set display_name="qwen-auto-qc-$mode"
done
```

If your YAML still contains `${...}` placeholders, use the notebook flow (or first render placeholders) before `az ml job create`.

## Option B: Direct private Git source in Azure ML (if supported in your tenant UI)

1. Create a Git/PAT connection in Azure ML.
2. Use repository URL:

- your Git repository URL for this project

3. Branch: `main`
4. Path: `/`
5. Keep the same command and inputs as Option A.

## Notes

- If model download from Hugging Face is blocked, keep `model_path` as a datastore path and upload model files first.
- If compute is expensive, keep `min nodes=0` and run small smoke tests (`max_samples: 1`) first.
