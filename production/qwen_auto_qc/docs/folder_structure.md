# Folder Structure

This document explains what each folder is for and how to study the codebase in a logical order.

## Project root

- `qwen_auto_qc/`  
  Main Python package. Core product code lives here.
- `configs/`  
  YAML configs for local runs, experiments, and Azure ML jobs.
- `scripts/`  
  Helper scripts for local development, testing, linting, typing, Azure jobs, and utilities.
- `tests/`  
  Unit and integration-style tests for config, dataset parsing, inference plumbing, scoring, and pipeline behavior.
- `docs/`  
  Project documentation (run guides, architecture, testing, operations, etc.).
- `runs/`  
  Run artifacts from pipeline executions (metrics, summaries, manifests, tables).
- `outputs/`  
  Additional outputs/logs from some runs and utilities.
- `checkpoints/`  
  Intermediate saved pipeline checkpoints.
- `data/`  
  Local dataset storage area (images and labels).
- `models/`  
  Local model storage area.
- `mlruns/`  
  Local MLflow artifacts if local backend is used.
- `azure_upload_bundle/`  
  Upload-ready bundle for Azure execution (packaged copy of key code/configs/scripts).
- `databricks/`  
  Databricks-related run helper(s).
- `azure_devops/`  
  CI/CD configuration for Azure DevOps.
- `.github/`  
  GitHub-related automation/config.

## `qwen_auto_qc/` package layout

- `__main__.py`, `cli.py`  
  CLI entry and command wiring (`run`, `evaluate`, `benchmark`, `experiment`, `azureml-spec`).
- `config.py`  
  Config dataclasses, env expansion, loading, and validation.
- `types.py`  
  Shared data models/constants used across modules.
- `dataset.py`  
  Label/image parsing and sample generation.
- `model_loader.py`  
  Model and processor loading logic.
- `results.py`  
  Writes run artifacts (metrics, summaries, tables, manifests).
- `mlflow_tracking.py`  
  MLflow logging integration.
- `azureml_job.py`  
  Azure ML job spec helper generation.

Subpackages:

- `vlm/`  
  VLM-specific logic (prompt modes and classifier inference path).
- `analysis/`  
  Scoring/QC decision logic and confident-joint related utilities.
- `pipeline/`  
  End-to-end orchestration and checkpoint flow.
- `experiments/`  
  Experiment-matrix runner across inference modes.
- `viz/`  
  Plot/visual artifact helpers.

## `scripts/` layout

- Root scripts (`run_local.*`, `test_local.*`, `lint_local.*`, `typecheck_local.*`, `bootstrap_local.*`)  
  Convenience wrappers for common workflows.
- `run_autoqc.py`  
  Compatibility wrapper entrypoint.
- `convert_bundle_to_pdf.py`  
  Utility to turn a folder (for example `azure_upload_bundle`) into one text-based PDF.
- `azure/`  
  Azure ML setup/submit/prepare/inference/evaluation scripts.
  Submit scripts expand `${ENV_VAR}` placeholders from Azure YAML specs before job creation.

## `docs/` layout

- `local_run.md`  
  First-run quickstart.
- `operations.md`  
  Operational checklist and troubleshooting.
- `architecture.md`  
  End-to-end design and module responsibilities.
- `config_reference.md`  
  Config field-by-field behavior.
- `testing.md`  
  Test coverage and priorities.
- `experimentation.md`  
  Prompt-mode experiment matrix guide.
- `container.md`  
  Docker/Podman usage.
- `run_tracking.md`  
  Run manifests, metrics, and history files.
- `github_azure_manual_run.md`  
  Manual Azure run flow when auto-deploy is not available.
- `folder_structure.md`  
  This file.

## Recommended study order (to understand every function)

1. `qwen_auto_qc/types.py`
2. `qwen_auto_qc/config.py`
3. `qwen_auto_qc/dataset.py`
4. `qwen_auto_qc/vlm/prompt_modes.py` and `qwen_auto_qc/vlm/classifier.py`
5. `qwen_auto_qc/analysis/scoring.py`
6. `qwen_auto_qc/pipeline/processor.py` and `qwen_auto_qc/pipeline/checkpoint.py`
7. `qwen_auto_qc/results.py` and `qwen_auto_qc/mlflow_tracking.py`
8. `qwen_auto_qc/experiments/runner.py`
9. `qwen_auto_qc/cli.py` and `run_cli.py`
10. `tests/` files alongside each module

Tip:

- For each file, read: inputs -> core function flow -> outputs -> failure paths -> tests.
