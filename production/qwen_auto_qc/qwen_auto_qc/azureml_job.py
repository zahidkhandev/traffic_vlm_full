from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import RunConfig


@dataclass(slots=True)
class AzureMLJobSpec:
    display_name: str
    code: str
    command: str
    environment: str
    compute: str
    experiment_name: str
    inputs: dict[str, str]


def _require_value(value: str | None, field_name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{field_name} is required for Azure ML job specs.")
    return value


def build_azureml_command_job(
    config: RunConfig,
    compute: str = "autoqc-sandbox-dev-gpu-cluster",
    environment: str = "qwen-autoqc:latest",
) -> AzureMLJobSpec:
    return AzureMLJobSpec(
        display_name="qwen-auto-qc-batch",
        code=".",
        command=(
            "python run_cli.py run --config configs/azureml.yaml "
            "--images-path ${{inputs.images_path}} "
            "--labels-path ${{inputs.labels_path}} "
            "--model-path ${{inputs.model_path}}"
        ),
        environment=environment,
        compute=compute,
        experiment_name=config.mlflow.experiment_name,
        inputs={
            "images_path": _require_value(config.images_path, "images_path"),
            "labels_path": _require_value(config.labels_path, "labels_path"),
            "model_path": _require_value(config.model_path, "model_path"),
        },
    )


def azureml_job_as_dict(config: RunConfig) -> dict[str, object]:
    return asdict(build_azureml_command_job(config))
