import pytest
from qwen_auto_qc.config import RunConfig, load_run_config, validate_run_config


def test_default_config_requires_runtime_paths():
    with pytest.raises(ValueError, match="Missing required config values"):
        load_run_config(None)


def test_mlflow_requires_tracking_uri_when_enabled():
    config = RunConfig(model_path="m", images_path="i", labels_path="l")
    config.mlflow.enabled = True
    config.mlflow.tracking_uri = ""
    with pytest.raises(ValueError, match="tracking_uri is missing"):
        validate_run_config(config)


def test_mlflow_rejects_local_file_tracking_uri():
    config = RunConfig(model_path="m", images_path="i", labels_path="l")
    config.mlflow.enabled = True
    config.mlflow.tracking_uri = "file:./mlruns"
    with pytest.raises(ValueError, match="stores runs locally"):
        validate_run_config(config)
