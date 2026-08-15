import os
import shutil
from pathlib import Path

from qwen_auto_qc.config import load_run_config


def test_load_run_config_expands_environment_variables():
    temp_root = Path("test_output/config_env")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    config_file = temp_root / "config.yaml"
    config_file.write_text(
        (
            "model_path: ${AUTOQC_MODEL_PATH}\n"
            "images_path: ${AUTOQC_IMAGES_PATH}\n"
            "labels_path: ${AUTOQC_LABELS_PATH}\n"
        ),
        encoding="utf-8",
    )
    os.environ["AUTOQC_MODEL_PATH"] = "models/qwen-vl-4b"
    os.environ["AUTOQC_IMAGES_PATH"] = "images/path"
    os.environ["AUTOQC_LABELS_PATH"] = "labels/path"

    config = load_run_config(config_file)

    assert config.model_path == "models/qwen-vl-4b"
    assert config.images_path == "images/path"
    assert config.labels_path == "labels/path"


def test_load_run_config_reads_dotenv_automatically():
    temp_root = Path("test_output/config_dotenv")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    previous_values = {
        key: os.environ.pop(key, None)
        for key in ("AUTOQC_MODEL_PATH", "AUTOQC_IMAGES_PATH", "AUTOQC_LABELS_PATH")
    }
    try:
        os.chdir(temp_root)
        Path(".env").write_text(
            (
                "AUTOQC_MODEL_PATH=models/qwen-vl-4b\n"
                "AUTOQC_IMAGES_PATH=data/raw/mini/images/test\n"
                "AUTOQC_LABELS_PATH=data/raw/mini/labels/test\n"
            ),
            encoding="utf-8",
        )
        config_file = Path("config.yaml")
        config_file.write_text(
            (
                "model_path: ${AUTOQC_MODEL_PATH}\n"
                "images_path: ${AUTOQC_IMAGES_PATH}\n"
                "labels_path: ${AUTOQC_LABELS_PATH}\n"
            ),
            encoding="utf-8",
        )

        config = load_run_config(config_file)

        assert config.model_path == "models/qwen-vl-4b"
        assert config.images_path == "data/raw/mini/images/test"
        assert config.labels_path == "data/raw/mini/labels/test"
    finally:
        os.chdir(old_cwd)
        for key, value in previous_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_load_run_config_leaves_thresholds_empty_when_not_provided():
    temp_root = Path("test_output/config_thresholds")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    config_file = temp_root / "config.yaml"
    config_file.write_text(
        (
            "model_path: ${AUTOQC_MODEL_PATH}\n"
            "images_path: ${AUTOQC_IMAGES_PATH}\n"
            "labels_path: ${AUTOQC_LABELS_PATH}\n"
        ),
        encoding="utf-8",
    )
    os.environ["AUTOQC_MODEL_PATH"] = "models/qwen-vl-4b"
    os.environ["AUTOQC_IMAGES_PATH"] = "images/path"
    os.environ["AUTOQC_LABELS_PATH"] = "labels/path"

    config = load_run_config(config_file)

    assert config.thresholds == {}
