import pytest
from qwen_auto_qc.vlm.prompt_modes import validate_inference_mode


def test_validate_inference_mode_accepts_supported_modes():
    assert validate_inference_mode("without_red_rectangle") == "without_red_rectangle"
    assert validate_inference_mode("with_red_rectangle") == "with_red_rectangle"
    assert validate_inference_mode("coordinates_text") == "coordinates_text"
    assert validate_inference_mode("crop_only") == "crop_only"


def test_validate_inference_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        validate_inference_mode("random_mode")
