from pathlib import Path

from qwen_auto_qc.model_loader import resolve_model_path


def test_resolve_model_path_expands_known_alias():
    resolved = resolve_model_path("qwen-vl-4b")
    if (Path("models") / "qwen-vl-4b").exists():
        assert resolved == str(Path("models") / "qwen-vl-4b")
    else:
        assert resolved == "Qwen/Qwen3-VL-4B-Instruct"


def test_resolve_model_path_keeps_unmapped_model_id():
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    assert resolve_model_path(model_id) == model_id
