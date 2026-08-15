from pathlib import Path

from qwen_auto_qc.analysis.scoring import compute_margin, make_decisions
from qwen_auto_qc.types import DetectionSample, InferenceResult


def test_compute_margin():
    self_conf, normalized = compute_margin({"car": 0.2, "truck": 0.8}, "car")
    assert self_conf == 0.2
    assert normalized > 0


def test_make_decisions_flags_low_self_confidence():
    sample = DetectionSample(
        sample_idx=0,
        image_id="img",
        image_path=Path("img.jpg"),
        label_path=Path("img.json"),
        obj_id=1,
        given_label="car",
        given_label_idx=2,
        box=[0, 0, 20, 20],
    )
    inference = InferenceResult(
        predicted_label="truck",
        predicted_label_idx=3,
        predicted_confidence=0.8,
        pred_probs={"car": 0.1, "truck": 0.8},
        embedding=[0.1, 0.2],
    )
    decisions = make_decisions([sample], [inference], {"car": 0.2})
    assert decisions[0].is_error is True
