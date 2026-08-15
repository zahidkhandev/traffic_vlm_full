from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OBJECT_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]


@dataclass(slots=True)
class DetectionSample:
    sample_idx: int
    image_id: str
    image_path: Path
    label_path: Path
    obj_id: int | str | None
    given_label: str
    given_label_idx: int
    box: list[int]
    occluded: bool = False
    truncated: bool = False


@dataclass(slots=True)
class InferenceResult:
    predicted_label: str
    predicted_label_idx: int
    predicted_confidence: float
    pred_probs: dict[str, float]
    embedding: list[float]
    raw_response: str = ""
    latency_ms: float = 0.0


@dataclass(slots=True)
class QCDecision:
    sample: DetectionSample
    inference: InferenceResult
    self_confidence: float
    margin: float
    normalized_margin: float
    is_error: bool
    reason_code: str = "low_self_confidence"

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_idx": self.sample.sample_idx,
            "image": self.sample.image_path.name,
            "image_id": self.sample.image_id,
            "obj_id": self.sample.obj_id,
            "given_label": self.sample.given_label,
            "given_label_idx": self.sample.given_label_idx,
            "box": self.sample.box,
            "occluded": self.sample.occluded,
            "truncated": self.sample.truncated,
            "predicted_label": self.inference.predicted_label,
            "predicted_label_idx": self.inference.predicted_label_idx,
            "predicted_confidence": self.inference.predicted_confidence,
            "pred_probs": self.inference.pred_probs,
            "self_confidence": self.self_confidence,
            "margin": self.margin,
            "normalized_margin": self.normalized_margin,
            "latency_ms": self.inference.latency_ms,
            "is_error": self.is_error,
            "reason_code": self.reason_code,
        }


@dataclass(slots=True)
class RunSummary:
    run_id: str
    total_samples: int
    flagged_samples: int
    error_rate: float
    mean_latency_ms: float
    thresholds: dict[str, float]
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
