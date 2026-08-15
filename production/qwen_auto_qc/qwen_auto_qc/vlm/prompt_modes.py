from __future__ import annotations

SUPPORTED_INFERENCE_MODES = [
    "without_red_rectangle",
    "with_red_rectangle",
    "coordinates_text",
    "crop_only",
]


def validate_inference_mode(mode: str) -> str:
    if mode not in SUPPORTED_INFERENCE_MODES:
        allowed = ", ".join(SUPPORTED_INFERENCE_MODES)
        raise ValueError(f"Unsupported inference_mode '{mode}'. Allowed: {allowed}")
    return mode
