from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

MODEL_ID_ALIASES = {
    "qwen-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
}


@dataclass(slots=True)
class LoadedModel:
    processor: object
    model: object
    device: str


def resolve_torch_device(requested: str) -> tuple[str, torch.dtype]:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu", torch.float32
    return requested, torch.float16 if requested == "cuda" else torch.float32


def resolve_model_path(model_path: str) -> str:
    normalized = model_path.strip()
    direct_path = Path(normalized)
    if direct_path.exists():
        return normalized

    local_models_path = Path("models") / normalized
    if local_models_path.exists():
        return str(local_models_path)

    alias_target = MODEL_ID_ALIASES.get(normalized.lower())
    if alias_target is None:
        return normalized

    alias_local_path = Path("models") / normalized.lower()
    if alias_local_path.exists():
        return str(alias_local_path)

    return alias_target


def load_qwen_model(model_path: str, device: str = "auto") -> LoadedModel:
    try:
        from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers with Qwen3-VL support is required.") from exc

    resolved_device, torch_dtype = resolve_torch_device(device)
    resolved_model_path = resolve_model_path(model_path)
    try:
        try:
            processor = AutoProcessor.from_pretrained(
                resolved_model_path, trust_remote_code=True
            )
        except ImportError as proc_exc:
            message = str(proc_exc)
            if (
                "Qwen3VLVideoProcessor requires the Torchvision library" not in message
                and "Torchvision library" not in message
            ):
                raise

            tokenizer = AutoTokenizer.from_pretrained(
                resolved_model_path, trust_remote_code=True
            )
            image_processor = AutoImageProcessor.from_pretrained(
                resolved_model_path, trust_remote_code=True
            )
            video_processor_cls = Qwen3VLProcessor.get_possibly_dynamic_module(
                "BaseVideoProcessor"
            )
            video_processor = video_processor_cls.__new__(video_processor_cls)
            processor = Qwen3VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=cast(Any, video_processor),
                chat_template=tokenizer.chat_template,
            )

        model = cast(
            Any,
            Qwen3VLForConditionalGeneration.from_pretrained(
                resolved_model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ),
        )
        model.to(resolved_device)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load Qwen model. "
            f"Configured model_path='{model_path}', resolved='{resolved_model_path}'. "
            "Use a valid Hugging Face repo id (for example "
            "'Qwen/Qwen3-VL-4B-Instruct') or a local model directory. "
            "If the repo is private or gated, authenticate first with `hf auth login` "
            "or set `HF_TOKEN`."
        ) from exc
    model.eval()
    return LoadedModel(processor=processor, model=model, device=resolved_device)
