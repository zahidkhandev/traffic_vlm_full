from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..config import RunConfig
from ..model_loader import LoadedModel, load_qwen_model
from ..types import DetectionSample, InferenceResult
from .prompt_modes import validate_inference_mode

CLASS_TOKEN_ALIASES = {
    "traffic light": "trafficlight",
    "traffic sign": "trafficsign",
}
CLASS_ALIAS_REVERSE = {value: key for key, value in CLASS_TOKEN_ALIASES.items()}


@dataclass(slots=True)
class PreparedPrompt:
    image: Image.Image
    prompt: str


class TokenizerProtocol(Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, add_special_tokens: bool = False) -> Any: ...


class ProcessorProtocol(Protocol):
    tokenizer: TokenizerProtocol

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str: ...

    def __call__(self, **kwargs: Any) -> dict[str, Any]: ...

    def batch_decode(
        self, sequences: Any, skip_special_tokens: bool = True
    ) -> list[str]: ...


class ModelProtocol(Protocol):
    device: str | torch.device

    def __call__(self, **kwargs: Any) -> Any: ...

    def generate(self, **kwargs: Any) -> Any: ...


class QwenGroundingInference:
    def __init__(self, config: RunConfig, loaded: LoadedModel | None = None):
        self.config = config
        if loaded is not None:
            self.loaded = loaded
        else:
            model_path = config.model_path
            if model_path is None:
                raise ValueError("model_path must be provided before inference.")
            self.loaded = load_qwen_model(model_path, config.device)
        self.processor = cast(ProcessorProtocol, self.loaded.processor)
        self.model = cast(ModelProtocol, self.loaded.model)
        self.class_names = config.class_names
        self.prompt_names = [
            CLASS_TOKEN_ALIASES.get(name, name) for name in self.class_names
        ]
        self.class_first_token_ids = [
            self.processor.tokenizer(name, add_special_tokens=False).input_ids[0]
            for name in self.prompt_names
        ]

    def _normalize_box(
        self, full_img: Image.Image, box: list[int]
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        width, height = full_img.size
        norm_x1 = int((x1 / width) * 1000)
        norm_y1 = int((y1 / height) * 1000)
        norm_x2 = int((x2 / width) * 1000)
        norm_y2 = int((y2 / height) * 1000)
        return norm_x1, norm_y1, norm_x2, norm_y2

    def build_grounding_prompt(
        self, full_img: Image.Image, box: list[int]
    ) -> PreparedPrompt:
        norm_x1, norm_y1, norm_x2, norm_y2 = self._normalize_box(full_img, box)
        class_names_str = ", ".join(self.prompt_names)
        prompt = (
            f"<|box_start|>({norm_x1},{norm_y1}),({norm_x2},{norm_y2})<|box_end|>\n"
            "What object is inside this box region? "
            f"Choose exactly ONE from: {class_names_str}.\n\n"
            f"Reply with ONLY the single word (no explanation)."
        )
        return PreparedPrompt(image=full_img, prompt=prompt)

    def build_red_rectangle_prompt(
        self, full_img: Image.Image, box: list[int]
    ) -> PreparedPrompt:
        image_with_box = full_img.copy()
        draw = ImageDraw.Draw(image_with_box)
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        class_names_str = ", ".join(self.prompt_names)
        prompt = (
            f"Look at the object inside the red rectangle. "
            f"Choose exactly ONE from: {class_names_str}. "
            f"Reply with ONLY the single word."
        )
        return PreparedPrompt(image=image_with_box, prompt=prompt)

    def build_coordinates_prompt(
        self, full_img: Image.Image, box: list[int]
    ) -> PreparedPrompt:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        class_names_str = ", ".join(self.prompt_names)
        prompt = (
            f"There is one labeled object in this image.\n"
            "The object is at pixel coordinates "
            f"x={x1}, y={y1}, width={width}, height={height}.\n"
            "What is the object at that location? "
            f"Choose exactly ONE from: {class_names_str}.\n\n"
            f"Reply with ONLY the single word."
        )
        return PreparedPrompt(image=full_img, prompt=prompt)

    def build_crop_prompt(
        self, full_img: Image.Image, box: list[int]
    ) -> PreparedPrompt:
        x1, y1, x2, y2 = box
        crop = full_img.crop((x1, y1, x2, y2))
        class_names_str = ", ".join(self.prompt_names)
        prompt = (
            f"What is the main object in this image crop? "
            f"Choose exactly ONE from: {class_names_str}. "
            f"Reply with ONLY the single word."
        )
        return PreparedPrompt(image=crop, prompt=prompt)

    def prepare_prompt(self, full_img: Image.Image, box: list[int]) -> PreparedPrompt:
        mode = validate_inference_mode(self.config.inference_mode)
        if mode == "without_red_rectangle":
            return self.build_grounding_prompt(full_img, box)
        if mode == "with_red_rectangle":
            return self.build_red_rectangle_prompt(full_img, box)
        if mode == "coordinates_text":
            return self.build_coordinates_prompt(full_img, box)
        if mode == "crop_only":
            return self.build_crop_prompt(full_img, box)
        raise ValueError(f"Unsupported inference_mode: {mode}")

    def parse_response(
        self, response_text: str, first_token_logits: torch.Tensor
    ) -> tuple[dict[str, float], str]:
        response_lower = response_text.lower().strip()
        parsed_class = None

        for alias, original in CLASS_ALIAS_REVERSE.items():
            if alias in response_lower:
                parsed_class = original
                break

        if parsed_class is None:
            for cls in self.class_names:
                if cls in response_lower:
                    parsed_class = cls
                    break

        class_logits = []
        for token_id in self.class_first_token_ids:
            class_logits.append(
                float(first_token_logits[token_id])
                if token_id < len(first_token_logits)
                else 0.0
            )

        probs_array = (
            torch.nn.functional.softmax(
                torch.tensor(class_logits, dtype=torch.float32), dim=0
            )
            .cpu()
            .numpy()
        )
        probs = {
            cls: float(prob)
            for cls, prob in zip(self.class_names, probs_array, strict=True)
        }

        if parsed_class is None:
            parsed_class = self.class_names[int(np.argmax(probs_array))]

        return probs, parsed_class

    def _process_vision_info(self, messages: list[dict]) -> tuple[list, None]:
        try:
            module = import_module("qwen_vl_utils")
            process_vision_info = module.process_vision_info
        except (ImportError, AttributeError):
            images = []
            for message in messages:
                for content in message.get("content", []):
                    if content.get("type") == "image":
                        images.append(content.get("image"))
            return images, None

        vision_outputs = process_vision_info(messages)
        image_inputs = vision_outputs[0]
        video_inputs = vision_outputs[1] if len(vision_outputs) > 1 else None
        return image_inputs, video_inputs

    def predict(self, sample: DetectionSample) -> InferenceResult:
        full_img = Image.open(sample.image_path).convert("RGB")
        prepared = self.prepare_prompt(full_img, sample.box)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": prepared.image},
                    {"type": "text", "text": prepared.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        processor_kwargs = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        try:
            # Required by newer Qwen3-VL/transformers builds for M-RoPE.
            inputs = self.processor(**processor_kwargs, return_mm_token_type_ids=True)
        except TypeError:
            # Older processor signatures may not support this argument.
            inputs = self.processor(**processor_kwargs)
        inputs = {
            key: value.to(self.model.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }

        started = time.perf_counter()
        with torch.no_grad():
            common_kwargs = {
                "input_ids": inputs.get("input_ids"),
                "attention_mask": inputs.get("attention_mask"),
                "pixel_values": inputs.get("pixel_values"),
                "image_grid_thw": inputs.get("image_grid_thw"),
                "mm_token_type_ids": inputs.get("mm_token_type_ids"),
            }
            common_kwargs = {
                key: value for key, value in common_kwargs.items() if value is not None
            }
            has_multimodal_grid = (
                "image_grid_thw" in common_kwargs or "video_grid_thw" in common_kwargs
            )
            if has_multimodal_grid and "mm_token_type_ids" not in common_kwargs:
                raise RuntimeError(
                    "Processor did not return mm_token_type_ids for multimodal input. "
                    "Upgrade transformers/Qwen processor or ensure "
                    "return_mm_token_type_ids is supported."
                )
            outputs = self.model(
                **common_kwargs,
                output_hidden_states=True,
            )
            embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0].tolist()

            generated = self.model.generate(
                **common_kwargs,
                max_new_tokens=5,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0

        response = self.processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )[0]
        response_text = response.split("assistant")[-1].strip()
        first_token_logits = generated.scores[0][0]
        pred_probs, predicted_label = self.parse_response(
            response_text, first_token_logits
        )

        return InferenceResult(
            predicted_label=predicted_label,
            predicted_label_idx=self.class_names.index(predicted_label),
            predicted_confidence=pred_probs[predicted_label],
            pred_probs=pred_probs,
            embedding=embedding,
            raw_response=response_text,
            latency_ms=latency_ms,
        )
