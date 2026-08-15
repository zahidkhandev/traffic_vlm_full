from __future__ import annotations

import json
from pathlib import Path

from .config import RunConfig
from .types import DetectionSample


class BDDDatasetParser:
    def __init__(self, config: RunConfig):
        self.config = config
        self.class_to_idx = {name: idx for idx, name in enumerate(config.class_names)}

    def iter_samples(self) -> list[DetectionSample]:
        if self.config.labels_path is None or self.config.images_path is None:
            raise ValueError("images_path and labels_path are required.")
        labels_path = Path(self.config.labels_path)
        images_path = Path(self.config.images_path)
        sample_idx = 0
        results: list[DetectionSample] = []
        skipped_unknown_category = 0
        skipped_missing_box = 0
        skipped_small_or_invalid_box = 0
        skipped_missing_image = 0
        total_objects = 0

        label_files = sorted(labels_path.rglob("*.json"))
        if self.config.max_samples is not None:
            label_files = label_files[: self.config.max_samples]

        image_by_id: dict[str, Path] = {}
        for image_file in images_path.rglob("*.jpg"):
            image_by_id.setdefault(image_file.stem, image_file)

        for label_file in label_files:
            image_id = label_file.stem
            image_path = image_by_id.get(image_id)
            if image_path is None or not image_path.exists():
                skipped_missing_image += 1
                continue

            with label_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            for frame in payload.get("frames", []):
                for obj in frame.get("objects", []):
                    total_objects += 1
                    category_raw = obj.get("category")
                    category = (
                        category_raw.strip().lower()
                        if isinstance(category_raw, str)
                        else category_raw
                    )
                    if category not in self.class_to_idx:
                        skipped_unknown_category += 1
                        continue

                    box = obj.get("box2d")
                    if not box:
                        skipped_missing_box += 1
                        continue

                    normalized = self._normalize_box(box)
                    if normalized is None:
                        skipped_small_or_invalid_box += 1
                        continue

                    attributes = obj.get("attributes", {})
                    results.append(
                        DetectionSample(
                            sample_idx=sample_idx,
                            image_id=image_id,
                            image_path=image_path,
                            label_path=label_file,
                            obj_id=obj.get("id"),
                            given_label=category,
                            given_label_idx=self.class_to_idx[category],
                            box=normalized,
                            occluded=bool(attributes.get("occluded", False)),
                            truncated=bool(attributes.get("truncated", False)),
                        )
                    )
                    sample_idx += 1

        if not results:
            raise RuntimeError(
                "No usable samples found after parsing labels. "
                f"Diagnostics: label_files={len(label_files)}, "
                f"image_files={len(image_by_id)}, total_objects={total_objects}, "
                f"skipped_missing_image={skipped_missing_image}, "
                f"skipped_unknown_category={skipped_unknown_category}, "
                f"skipped_missing_box={skipped_missing_box}, "
                f"skipped_small_or_invalid_box={skipped_small_or_invalid_box}."
            )

        return results

    def _normalize_box(self, box: dict[str, int | float]) -> list[int] | None:
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        if x2 <= x1 or y2 <= y1:
            return None
        if (x2 - x1) < self.config.min_box_size or (y2 - y1) < self.config.min_box_size:
            return None
        return [max(0, x1), max(0, y1), max(0, x2), max(0, y2)]
