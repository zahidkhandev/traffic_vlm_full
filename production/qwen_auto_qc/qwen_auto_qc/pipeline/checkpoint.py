from __future__ import annotations

import json
from pathlib import Path


def save_processed_images(checkpoint_path: Path, processed_image_ids: set[str]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps({"processed_images": sorted(processed_image_ids)}, indent=2),
        encoding="utf-8",
    )


def load_processed_images(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return set(payload.get("processed_images", []))
