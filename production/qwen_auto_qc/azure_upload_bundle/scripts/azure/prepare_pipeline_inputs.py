from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and prepare Azure ML pipeline inputs."
    )
    parser.add_argument("--images-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_path = Path(args.images_path)
    labels_path = Path(args.labels_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_path.exists():
        raise FileNotFoundError(f"images path does not exist: {images_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels path does not exist: {labels_path}")

    label_files = sorted(labels_path.rglob("*.json"))
    image_files = sorted(images_path.rglob("*.jpg"))
    if not label_files:
        raise RuntimeError(f"no label files found in: {labels_path}")
    if not image_files:
        raise RuntimeError(f"no image files found in: {images_path}")

    staged_images_dir = output_dir / "images"
    staged_labels_dir = output_dir / "labels"
    staged_images_dir.mkdir(parents=True, exist_ok=True)
    staged_labels_dir.mkdir(parents=True, exist_ok=True)

    image_by_stem = {path.stem: path for path in image_files}
    matched_pairs = 0
    for label_file in label_files:
        image_file = image_by_stem.get(label_file.stem)
        if image_file is None:
            continue
        shutil.copy2(label_file, staged_labels_dir / label_file.name)
        shutil.copy2(image_file, staged_images_dir / image_file.name)
        matched_pairs += 1

    if matched_pairs == 0:
        raise RuntimeError(
            "no matched image/label pairs found for staging: "
            f"labels={len(label_files)}, images={len(image_files)}"
        )

    payload = {
        "images_subdir": "images",
        "labels_subdir": "labels",
        "label_file_count": len(label_files),
        "image_file_count": len(image_files),
        "matched_pair_count": matched_pairs,
    }
    target = output_dir / "prepared_inputs.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
