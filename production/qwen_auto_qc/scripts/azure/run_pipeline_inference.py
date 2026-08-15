from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from qwen_auto_qc.config import load_run_config, validate_run_config
from qwen_auto_qc.pipeline.processor import AutoQCPipeline

# Ensure package imports work when executed as a script in Azure ML command jobs.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AutoQC inference stage from prepared pipeline inputs."
    )
    parser.add_argument("--config", default="configs/azureml.yaml")
    parser.add_argument("--prepared-inputs", required=True)
    parser.add_argument("--images-path")
    parser.add_argument("--labels-path")
    parser.add_argument("--model-path")
    parser.add_argument("--inference-mode")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_inputs = Path(args.prepared_inputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(prepared_inputs.read_text(encoding="utf-8"))
    config = load_run_config(args.config)
    if "images_subdir" in payload and "labels_subdir" in payload:
        selected_images_path = str(prepared_inputs.parent / payload["images_subdir"])
        selected_labels_path = str(prepared_inputs.parent / payload["labels_subdir"])
    else:
        selected_images_path = args.images_path or payload["images_path"]
        selected_labels_path = args.labels_path or payload["labels_path"]
    images_path_obj = Path(selected_images_path)
    labels_path_obj = Path(selected_labels_path)
    image_count = len(list(images_path_obj.rglob("*.jpg"))) if images_path_obj.exists() else 0
    label_count = len(list(labels_path_obj.rglob("*.json"))) if labels_path_obj.exists() else 0
    print(
        json.dumps(
            {
                "event": "inference_input_diagnostics",
                "cwd": os.getcwd(),
                "prepared_inputs_path": str(prepared_inputs),
                "selected_images_path": str(images_path_obj),
                "selected_labels_path": str(labels_path_obj),
                "selected_images_exists": images_path_obj.exists(),
                "selected_labels_exists": labels_path_obj.exists(),
                "selected_image_count_jpg": image_count,
                "selected_label_count_json": label_count,
                "args_images_path": args.images_path,
                "args_labels_path": args.labels_path,
                "payload_images_path": payload.get("images_path"),
                "payload_labels_path": payload.get("labels_path"),
            },
            indent=2,
        )
    )
    config = config.copy_with(
        images_path=selected_images_path,
        labels_path=selected_labels_path,
        output_root=str(output_dir),
        checkpoint_root=str(output_dir / "checkpoints"),
        model_path=args.model_path or config.model_path,
        inference_mode=args.inference_mode or config.inference_mode,
    )
    config = validate_run_config(config)

    summary, run_dir = AutoQCPipeline(config).run()
    stage_summary = {
        "run_dir": str(run_dir),
        "run_id": summary.run_id,
        "metrics": summary.metrics,
        "artifact_paths": summary.artifact_paths,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(stage_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(stage_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
