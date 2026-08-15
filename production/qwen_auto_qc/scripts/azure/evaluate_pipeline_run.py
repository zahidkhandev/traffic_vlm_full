from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AutoQC inference output as a pipeline stage."
    )
    parser.add_argument("--inference-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-error-rate", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inference_output = Path(args.inference_output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_summary_path = inference_output / "inference_summary.json"
    if not inference_summary_path.exists():
        raise FileNotFoundError(f"missing inference summary: {inference_summary_path}")
    inference_summary = json.loads(inference_summary_path.read_text(encoding="utf-8"))

    metrics = inference_summary.get("metrics", {})
    error_rate = float(metrics.get("error_rate", 1.0))
    total_samples = int(metrics.get("total_samples", 0))
    passed = total_samples > 0 and error_rate <= args.max_error_rate

    evaluation = {
        "run_id": inference_summary.get("run_id"),
        "run_dir": inference_summary.get("run_dir"),
        "metrics": metrics,
        "quality_gate": {
            "max_error_rate": args.max_error_rate,
            "actual_error_rate": error_rate,
            "total_samples": total_samples,
            "passed": passed,
        },
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    print(json.dumps(evaluation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
