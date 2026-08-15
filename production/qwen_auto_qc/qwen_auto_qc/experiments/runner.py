from __future__ import annotations

import json
from pathlib import Path

from ..config import RunConfig
from ..pipeline.processor import AutoQCPipeline


def run_experiment_matrix(config: RunConfig) -> dict[str, object]:
    modes = config.experiment_modes or [config.inference_mode]
    experiment_root = Path(config.output_root) / "experiments"
    experiment_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for mode in modes:
        mode_config = config.copy_with(
            inference_mode=mode,
            output_root=str(experiment_root / mode),
            checkpoint_root=str(Path(config.checkpoint_root) / mode),
        )
        summary, run_dir = AutoQCPipeline(mode_config).run()
        results.append(
            {
                "mode": mode,
                "run_dir": str(run_dir),
                "error_rate": summary.error_rate,
                "flagged_samples": summary.flagged_samples,
                "total_samples": summary.total_samples,
                "mean_latency_ms": summary.mean_latency_ms,
            }
        )

    summary_payload = {
        "modes": modes,
        "results": results,
    }
    (experiment_root / "experiment_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    return summary_payload
