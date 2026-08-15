from __future__ import annotations

import json
from pathlib import Path

from ..analysis.scoring import derive_thresholds, make_decisions
from ..config import RunConfig
from ..dataset import BDDDatasetParser
from ..mlflow_tracking import log_run_to_mlflow
from ..results import create_run_dir, persist_run
from ..types import InferenceResult, RunSummary
from ..vlm.classifier import QwenGroundingInference
from .checkpoint import save_processed_images


class AutoQCPipeline:
    def __init__(
        self, config: RunConfig, inferencer: QwenGroundingInference | None = None
    ):
        self.config = config
        self.parser = BDDDatasetParser(config)
        self.inferencer = inferencer or QwenGroundingInference(config)

    def _emit_log(self, payload: dict, run_log_path: Path | None) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        if self.config.console_log_each_object:
            print(line)
        if run_log_path is not None:
            with run_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")

    def _log_inference(
        self, index: int, sample, result: InferenceResult, run_log_path: Path | None
    ) -> None:
        payload = {
            "event": "object_inference",
            "index": index,
            "sample_idx": sample.sample_idx,
            "image_id": sample.image_id,
            "image": sample.image_path.name,
            "obj_id": sample.obj_id,
            "box": sample.box,
            "gt_label": sample.given_label,
            "pred_label": result.predicted_label,
            "pred_confidence": float(result.predicted_confidence),
            "latency_ms": float(result.latency_ms),
        }
        self._emit_log(payload, run_log_path)

    def _log_decision(
        self, index: int, decision, threshold: float, run_log_path: Path | None
    ) -> None:
        payload = {
            "event": "object_decision",
            "index": index,
            "sample_idx": decision.sample.sample_idx,
            "image_id": decision.sample.image_id,
            "image": decision.sample.image_path.name,
            "obj_id": decision.sample.obj_id,
            "box": decision.sample.box,
            "gt_label": decision.sample.given_label,
            "pred_label": decision.inference.predicted_label,
            "self_confidence": float(decision.self_confidence),
            "pred_confidence": float(decision.inference.predicted_confidence),
            "threshold": float(threshold),
            "margin": float(decision.margin),
            "normalized_margin": float(decision.normalized_margin),
            "latency_ms": float(decision.inference.latency_ms),
            "is_error": bool(decision.is_error),
            "reason_code": decision.reason_code,
        }
        self._emit_log(payload, run_log_path)

    def run(self) -> tuple[RunSummary, Path]:
        run_dir = create_run_dir(self.config.output_root)
        run_log_path = run_dir / "run.log"
        run_log_path.write_text("", encoding="utf-8")
        checkpoint_path = Path(self.config.checkpoint_root) / f"{run_dir.name}.json"

        samples = self.parser.iter_samples()
        results: list[InferenceResult] = []
        processed_image_ids: set[str] = set()
        self._emit_log(
            {
                "event": "run_start",
                "run_id": run_dir.name,
                "sample_count": len(samples),
                "inference_mode": self.config.inference_mode,
            },
            run_log_path,
        )

        for index, sample in enumerate(samples, start=1):
            result = self.inferencer.predict(sample)
            results.append(result)
            processed_image_ids.add(sample.image_id)
            self._log_inference(index, sample, result, run_log_path)

            if index % self.config.checkpoint_every == 0:
                save_processed_images(checkpoint_path, processed_image_ids)

        thresholds = self.config.thresholds or derive_thresholds(samples, results)
        decisions = make_decisions(samples, results, thresholds)
        for index, decision in enumerate(decisions, start=1):
            threshold = thresholds.get(decision.sample.given_label, 0.0)
            self._log_decision(index, decision, threshold, run_log_path)
        summary = persist_run(self.config, decisions, thresholds, run_dir)
        self._emit_log(
            {
                "event": "run_summary",
                "run_id": summary.run_id,
                "total_samples": summary.total_samples,
                "flagged_samples": summary.flagged_samples,
                "error_rate": summary.error_rate,
                "mean_latency_ms": summary.mean_latency_ms,
            },
            run_log_path,
        )
        log_run_to_mlflow(self.config, summary, run_dir)
        return summary, run_dir
