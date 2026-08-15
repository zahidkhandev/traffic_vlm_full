from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig, save_run_config
from .types import QCDecision, RunSummary
from .viz.plots import (
    save_annotation_verification_images,
    save_error_crops,
    write_verification_tables,
)


def create_run_dir(output_root: str) -> Path:
    run_id = datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_table(df: pd.DataFrame, target: Path) -> Path:
    try:
        df.to_parquet(target, index=False)
        return target
    except Exception:
        fallback = target.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _find_previous_run(output_root: Path, current_run_id: str) -> Path | None:
    candidates = sorted(
        path
        for path in output_root.glob("run_*")
        if path.is_dir() and path.name != current_run_id
    )
    return candidates[-1] if candidates else None


def _config_diff(
    current_config: dict[str, Any], previous_config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    current_flat = _flatten_dict(current_config)
    previous_flat = _flatten_dict(previous_config)
    changed: dict[str, dict[str, Any]] = {}
    for key in sorted(set(current_flat) | set(previous_flat)):
        current_value = current_flat.get(key)
        previous_value = previous_flat.get(key)
        if current_value != previous_value:
            changed[key] = {"current": current_value, "previous": previous_value}
    return changed


def _write_run_manifest(
    config: RunConfig,
    summary: RunSummary,
    run_dir: Path,
    output_root: Path,
) -> None:
    config_dict = config.to_dict()
    config_hash = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()

    previous_run = _find_previous_run(output_root, run_dir.name)
    previous_run_id = previous_run.name if previous_run else None
    config_changes: dict[str, dict[str, Any]] = {}

    if previous_run is not None:
        previous_config_path = previous_run / "run_config.json"
        if previous_config_path.exists():
            previous_config = json.loads(previous_config_path.read_text(encoding="utf-8"))
            config_changes = _config_diff(config_dict, previous_config)

    manifest = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_hash": config_hash,
        "previous_run_id": previous_run_id,
        "changed_config_keys": sorted(config_changes.keys()),
        "config_changes": config_changes,
        "metrics": summary.metrics,
        "artifact_paths": summary.artifact_paths,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _update_run_index(
    config: RunConfig, summary: RunSummary, run_dir: Path, output_root: Path
) -> None:
    index_path = output_root / "run_index.json"
    entries: list[dict[str, Any]] = []
    if index_path.exists():
        entries = json.loads(index_path.read_text(encoding="utf-8"))

    entries.append(
        {
            "run_id": summary.run_id,
            "run_dir": str(run_dir),
            "inference_mode": config.to_dict().get("inference_mode"),
            "total_samples": summary.total_samples,
            "flagged_samples": summary.flagged_samples,
            "error_rate": summary.error_rate,
            "mean_latency_ms": summary.mean_latency_ms,
        }
    )
    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _macro_precision_recall(decisions: list[QCDecision]) -> tuple[float, float]:
    if not decisions:
        return 0.0, 0.0

    labels = sorted(
        {decision.sample.given_label for decision in decisions}
        | {decision.inference.predicted_label for decision in decisions}
    )
    if not labels:
        return 0.0, 0.0

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    for label in labels:
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for decision in decisions:
            given = decision.sample.given_label
            predicted = decision.inference.predicted_label
            if predicted == label and given == label:
                true_pos += 1
            elif predicted == label and given != label:
                false_pos += 1
            elif predicted != label and given == label:
                false_neg += 1
        precision_denom = true_pos + false_pos
        recall_denom = true_pos + false_neg
        precision_scores.append(true_pos / precision_denom if precision_denom else 0.0)
        recall_scores.append(true_pos / recall_denom if recall_denom else 0.0)

    macro_precision = sum(precision_scores) / len(precision_scores)
    macro_recall = sum(recall_scores) / len(recall_scores)
    return macro_precision, macro_recall


def persist_run(
    config: RunConfig,
    decisions: list[QCDecision],
    thresholds: dict[str, float],
    run_dir: Path,
) -> RunSummary:
    output_root = run_dir.parent
    records = [decision.to_record() for decision in decisions]
    all_df = pd.DataFrame(records)
    if "is_error" in all_df.columns:
        flagged_df = all_df[all_df["is_error"] == True].copy()  # noqa: E712
    else:
        # Handle empty runs (or unexpected record schemas) without crashing.
        flagged_df = all_df.iloc[0:0].copy()

    all_path = _write_table(all_df, run_dir / "all_samples.parquet")
    flagged_path = _write_table(flagged_df, run_dir / "flagged_samples.parquet")
    verification_artifacts = write_verification_tables(
        all_df, flagged_df, run_dir / "tables"
    )

    visual_artifacts: dict[str, str] = {}
    if config.generate_visual_artifacts:
        verification_dir = save_annotation_verification_images(
            decisions,
            run_dir / "annotation_verifications",
            max_images=config.max_verification_images,
        )
        error_crops_dir = save_error_crops(
            decisions,
            run_dir / "label_errors_to_verify",
            max_error_crops=config.max_error_crops,
        )
        if verification_dir is not None:
            visual_artifacts["annotation_verifications"] = verification_dir
        if error_crops_dir is not None:
            visual_artifacts["label_errors_to_verify"] = error_crops_dir

    latencies = [decision.inference.latency_ms for decision in decisions]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    macro_precision, macro_recall = _macro_precision_recall(decisions)

    summary = RunSummary(
        run_id=run_dir.name,
        total_samples=len(decisions),
        flagged_samples=len(flagged_df),
        error_rate=(len(flagged_df) / len(decisions)) if decisions else 0.0,
        mean_latency_ms=mean_latency,
        thresholds=thresholds,
        metrics={
            "total_samples": float(len(decisions)),
            "flagged_samples": float(len(flagged_df)),
            "error_rate": (len(flagged_df) / len(decisions)) if decisions else 0.0,
            "mean_latency_ms": mean_latency,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
        },
        artifact_paths={
            "all_samples": str(all_path),
            "flagged_samples": str(flagged_path),
            "run_log": str(run_dir / "run.log"),
            **verification_artifacts,
            **visual_artifacts,
        },
    )

    save_run_config(config, run_dir / "run_config.json")
    (run_dir / "metrics.json").write_text(
        json.dumps(summary.metrics, indent=2), encoding="utf-8"
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    _write_run_manifest(config, summary, run_dir, output_root)
    _update_run_index(config, summary, run_dir, output_root)
    return summary
