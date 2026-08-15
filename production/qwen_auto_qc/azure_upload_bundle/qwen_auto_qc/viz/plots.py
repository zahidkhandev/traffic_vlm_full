from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from ..types import QCDecision


def _safe_text(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str
) -> None:
    try:
        draw.text(xy, text, fill=fill)
    except Exception:
        # Text rendering should never block artifact generation.
        pass


def write_verification_tables(
    all_df: pd.DataFrame, flagged_df: pd.DataFrame, output_dir: str | Path
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_csv = output_path / "all_samples.csv"
    flagged_csv = output_path / "flagged_samples.csv"
    class_summary_csv = output_path / "class_summary.csv"
    confusion_csv = output_path / "confusion_matrix.csv"

    all_df.to_csv(all_csv, index=False)
    flagged_df.to_csv(flagged_csv, index=False)

    required_columns = {
        "given_label",
        "sample_idx",
        "is_error",
        "self_confidence",
        "latency_ms",
        "predicted_label",
    }

    if not required_columns.issubset(set(all_df.columns)):
        class_summary = pd.DataFrame(
            columns=[
                "given_label",
                "total_samples",
                "flagged_samples",
                "mean_self_confidence",
                "mean_latency_ms",
                "error_rate",
            ]
        )
        confusion = pd.DataFrame(
            columns=["given_label", "predicted_label", "samples", "flagged"]
        )
    else:
        class_summary = (
            all_df.groupby("given_label", dropna=False)
            .agg(
                total_samples=("sample_idx", "count"),
                flagged_samples=("is_error", "sum"),
                mean_self_confidence=("self_confidence", "mean"),
                mean_latency_ms=("latency_ms", "mean"),
            )
            .reset_index()
        )
        class_summary["error_rate"] = class_summary["flagged_samples"] / class_summary[
            "total_samples"
        ].clip(lower=1)

        confusion = (
            all_df.groupby(["given_label", "predicted_label"], dropna=False)
            .agg(
                samples=("sample_idx", "count"),
                flagged=("is_error", "sum"),
            )
            .reset_index()
            .sort_values(["given_label", "samples"], ascending=[True, False])
        )

    class_summary.to_csv(class_summary_csv, index=False)
    confusion.to_csv(confusion_csv, index=False)

    return {
        "all_samples_csv": str(all_csv),
        "flagged_samples_csv": str(flagged_csv),
        "class_summary_csv": str(class_summary_csv),
        "confusion_matrix_csv": str(confusion_csv),
    }


def save_annotation_verification_images(
    decisions: list[QCDecision], output_dir: str | Path, max_images: int = 25
) -> str | None:
    image_groups: dict[Path, list[QCDecision]] = defaultdict(list)
    for decision in decisions:
        image_groups[decision.sample.image_path].append(decision)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    saved = 0
    for image_path, grouped_decisions in image_groups.items():
        if saved >= max_images:
            break
        if not image_path.exists():
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        draw = ImageDraw.Draw(image)
        grouped_decisions = sorted(grouped_decisions, key=lambda d: d.sample.sample_idx)
        for decision in grouped_decisions:
            x1, y1, x2, y2 = decision.sample.box
            color = "red" if decision.is_error else "green"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = (
                f"gt={decision.sample.given_label} | "
                f"pred={decision.inference.predicted_label} "
                f"({decision.inference.predicted_confidence:.2f})"
            )
            text_y = max(0, y1 - 14)
            _safe_text(draw, (x1, text_y), label, fill=color)

        out_name = f"{image_path.stem}_verification.png"
        image.save(destination / out_name)
        saved += 1

    return str(destination) if saved > 0 else None


def save_error_crops(
    decisions: list[QCDecision], output_dir: str | Path, max_error_crops: int = 100
) -> str | None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    saved = 0
    error_decisions = [decision for decision in decisions if decision.is_error]
    for idx, decision in enumerate(error_decisions):
        if saved >= max_error_crops:
            break
        if not decision.sample.image_path.exists():
            continue

        try:
            image = Image.open(decision.sample.image_path).convert("RGB")
        except Exception:
            continue

        x1, y1, x2, y2 = decision.sample.box
        crop = image.crop((x1, y1, x2, y2))
        filename = (
            f"error_{idx:04d}_gt_{decision.sample.given_label.replace(' ', '_')}"
            f"_pred_{decision.inference.predicted_label.replace(' ', '_')}"
            f"_conf_{decision.inference.predicted_confidence:.3f}.png"
        )
        crop.save(destination / filename)
        saved += 1

    return str(destination) if saved > 0 else None


def save_placeholder_plot(output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "This project now writes concrete QC visual artifacts under "
            "annotation_verifications/."
        ),
        encoding="utf-8",
    )
