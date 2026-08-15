from __future__ import annotations

from collections import defaultdict

from ..types import DetectionSample, InferenceResult, QCDecision


def compute_margin(
    pred_probs: dict[str, float], given_label: str
) -> tuple[float, float]:
    self_confidence = float(pred_probs.get(given_label, 0.0))
    ordered = sorted(pred_probs.values(), reverse=True)
    top_conf = ordered[0] if ordered else 0.0
    margin = max(top_conf - self_confidence, 0.0)
    normalized = margin / top_conf if top_conf > 0 else 0.0
    return self_confidence, normalized


def derive_thresholds(
    samples: list[DetectionSample], results: list[InferenceResult]
) -> dict[str, float]:
    by_class: dict[str, list[float]] = defaultdict(list)
    for sample, result in zip(samples, results, strict=True):
        by_class[sample.given_label].append(
            float(result.pred_probs.get(sample.given_label, 0.0))
        )

    thresholds: dict[str, float] = {}
    for label, values in by_class.items():
        if not values:
            thresholds[label] = 0.0
            continue
        # Using min() made thresholds collapse to near-zero when one bad prediction
        # existed, which caused obvious mismatches to be marked "accepted".
        ordered = sorted(values)
        quantile_index = int(round(0.2 * (len(ordered) - 1)))
        thresholds[label] = float(ordered[quantile_index])
    return thresholds


def make_decisions(
    samples: list[DetectionSample],
    results: list[InferenceResult],
    thresholds: dict[str, float],
) -> list[QCDecision]:
    decisions: list[QCDecision] = []
    min_margin = 0.20
    min_pred_conf_for_mismatch = 0.50
    for sample, result in zip(samples, results, strict=True):
        self_confidence, normalized_margin = compute_margin(
            result.pred_probs, sample.given_label
        )
        threshold = thresholds.get(sample.given_label, 0.0)
        predicted_label = result.predicted_label
        predicted_confidence = float(result.predicted_confidence)

        is_mismatch = predicted_label != sample.given_label
        low_self_confidence = self_confidence <= threshold
        high_margin = normalized_margin >= min_margin
        high_alt_confidence = predicted_confidence >= min_pred_conf_for_mismatch

        is_error = is_mismatch and (
            low_self_confidence or (high_margin and high_alt_confidence)
        )

        if not is_mismatch:
            reason_code = "accepted_match"
        elif low_self_confidence:
            reason_code = "mismatch_low_self_confidence"
        elif high_margin and high_alt_confidence:
            reason_code = "mismatch_high_margin"
        else:
            reason_code = "accepted_uncertain_mismatch"

        decisions.append(
            QCDecision(
                sample=sample,
                inference=result,
                self_confidence=self_confidence,
                margin=predicted_confidence - self_confidence,
                normalized_margin=normalized_margin,
                is_error=is_error,
                reason_code=reason_code,
            )
        )
    return decisions
