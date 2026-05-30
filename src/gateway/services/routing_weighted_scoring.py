from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _non_negative_float_or_none(value: Any) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _score_or_none(value: Any) -> float | None:
    parsed = _non_negative_float_or_none(value)
    if parsed is None:
        return None
    if parsed <= 1.0:
        return parsed
    if parsed <= 100.0:
        return parsed / 100.0
    return None


def _weighted_scoring_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    scoring = config.get("scoring")
    if isinstance(scoring, dict):
        return scoring
    score_weights = config.get("score_weights")
    return score_weights if isinstance(score_weights, dict) else {}


def _score_weight(scoring: Mapping[str, Any], key: str, default: float) -> float:
    weights = scoring.get("weights")
    weight_value: Any = None
    if isinstance(weights, dict):
        weight_value = weights.get(key, weights.get(f"{key}_weight"))
    if weight_value is None:
        weight_value = scoring.get(f"{key}_weight", scoring.get(key))
    parsed = _non_negative_float_or_none(weight_value)
    return default if parsed is None else parsed


def _weighted_score_weights(config: Mapping[str, Any]) -> dict[str, float]:
    scoring = _weighted_scoring_config(config)
    weights = {
        "quality": _score_weight(scoring, "quality", 0.5),
        "cost": _score_weight(scoring, "cost", 0.3),
        "latency": _score_weight(scoring, "latency", 0.2),
    }
    if sum(weights.values()) <= 0:
        return {"quality": 0.5, "cost": 0.3, "latency": 0.2}
    return weights


def _normalized_lower_is_better(
    value: float | None,
    known_values: Sequence[float],
    *,
    unknown_score: float,
) -> float:
    if value is None:
        return unknown_score
    if not known_values:
        return 1.0
    minimum = min(known_values)
    maximum = max(known_values)
    if maximum == minimum:
        return 1.0
    return max(0.0, min(1.0, 1.0 - ((value - minimum) / (maximum - minimum))))


def attach_weighted_scores(
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[Any]:
    scoring = _weighted_scoring_config(config)
    weights = _weighted_score_weights(config)
    weight_total = sum(weights.values())
    default_quality = _score_or_none(scoring.get("default_quality_score"))
    unknown_cost = _score_or_none(scoring.get("unknown_cost_score"))
    unknown_latency = _score_or_none(scoring.get("unknown_latency_score"))
    default_quality_score = 0.5 if default_quality is None else default_quality
    unknown_cost_score = 0.0 if unknown_cost is None else unknown_cost
    unknown_latency_score = 0.5 if unknown_latency is None else unknown_latency

    known_costs = [
        candidate.estimated_cost
        for candidate in candidates
        if candidate.estimated_cost is not None
    ]
    known_latencies = [
        candidate.average_latency_ms
        for candidate in candidates
        if candidate.average_latency_ms is not None
    ]

    scored: list[Any] = []
    for candidate in candidates:
        quality_score = candidate.quality_score if candidate.quality_score is not None else default_quality_score
        cost_score = _normalized_lower_is_better(
            candidate.estimated_cost,
            known_costs,
            unknown_score=unknown_cost_score,
        )
        latency_score = _normalized_lower_is_better(
            candidate.average_latency_ms,
            known_latencies,
            unknown_score=unknown_latency_score,
        )
        routing_score = (
            (quality_score * weights["quality"])
            + (cost_score * weights["cost"])
            + (latency_score * weights["latency"])
        ) / weight_total
        scored.append(
            replace(
                candidate,
                routing_score=routing_score,
                score_components={
                    "quality": quality_score,
                    "cost": cost_score,
                    "latency": latency_score,
                    "quality_weight": weights["quality"],
                    "cost_weight": weights["cost"],
                    "latency_weight": weights["latency"],
                },
            )
        )
    return scored
