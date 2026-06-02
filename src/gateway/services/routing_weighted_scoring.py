from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from gateway.services.routing_config_values import nested_dict_or_empty, non_negative_float_or_none, score_or_none

_DEFAULT_SCORE_WEIGHTS = {"quality": 0.5, "cost": 0.3, "latency": 0.2}
_DEFAULT_QUALITY_SCORE = 0.5
_DEFAULT_UNKNOWN_COST_SCORE = 0.0
_DEFAULT_UNKNOWN_LATENCY_SCORE = 0.5


def _score_weight(scoring: Mapping[str, Any], key: str, default: float) -> float:
    weights = scoring.get("weights")
    if isinstance(weights, dict):
        weight_value = weights.get(key, weights.get(f"{key}_weight"))
        if weight_value is not None:
            parsed = non_negative_float_or_none(weight_value)
            return default if parsed is None else parsed
    parsed = non_negative_float_or_none(scoring.get(f"{key}_weight", scoring.get(key)))
    return default if parsed is None else parsed


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
    scoring = nested_dict_or_empty(config, "scoring", "score_weights")
    weights = {
        key: _score_weight(scoring, key, default)
        for key, default in _DEFAULT_SCORE_WEIGHTS.items()
    }
    weight_total = sum(weights.values())
    if weight_total <= 0:
        weights = dict(_DEFAULT_SCORE_WEIGHTS)
        weight_total = sum(weights.values())
    default_quality_score = score_or_none(scoring.get("default_quality_score"))
    if default_quality_score is None:
        default_quality_score = _DEFAULT_QUALITY_SCORE
    unknown_cost_score = score_or_none(scoring.get("unknown_cost_score"))
    if unknown_cost_score is None:
        unknown_cost_score = _DEFAULT_UNKNOWN_COST_SCORE
    unknown_latency_score = score_or_none(scoring.get("unknown_latency_score"))
    if unknown_latency_score is None:
        unknown_latency_score = _DEFAULT_UNKNOWN_LATENCY_SCORE

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
