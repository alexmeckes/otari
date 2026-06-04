from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from gateway.services.routing_config_values import nested_dict_or_empty, non_negative_float_or_none, score_or_none


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


def _score_setting(scoring: Mapping[str, Any], key: str, default: float) -> float:
    parsed = score_or_none(scoring.get(key))
    return default if parsed is None else parsed


def attach_weighted_scores(
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> list[Any]:
    scoring = nested_dict_or_empty(config, "scoring", "score_weights")
    default_weights = {"quality": 0.5, "cost": 0.3, "latency": 0.2}
    weights: dict[str, float] = {}
    configured_weights = scoring.get("weights")
    for key, default in default_weights.items():
        parsed_weight = None
        if isinstance(configured_weights, dict):
            weight_value = configured_weights.get(key, configured_weights.get(f"{key}_weight"))
            parsed_weight = non_negative_float_or_none(weight_value)
        if parsed_weight is None:
            parsed_weight = non_negative_float_or_none(scoring.get(f"{key}_weight", scoring.get(key)))
        weights[key] = default if parsed_weight is None else parsed_weight
    weight_total = sum(weights.values())
    if weight_total <= 0:
        weights = dict(default_weights)
        weight_total = sum(weights.values())
    default_quality_score = _score_setting(scoring, "default_quality_score", 0.5)
    unknown_cost_score = _score_setting(scoring, "unknown_cost_score", 0.0)
    unknown_latency_score = _score_setting(scoring, "unknown_latency_score", 0.5)

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
