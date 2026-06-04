from collections.abc import Mapping
from typing import Any

from gateway.services.routing_config_values import score_or_none

QUALITY_SCORE_KEYS = ("quality_score", "benchmark_score", "score", "intelligence_score")


def first_score_value(item: Mapping[str, Any], keys: tuple[str, ...] = QUALITY_SCORE_KEYS) -> float | None:
    for key in keys:
        value = score_or_none(item.get(key))
        if value is not None:
            return value
    return None


def candidate_quality_score(item: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> float | None:
    item_score = first_score_value(item)
    if item_score is not None:
        return item_score

    metadata_value: Any = metadata if metadata is not None else item.get("metadata")
    if isinstance(metadata_value, Mapping):
        return first_score_value(metadata_value)
    return None
