from collections.abc import Mapping, Sequence
from typing import Any

from gateway.services.routing_candidate_specs import split_model_selector
from gateway.services.routing_config_values import bool_config, dict_or_empty, non_negative_float_or_none, string_list


def _string_set(value: Any) -> set[str]:
    return set(string_list(value))


def _constraint_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict_or_empty(config.get("constraints"))


def _normalize_model_key_for_constraint(value: str) -> str:
    try:
        _provider, _model_name, normalized = split_model_selector(value)
    except ValueError:
        return value
    return normalized


def _constraint_model_set(value: Any) -> set[str]:
    return {_normalize_model_key_for_constraint(item) for item in _string_set(value)}


def _region_set(value: Any) -> set[str]:
    return {item.strip().lower() for item in _string_set(value) if item.strip()}


def _candidate_regions(candidate: Any) -> set[str]:
    metadata = dict_or_empty(candidate.metadata)
    regions = _region_set(metadata.get("regions"))
    region = metadata.get("region")
    if isinstance(region, str) and region.strip():
        regions.add(region.strip().lower())
    return regions


def _request_region(constraints: Mapping[str, Any], tags: Mapping[str, str]) -> str | None:
    region_tag = constraints.get("region_tag", "region")
    if not isinstance(region_tag, str) or not region_tag.strip():
        return None
    value = tags.get(region_tag.strip())
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _constraint_failure(
    candidate: Any,
    constraints: Mapping[str, Any],
    *,
    tags: Mapping[str, str],
) -> str | None:
    allowed_providers = _string_set(constraints.get("allowed_providers"))
    blocked_providers = _string_set(constraints.get("blocked_providers"))
    allowed_models = _constraint_model_set(constraints.get("allowed_models"))
    blocked_models = _constraint_model_set(constraints.get("blocked_models"))
    allowed_regions = _region_set(constraints.get("allowed_regions"))
    blocked_regions = _region_set(constraints.get("blocked_regions"))

    if allowed_providers and candidate.provider not in allowed_providers:
        return "provider_not_allowed"
    if candidate.provider in blocked_providers:
        return "provider_blocked"
    if allowed_models and candidate.model not in allowed_models:
        return "model_not_allowed"
    if candidate.model in blocked_models:
        return "model_blocked"

    candidate_regions = _candidate_regions(candidate)
    if allowed_regions:
        if not candidate_regions:
            return "region_unknown"
        if not candidate_regions & allowed_regions:
            return "region_not_allowed"
    if blocked_regions and candidate_regions & blocked_regions:
        return "region_blocked"

    if bool_config(constraints.get("require_region_match"), False, coerce_strings=True):
        requested_region = _request_region(constraints, tags)
        if requested_region is not None:
            if not candidate_regions:
                return "region_unknown"
            if requested_region not in candidate_regions:
                return "region_not_supported"

    max_estimated_cost = non_negative_float_or_none(constraints.get("max_estimated_cost"))
    if max_estimated_cost is None:
        return None

    allow_unknown_cost = bool_config(constraints.get("allow_unknown_cost"), False, coerce_strings=True)
    if candidate.estimated_cost is None:
        return None if allow_unknown_cost else "estimated_cost_unknown"
    if candidate.estimated_cost > max_estimated_cost:
        return "estimated_cost_exceeds_max"
    return None


def apply_constraints(
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
    tags: Mapping[str, str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    constraints = _constraint_config(config)
    if not constraints:
        return list(candidates), []

    allowed: list[Any] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        reason = _constraint_failure(candidate, constraints, tags=tags)
        if reason is None:
            allowed.append(candidate)
            continue
        candidate_regions = sorted(_candidate_regions(candidate))
        rejected.append(
            {
                "model": candidate.model,
                "provider": candidate.provider,
                "reason": reason,
                "estimated_cost": candidate.estimated_cost,
                "regions": candidate_regions,
            }
        )
    return allowed, rejected
