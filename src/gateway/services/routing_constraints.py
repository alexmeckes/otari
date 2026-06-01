from collections.abc import Mapping, Sequence
from typing import Any

from gateway.services.routing_candidate_specs import split_model_selector
from gateway.services.routing_config_values import (
    bool_config,
    dict_or_empty,
    non_negative_float_or_none,
    string_list,
    string_or_none,
)


def _string_set(value: Any) -> set[str]:
    return set(string_list(value))


def _constraint_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict_or_empty(config.get("constraints"))


def _constraint_value(constraints: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return constraints.get(key, default)


def _normalize_model_key_for_constraint(value: str) -> str:
    try:
        _provider, _model_name, normalized = split_model_selector(value)
    except ValueError:
        return value
    return normalized


def _constraint_model_set(value: Any) -> set[str]:
    return {_normalize_model_key_for_constraint(item) for item in _string_set(value)}


def _region_set(value: Any) -> set[str]:
    return {item.lower() for item in _string_set(value)}


def _candidate_regions(candidate: Any) -> set[str]:
    metadata = dict_or_empty(candidate.metadata)
    regions = _region_set(metadata.get("regions"))
    region = string_or_none(metadata.get("region"))
    if region is not None:
        regions.add(region.lower())
    return regions


def _request_region(constraints: Mapping[str, Any], tags: Mapping[str, str]) -> str | None:
    region_tag = string_or_none(_constraint_value(constraints, "region_tag", "region"))
    if region_tag is None:
        return None
    region = string_or_none(tags.get(region_tag))
    return region.lower() if region is not None else None


def _region_presence_failure(
    candidate_regions: set[str],
    *,
    matches: bool,
    mismatch_reason: str,
) -> str | None:
    if not candidate_regions:
        return "region_unknown"
    if not matches:
        return mismatch_reason
    return None


def _membership_failure(
    value: str,
    *,
    allowed_values: set[str],
    blocked_values: set[str],
    not_allowed_reason: str,
    blocked_reason: str,
) -> str | None:
    if allowed_values and value not in allowed_values:
        return not_allowed_reason
    if value in blocked_values:
        return blocked_reason
    return None


def _estimated_cost_failure(candidate: Any, constraints: Mapping[str, Any]) -> str | None:
    max_estimated_cost = non_negative_float_or_none(_constraint_value(constraints, "max_estimated_cost"))
    if max_estimated_cost is None:
        return None

    allow_unknown_cost = bool_config(_constraint_value(constraints, "allow_unknown_cost"), False, coerce_strings=True)
    if candidate.estimated_cost is None:
        return None if allow_unknown_cost else "estimated_cost_unknown"
    if candidate.estimated_cost > max_estimated_cost:
        return "estimated_cost_exceeds_max"
    return None


def _constraint_failure(
    candidate: Any,
    constraints: Mapping[str, Any],
    *,
    tags: Mapping[str, str],
) -> str | None:
    allowed_providers = _string_set(_constraint_value(constraints, "allowed_providers"))
    blocked_providers = _string_set(_constraint_value(constraints, "blocked_providers"))
    allowed_models = _constraint_model_set(_constraint_value(constraints, "allowed_models"))
    blocked_models = _constraint_model_set(_constraint_value(constraints, "blocked_models"))
    allowed_regions = _region_set(_constraint_value(constraints, "allowed_regions"))
    blocked_regions = _region_set(_constraint_value(constraints, "blocked_regions"))

    failure = _membership_failure(
        candidate.provider,
        allowed_values=allowed_providers,
        blocked_values=blocked_providers,
        not_allowed_reason="provider_not_allowed",
        blocked_reason="provider_blocked",
    )
    if failure is not None:
        return failure
    failure = _membership_failure(
        candidate.model,
        allowed_values=allowed_models,
        blocked_values=blocked_models,
        not_allowed_reason="model_not_allowed",
        blocked_reason="model_blocked",
    )
    if failure is not None:
        return failure

    candidate_regions = _candidate_regions(candidate)
    if allowed_regions:
        failure = _region_presence_failure(
            candidate_regions,
            matches=bool(candidate_regions & allowed_regions),
            mismatch_reason="region_not_allowed",
        )
        if failure is not None:
            return failure
    if blocked_regions and candidate_regions & blocked_regions:
        return "region_blocked"

    if bool_config(_constraint_value(constraints, "require_region_match"), False, coerce_strings=True):
        requested_region = _request_region(constraints, tags)
        if requested_region is not None:
            failure = _region_presence_failure(
                candidate_regions,
                matches=requested_region in candidate_regions,
                mismatch_reason="region_not_supported",
            )
            if failure is not None:
                return failure

    return _estimated_cost_failure(candidate, constraints)


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
