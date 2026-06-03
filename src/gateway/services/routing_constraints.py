from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


def _normalize_model_key_for_constraint(value: str) -> str:
    try:
        _provider, _model_name, normalized = split_model_selector(value)
    except ValueError:
        return value
    return normalized


def _region_set(value: Any) -> set[str]:
    return {item.lower() for item in _string_set(value)}


@dataclass(frozen=True)
class _ConstraintSets:
    allowed_providers: set[str]
    blocked_providers: set[str]
    allowed_models: set[str]
    blocked_models: set[str]
    allowed_regions: set[str]
    blocked_regions: set[str]


def _constraint_sets(constraints: Mapping[str, Any]) -> _ConstraintSets:
    return _ConstraintSets(
        allowed_providers=_string_set(constraints.get("allowed_providers")),
        blocked_providers=_string_set(constraints.get("blocked_providers")),
        allowed_models={
            _normalize_model_key_for_constraint(item) for item in _string_set(constraints.get("allowed_models"))
        },
        blocked_models={
            _normalize_model_key_for_constraint(item) for item in _string_set(constraints.get("blocked_models"))
        },
        allowed_regions=_region_set(constraints.get("allowed_regions")),
        blocked_regions=_region_set(constraints.get("blocked_regions")),
    )


def _provider_model_failure(candidate: Any, constraint_sets: _ConstraintSets) -> str | None:
    if constraint_sets.allowed_providers and candidate.provider not in constraint_sets.allowed_providers:
        return "provider_not_allowed"
    if candidate.provider in constraint_sets.blocked_providers:
        return "provider_blocked"
    if constraint_sets.allowed_models and candidate.model not in constraint_sets.allowed_models:
        return "model_not_allowed"
    if candidate.model in constraint_sets.blocked_models:
        return "model_blocked"
    return None


def _region_failure(
    candidate_regions: set[str],
    constraint_sets: _ConstraintSets,
    requested_region: str | None,
) -> str | None:
    if constraint_sets.allowed_regions:
        if not candidate_regions:
            return "region_unknown"
        if not candidate_regions & constraint_sets.allowed_regions:
            return "region_not_allowed"
    if constraint_sets.blocked_regions and candidate_regions & constraint_sets.blocked_regions:
        return "region_blocked"

    if requested_region is not None:
        if not candidate_regions:
            return "region_unknown"
        if requested_region not in candidate_regions:
            return "region_not_supported"

    return None


def apply_constraints(
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
    tags: Mapping[str, str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    constraints = dict_or_empty(config.get("constraints"))
    if not constraints:
        return list(candidates), []

    constraint_sets = _constraint_sets(constraints)
    requested_region = None
    if bool_config(constraints.get("require_region_match"), False, coerce_strings=True):
        region_tag = string_or_none(constraints.get("region_tag", "region"))
        if region_tag is not None:
            region = string_or_none(tags.get(region_tag))
            requested_region = region.lower() if region is not None else None
    max_estimated_cost = non_negative_float_or_none(constraints.get("max_estimated_cost"))
    allow_unknown_cost = bool_config(
        constraints.get("allow_unknown_cost"),
        False,
        coerce_strings=True,
    )
    allowed: list[Any] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        metadata = dict_or_empty(candidate.metadata)
        candidate_regions = _region_set(metadata.get("regions"))
        region = string_or_none(metadata.get("region"))
        if region is not None:
            candidate_regions.add(region.lower())
        reason = _provider_model_failure(candidate, constraint_sets)
        if reason is None:
            reason = _region_failure(
                candidate_regions,
                constraint_sets,
                requested_region,
            )
        if reason is None:
            if max_estimated_cost is not None:
                if candidate.estimated_cost is None:
                    reason = None if allow_unknown_cost else "estimated_cost_unknown"
                elif candidate.estimated_cost > max_estimated_cost:
                    reason = "estimated_cost_exceeds_max"
        if reason is None:
            allowed.append(candidate)
            continue
        rejected.append(
            {
                "model": candidate.model,
                "provider": candidate.provider,
                "reason": reason,
                "estimated_cost": candidate.estimated_cost,
                "regions": sorted(candidate_regions),
            }
        )
    return allowed, rejected
