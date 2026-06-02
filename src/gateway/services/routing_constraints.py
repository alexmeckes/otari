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


@dataclass(frozen=True)
class _CostConstraint:
    max_estimated_cost: float | None
    allow_unknown_cost: bool


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


def _cost_constraint(constraints: Mapping[str, Any]) -> _CostConstraint:
    return _CostConstraint(
        max_estimated_cost=non_negative_float_or_none(constraints.get("max_estimated_cost")),
        allow_unknown_cost=bool_config(
            constraints.get("allow_unknown_cost"),
            False,
            coerce_strings=True,
        ),
    )


def _candidate_regions(candidate: Any) -> set[str]:
    metadata = dict_or_empty(candidate.metadata)
    regions = _region_set(metadata.get("regions"))
    region = string_or_none(metadata.get("region"))
    if region is not None:
        regions.add(region.lower())
    return regions


def _request_region(constraints: Mapping[str, Any], tags: Mapping[str, str]) -> str | None:
    region_tag = string_or_none(constraints.get("region_tag", "region"))
    if region_tag is None:
        return None
    region = string_or_none(tags.get(region_tag))
    return region.lower() if region is not None else None


def _required_request_region(constraints: Mapping[str, Any], tags: Mapping[str, str]) -> str | None:
    if not bool_config(constraints.get("require_region_match"), False, coerce_strings=True):
        return None
    return _request_region(constraints, tags)


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


def _provider_model_failure(candidate: Any, constraint_sets: _ConstraintSets) -> str | None:
    failure = _membership_failure(
        candidate.provider,
        allowed_values=constraint_sets.allowed_providers,
        blocked_values=constraint_sets.blocked_providers,
        not_allowed_reason="provider_not_allowed",
        blocked_reason="provider_blocked",
    )
    if failure is not None:
        return failure

    return _membership_failure(
        candidate.model,
        allowed_values=constraint_sets.allowed_models,
        blocked_values=constraint_sets.blocked_models,
        not_allowed_reason="model_not_allowed",
        blocked_reason="model_blocked",
    )


def _estimated_cost_failure(candidate: Any, cost_constraint: _CostConstraint) -> str | None:
    if cost_constraint.max_estimated_cost is None:
        return None

    if candidate.estimated_cost is None:
        return None if cost_constraint.allow_unknown_cost else "estimated_cost_unknown"
    if candidate.estimated_cost > cost_constraint.max_estimated_cost:
        return "estimated_cost_exceeds_max"
    return None


def _region_failure(
    candidate_regions: set[str],
    constraint_sets: _ConstraintSets,
    requested_region: str | None,
) -> str | None:
    if constraint_sets.allowed_regions:
        failure = _region_presence_failure(
            candidate_regions,
            matches=bool(candidate_regions & constraint_sets.allowed_regions),
            mismatch_reason="region_not_allowed",
        )
        if failure is not None:
            return failure
    if constraint_sets.blocked_regions and candidate_regions & constraint_sets.blocked_regions:
        return "region_blocked"

    if requested_region is not None:
        failure = _region_presence_failure(
            candidate_regions,
            matches=requested_region in candidate_regions,
            mismatch_reason="region_not_supported",
        )
        if failure is not None:
            return failure

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
    requested_region = _required_request_region(constraints, tags)
    cost_constraint = _cost_constraint(constraints)
    allowed: list[Any] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        reason = _provider_model_failure(candidate, constraint_sets)
        if reason is None:
            reason = _region_failure(
                _candidate_regions(candidate),
                constraint_sets,
                requested_region,
            )
        if reason is None:
            reason = _estimated_cost_failure(candidate, cost_constraint)
        if reason is None:
            allowed.append(candidate)
            continue
        rejected.append(
            {
                "model": candidate.model,
                "provider": candidate.provider,
                "reason": reason,
                "estimated_cost": candidate.estimated_cost,
                "regions": sorted(_candidate_regions(candidate)),
            }
        )
    return allowed, rejected
