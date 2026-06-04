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


@dataclass(frozen=True)
class _PreparedConstraints:
    allowed_providers: set[str]
    blocked_providers: set[str]
    allowed_models: set[str]
    blocked_models: set[str]
    allowed_regions: set[str]
    blocked_regions: set[str]
    requested_region: str | None
    max_estimated_cost: float | None
    allow_unknown_cost: bool


def _normalize_model_key_for_constraint(value: str) -> str:
    try:
        _provider, _model_name, normalized = split_model_selector(value)
    except ValueError:
        return value
    return normalized


def _string_set(value: Any) -> set[str]:
    return set(string_list(value))


def _model_constraint_set(value: Any) -> set[str]:
    return {_normalize_model_key_for_constraint(item) for item in string_list(value)}


def _lower_string_set(value: Any) -> set[str]:
    return {item.lower() for item in string_list(value)}


def _provider_model_constraint_failure(
    *,
    provider: str,
    model: str,
    constraints: _PreparedConstraints,
) -> str | None:
    if constraints.allowed_providers and provider not in constraints.allowed_providers:
        return "provider_not_allowed"
    if provider in constraints.blocked_providers:
        return "provider_blocked"
    if constraints.allowed_models and model not in constraints.allowed_models:
        return "model_not_allowed"
    if model in constraints.blocked_models:
        return "model_blocked"
    return None


def _candidate_regions(metadata: Mapping[str, Any]) -> set[str]:
    regions = _lower_string_set(metadata.get("regions"))
    region = string_or_none(metadata.get("region"))
    if region is not None:
        regions.add(region.lower())
    return regions


def _region_constraint_failure(
    candidate_regions: set[str],
    *,
    constraints: _PreparedConstraints,
) -> str | None:
    if constraints.allowed_regions:
        if not candidate_regions:
            return "region_unknown"
        if not candidate_regions & constraints.allowed_regions:
            return "region_not_allowed"
    if constraints.blocked_regions and candidate_regions & constraints.blocked_regions:
        return "region_blocked"
    if constraints.requested_region is None:
        return None
    if not candidate_regions:
        return "region_unknown"
    if constraints.requested_region not in candidate_regions:
        return "region_not_supported"
    return None


def _requested_region(constraint_config: Mapping[str, Any], tags: Mapping[str, str]) -> str | None:
    if not bool_config(constraint_config.get("require_region_match"), False, coerce_strings=True):
        return None
    region_tag = string_or_none(constraint_config.get("region_tag", "region"))
    if region_tag is None:
        return None
    region = string_or_none(tags.get(region_tag))
    return region.lower() if region is not None else None


def _prepare_constraints(constraint_config: Mapping[str, Any], tags: Mapping[str, str]) -> _PreparedConstraints:
    return _PreparedConstraints(
        allowed_providers=_string_set(constraint_config.get("allowed_providers")),
        blocked_providers=_string_set(constraint_config.get("blocked_providers")),
        allowed_models=_model_constraint_set(constraint_config.get("allowed_models")),
        blocked_models=_model_constraint_set(constraint_config.get("blocked_models")),
        allowed_regions=_lower_string_set(constraint_config.get("allowed_regions")),
        blocked_regions=_lower_string_set(constraint_config.get("blocked_regions")),
        requested_region=_requested_region(constraint_config, tags),
        max_estimated_cost=non_negative_float_or_none(constraint_config.get("max_estimated_cost")),
        allow_unknown_cost=bool_config(
            constraint_config.get("allow_unknown_cost"),
            False,
            coerce_strings=True,
        ),
    )


def _estimated_cost_constraint_failure(
    estimated_cost: float | None,
    *,
    constraints: _PreparedConstraints,
) -> str | None:
    if constraints.max_estimated_cost is None:
        return None
    if estimated_cost is None:
        return None if constraints.allow_unknown_cost else "estimated_cost_unknown"
    if estimated_cost > constraints.max_estimated_cost:
        return "estimated_cost_exceeds_max"
    return None


def _constraint_rejection(candidate: Any, *, reason: str, candidate_regions: set[str]) -> dict[str, Any]:
    return {
        "model": candidate.model,
        "provider": candidate.provider,
        "reason": reason,
        "estimated_cost": candidate.estimated_cost,
        "regions": sorted(candidate_regions),
    }


def _constraint_failure(
    candidate: Any,
    *,
    candidate_regions: set[str],
    constraints: _PreparedConstraints,
) -> str | None:
    reason = _provider_model_constraint_failure(
        provider=candidate.provider,
        model=candidate.model,
        constraints=constraints,
    )
    if reason is not None:
        return reason
    reason = _region_constraint_failure(
        candidate_regions,
        constraints=constraints,
    )
    if reason is not None:
        return reason
    return _estimated_cost_constraint_failure(
        candidate.estimated_cost,
        constraints=constraints,
    )


def _constraint_rejection_for_candidate(candidate: Any, *, constraints: _PreparedConstraints) -> dict[str, Any] | None:
    metadata = dict_or_empty(candidate.metadata)
    candidate_regions = _candidate_regions(metadata)
    reason = _constraint_failure(
        candidate,
        candidate_regions=candidate_regions,
        constraints=constraints,
    )
    if reason is None:
        return None
    return _constraint_rejection(candidate, reason=reason, candidate_regions=candidate_regions)


def apply_constraints(
    candidates: Sequence[Any],
    *,
    config: Mapping[str, Any],
    tags: Mapping[str, str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    constraint_config = dict_or_empty(config.get("constraints"))
    if not constraint_config:
        return list(candidates), []

    constraints = _prepare_constraints(constraint_config, tags)
    allowed: list[Any] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        rejection = _constraint_rejection_for_candidate(candidate, constraints=constraints)
        if rejection is None:
            allowed.append(candidate)
            continue
        rejected.append(rejection)
    return allowed, rejected
