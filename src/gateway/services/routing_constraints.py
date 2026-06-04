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


def _normalize_model_key_for_constraint(value: str) -> str:
    try:
        _provider, _model_name, normalized = split_model_selector(value)
    except ValueError:
        return value
    return normalized


def _provider_model_constraint_failure(
    *,
    provider: str,
    model: str,
    allowed_providers: set[str],
    blocked_providers: set[str],
    allowed_models: set[str],
    blocked_models: set[str],
) -> str | None:
    if allowed_providers and provider not in allowed_providers:
        return "provider_not_allowed"
    if provider in blocked_providers:
        return "provider_blocked"
    if allowed_models and model not in allowed_models:
        return "model_not_allowed"
    if model in blocked_models:
        return "model_blocked"
    return None


def _region_constraint_failure(
    candidate_regions: set[str],
    *,
    allowed_regions: set[str],
    blocked_regions: set[str],
    requested_region: str | None,
) -> str | None:
    if allowed_regions:
        if not candidate_regions:
            return "region_unknown"
        if not candidate_regions & allowed_regions:
            return "region_not_allowed"
    if blocked_regions and candidate_regions & blocked_regions:
        return "region_blocked"
    if requested_region is None:
        return None
    if not candidate_regions:
        return "region_unknown"
    if requested_region not in candidate_regions:
        return "region_not_supported"
    return None


def _estimated_cost_constraint_failure(
    estimated_cost: float | None,
    *,
    max_estimated_cost: float | None,
    allow_unknown_cost: bool,
) -> str | None:
    if max_estimated_cost is None:
        return None
    if estimated_cost is None:
        return None if allow_unknown_cost else "estimated_cost_unknown"
    if estimated_cost > max_estimated_cost:
        return "estimated_cost_exceeds_max"
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

    allowed_providers = set(string_list(constraints.get("allowed_providers")))
    blocked_providers = set(string_list(constraints.get("blocked_providers")))
    allowed_models = {
        _normalize_model_key_for_constraint(item) for item in string_list(constraints.get("allowed_models"))
    }
    blocked_models = {
        _normalize_model_key_for_constraint(item) for item in string_list(constraints.get("blocked_models"))
    }
    allowed_regions = {item.lower() for item in string_list(constraints.get("allowed_regions"))}
    blocked_regions = {item.lower() for item in string_list(constraints.get("blocked_regions"))}
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
        candidate_regions = {item.lower() for item in string_list(metadata.get("regions"))}
        region = string_or_none(metadata.get("region"))
        if region is not None:
            candidate_regions.add(region.lower())
        reason = _provider_model_constraint_failure(
            provider=candidate.provider,
            model=candidate.model,
            allowed_providers=allowed_providers,
            blocked_providers=blocked_providers,
            allowed_models=allowed_models,
            blocked_models=blocked_models,
        )
        if reason is None:
            reason = _region_constraint_failure(
                candidate_regions,
                allowed_regions=allowed_regions,
                blocked_regions=blocked_regions,
                requested_region=requested_region,
            )
        if reason is None:
            reason = _estimated_cost_constraint_failure(
                candidate.estimated_cost,
                max_estimated_cost=max_estimated_cost,
                allow_unknown_cost=allow_unknown_cost,
            )
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
