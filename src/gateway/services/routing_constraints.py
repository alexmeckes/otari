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
        reason = None
        if allowed_providers and candidate.provider not in allowed_providers:
            reason = "provider_not_allowed"
        elif candidate.provider in blocked_providers:
            reason = "provider_blocked"
        elif allowed_models and candidate.model not in allowed_models:
            reason = "model_not_allowed"
        elif candidate.model in blocked_models:
            reason = "model_blocked"
        if reason is None:
            if allowed_regions:
                if not candidate_regions:
                    reason = "region_unknown"
                elif not candidate_regions & allowed_regions:
                    reason = "region_not_allowed"
            if reason is None and blocked_regions and candidate_regions & blocked_regions:
                reason = "region_blocked"
            if reason is None and requested_region is not None:
                if not candidate_regions:
                    reason = "region_unknown"
                elif requested_region not in candidate_regions:
                    reason = "region_not_supported"
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
