from collections.abc import Iterable, Mapping
from typing import Any

from gateway.services.pricing_service import pricing_model_ref
from gateway.services.routing_config_values import dict_or_empty, float_or_none, score_or_none, string_or_none

DEFAULT_STRATEGY_TYPES = {"fallback", "intelligent", "weighted_score"}
INTELLIGENT_AXES = {"cost", "performance", "intelligence"}
AXIS_TIER_THRESHOLDS = {
    "cost": {"medium": 1600, "complex": 6000, "reasoning": 12000},
    "performance": {"medium": 800, "complex": 3000, "reasoning": 9000},
    "intelligence": {"medium": 300, "complex": 1400, "reasoning": 5000},
}

_DEFAULT_STRATEGY_PROVIDER_KEYS = {
    "provider",
    "model",
    "priority",
    "tier",
    "input_price_per_million",
    "output_price_per_million",
}
_DEFAULT_STRATEGY_CONFIG_KEYS = ("constraints", "health", "match", "tier_thresholds")
_WEIGHTED_SCORE_KEYS = {
    "weights",
    "quality_weight",
    "cost_weight",
    "latency_weight",
    "default_quality_score",
    "unknown_cost_score",
    "unknown_latency_score",
}


class RoutingPolicyShapeError(ValueError):
    """Raised when a public routing policy shape cannot be normalized."""


number_value = float_or_none
score_value = score_or_none


def model_selector(provider: str | None, model: str) -> str:
    model_value = string_or_none(model) or ""
    provider_value = string_or_none(provider)
    if provider_value is None:
        if "/" in model_value:
            return model_value.replace("/", ":", 1)
        return model_value
    if model_value.startswith(f"{provider_value}:") or model_value.startswith(f"{provider_value}/"):
        return model_value.replace("/", ":", 1)
    return pricing_model_ref(provider_value, model_value)


def normalized_model_selector(provider: str | None, model: str) -> str:
    return string_or_none(model_selector(provider, model).replace("/", ":", 1)) or ""


def split_model_selector(model_selector: str) -> tuple[str | None, str]:
    if ":" in model_selector:
        provider, model = model_selector.split(":", 1)
        return provider or None, model
    if "/" in model_selector:
        provider, model = model_selector.split("/", 1)
        return provider or None, model
    return None, model_selector


def _provider_priority(item: Mapping[str, Any], position: int) -> float:
    priority = number_value(item.get("priority"))
    return priority if priority is not None else float(position)


def _candidate_from_default_strategy_provider(item: Mapping[str, Any]) -> dict[str, Any]:
    provider = string_or_none(item.get("provider"))
    model = string_or_none(item.get("model"))
    if model is None:
        raise RoutingPolicyShapeError("default_strategy.providers entries must include a non-empty model")
    if "provider" in item and provider is None:
        raise RoutingPolicyShapeError("default_strategy.providers entries must include a non-empty provider when set")

    candidate: dict[str, Any] = {"model": model_selector(provider, model)}
    for key in ("tier", "input_price_per_million", "output_price_per_million"):
        if key in item:
            candidate[key] = item[key]

    metadata = {key: value for key, value in item.items() if key not in _DEFAULT_STRATEGY_PROVIDER_KEYS}
    if "priority" in item:
        metadata["priority"] = item["priority"]
    if metadata:
        candidate["metadata"] = metadata
    return candidate


def _candidates_from_default_strategy_providers(provider_items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_candidate_from_default_strategy_provider(provider_item) for provider_item in provider_items]


def _default_strategy_providers(default_strategy: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    providers = default_strategy.get("providers")
    if not isinstance(providers, list) or not providers:
        raise RoutingPolicyShapeError("default_strategy.providers must be a non-empty list")
    provider_items: list[Mapping[str, Any]] = []
    for item in providers:
        if not isinstance(item, dict):
            raise RoutingPolicyShapeError("default_strategy.providers entries must be objects")
        provider_items.append(item)
    return provider_items


def _copy_present_keys(source: Mapping[str, Any], target: dict[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        if key in source:
            target[key] = source[key]


def config_from_default_strategy(
    default_strategy: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    strategy_type_raw = default_strategy.get("type")
    strategy_type_value = string_or_none(strategy_type_raw)
    if strategy_type_value is None:
        raise RoutingPolicyShapeError("default_strategy.type is required")
    strategy_type = strategy_type_value.lower()
    if strategy_type not in DEFAULT_STRATEGY_TYPES:
        supported = ", ".join(sorted(DEFAULT_STRATEGY_TYPES))
        raise RoutingPolicyShapeError(
            f"Unsupported default_strategy.type '{strategy_type_raw}'. Supported types: {supported}"
        )

    provider_items = _default_strategy_providers(default_strategy)
    config = dict(base_config or {})
    _copy_present_keys(default_strategy, config, _DEFAULT_STRATEGY_CONFIG_KEYS)

    if strategy_type == "fallback":
        ordered_providers = sorted(
            enumerate(provider_items, start=1),
            key=lambda indexed: (_provider_priority(indexed[1], indexed[0]), indexed[0]),
        )
        config["candidates"] = _candidates_from_default_strategy_providers(
            provider_item for _, provider_item in ordered_providers
        )
        if "fallback_enabled" in default_strategy:
            config["fallback_enabled"] = bool(default_strategy["fallback_enabled"])
        strategy = "single" if len(provider_items) == 1 else "priority"
        return strategy, config

    if strategy_type == "weighted_score":
        config["candidates"] = _candidates_from_default_strategy_providers(provider_items)
        scoring_config = dict_or_empty(default_strategy.get("scoring"), copy_value=True)
        _copy_present_keys(default_strategy, scoring_config, _WEIGHTED_SCORE_KEYS)
        if scoring_config:
            config["scoring"] = scoring_config
        config.setdefault("fallback_enabled", True)
        return "weighted_score", config

    axis_raw = default_strategy.get("axis", "performance")
    axis_value = string_or_none(axis_raw)
    if axis_value is None:
        raise RoutingPolicyShapeError("default_strategy.axis must be a string")
    axis = axis_value.lower()
    if axis not in INTELLIGENT_AXES:
        supported = ", ".join(sorted(INTELLIGENT_AXES))
        raise RoutingPolicyShapeError(f"Unsupported default_strategy.axis '{axis_raw}'. Supported axes: {supported}")
    config["candidates"] = _candidates_from_default_strategy_providers(provider_items)
    config["axis"] = axis
    config.setdefault("fallback_enabled", True)
    config.setdefault("tier_thresholds", AXIS_TIER_THRESHOLDS[axis])
    return "intelligent", config


def _candidate_item_to_provider(item: Any, *, position: int, tier: str | None = None) -> dict[str, Any] | None:
    model_selector_value = string_or_none(item)
    if model_selector_value is not None:
        metadata: Mapping[str, Any] = {}
        input_price = None
        output_price = None
        candidate_tier = tier
    elif isinstance(item, dict):
        model_selector_value = string_or_none(item.get("model"))
        if model_selector_value is None:
            return None
        metadata = dict_or_empty(item.get("metadata"))
        input_price = item.get("input_price_per_million")
        output_price = item.get("output_price_per_million")
        candidate_tier = item.get("tier") if isinstance(item.get("tier"), str) else tier
    else:
        return None

    provider, model = split_model_selector(model_selector_value)
    provider_item: dict[str, Any] = {"provider": provider, "model": model, "priority": position}
    if candidate_tier is not None:
        provider_item["tier"] = candidate_tier
    if input_price is not None:
        provider_item["input_price_per_million"] = input_price
    if output_price is not None:
        provider_item["output_price_per_million"] = output_price
    for key, value in metadata.items():
        if key == "priority":
            provider_item["priority"] = value
        elif key not in provider_item:
            provider_item[key] = value
    return provider_item


def _providers_from_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    candidates = config.get("candidates")
    if isinstance(candidates, list):
        for item in candidates:
            provider_item = _candidate_item_to_provider(item, position=len(providers) + 1)
            if provider_item is not None:
                providers.append(provider_item)

    tiers = config.get("tiers")
    if isinstance(tiers, dict):
        for tier_name, items in tiers.items():
            if not isinstance(tier_name, str) or not isinstance(items, list):
                continue
            for item in items:
                provider_item = _candidate_item_to_provider(item, position=len(providers) + 1, tier=tier_name)
                if provider_item is not None:
                    providers.append(provider_item)
    return providers


def _default_strategy_fallback_enabled(
    strategy: str,
    config: Mapping[str, Any],
    *,
    default: bool = True,
) -> bool:
    if strategy == "single":
        return False
    return bool(config.get("fallback_enabled", default))


def default_strategy_from_internal(strategy: str, config: Mapping[str, Any]) -> dict[str, Any] | None:
    providers = _providers_from_config(config)
    if not providers:
        return None
    if strategy in {"single", "priority"}:
        return {
            "type": "fallback",
            "providers": providers,
            "fallback_enabled": _default_strategy_fallback_enabled(strategy, config),
        }
    if strategy == "intelligent":
        axis = config.get("axis")
        return {
            "type": "intelligent",
            "axis": axis if isinstance(axis, str) else "performance",
            "providers": providers,
            "fallback_enabled": _default_strategy_fallback_enabled(strategy, config),
        }
    if strategy == "weighted_score":
        response: dict[str, Any] = {
            "type": "weighted_score",
            "providers": providers,
            "fallback_enabled": _default_strategy_fallback_enabled(strategy, config),
        }
        scoring = config.get("scoring")
        if isinstance(scoring, dict):
            response["scoring"] = dict_or_empty(scoring, copy_value=True)
        return response
    return {
        "type": strategy,
        "providers": providers,
        "fallback_enabled": _default_strategy_fallback_enabled(strategy, config, default=strategy != "single"),
    }


def create_policy_shape(
    *,
    strategy: str,
    config: Mapping[str, Any],
    default_strategy: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if default_strategy is not None:
        return config_from_default_strategy(default_strategy, base_config=config)
    return strategy, dict(config)


def update_policy_shape(
    *,
    current_config: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    if "default_strategy" not in payload or payload["default_strategy"] is None:
        strategy = str(payload["strategy"]) if payload.get("strategy") is not None else None
        config = dict(payload["config"] or {}) if "config" in payload else None
        return strategy, config

    base_config = dict(payload["config"] or {}) if "config" in payload else dict(current_config)
    default_strategy = payload["default_strategy"]
    if not isinstance(default_strategy, dict):
        raise RoutingPolicyShapeError("default_strategy must be an object")
    return config_from_default_strategy(default_strategy, base_config=base_config)
