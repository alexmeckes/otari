from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from any_llm import AnyLLM

from gateway.services.routing_config_values import dict_or_empty, float_or_none, score_or_none

TIER_ORDER = ("simple", "medium", "complex", "reasoning")
_INFERRED_TIER_BY_OUTPUT_PRICE = (
    (0.10, "simple"),
    (1.50, "simple"),
    (2.00, "medium"),
    (5.00, "complex"),
)


@dataclass(frozen=True)
class CandidateSpec:
    model: str
    tier: str | None
    input_price_per_million: float | None
    output_price_per_million: float | None
    quality_score: float | None
    metadata: dict[str, Any]


def _normalize_tier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in TIER_ORDER else None


def _candidate_spec_from_item(item: Any, *, tier: str | None) -> CandidateSpec | None:
    if isinstance(item, str):
        model = item.strip()
        if not model:
            return None
        return CandidateSpec(
            model=model,
            tier=tier,
            input_price_per_million=None,
            output_price_per_million=None,
            quality_score=None,
            metadata={},
        )

    if not isinstance(item, dict):
        return None
    model_value = item.get("model")
    if not isinstance(model_value, str) or not model_value.strip():
        return None

    metadata = dict_or_empty(item.get("metadata"), copy_value=True)
    for key in ("region", "regions"):
        if key in item and key not in metadata:
            metadata[key] = item[key]
    quality_score = score_or_none(item.get("quality_score"))
    if quality_score is None:
        for key in ("benchmark_score", "score", "intelligence_score"):
            quality_score = score_or_none(item.get(key))
            if quality_score is not None:
                break
    if quality_score is None:
        for key in ("quality_score", "benchmark_score", "score", "intelligence_score"):
            quality_score = score_or_none(metadata.get(key))
            if quality_score is not None:
                break
    return CandidateSpec(
        model=model_value.strip(),
        tier=_normalize_tier(item.get("tier")) or tier,
        input_price_per_million=float_or_none(item.get("input_price_per_million")),
        output_price_per_million=float_or_none(item.get("output_price_per_million")),
        quality_score=quality_score,
        metadata=metadata,
    )


def infer_tier_from_output_price(output_price_per_million: float | None) -> str | None:
    """Infer an internal complexity tier from output-token pricing."""
    if output_price_per_million is None:
        return None
    for max_output_price, tier in _INFERRED_TIER_BY_OUTPUT_PRICE:
        if output_price_per_million < max_output_price:
            return tier
    return "reasoning"


def split_model_selector(model_selector: str) -> tuple[str, str, str]:
    """Split and normalize a provider:model selector."""
    provider, model_name = AnyLLM.split_model_provider(model_selector)
    provider_name = provider.value
    return provider_name, model_name, f"{provider_name}:{model_name}"


def configured_candidate_specs(config: Mapping[str, Any]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []

    candidates = config.get("candidates")
    if isinstance(candidates, list):
        for item in candidates:
            spec = _candidate_spec_from_item(item, tier=None)
            if spec is not None:
                specs.append(spec)

    tiers = config.get("tiers")
    if isinstance(tiers, dict):
        for tier_name, items in tiers.items():
            tier = _normalize_tier(tier_name)
            if tier is None or not isinstance(items, list):
                continue
            for item in items:
                spec = _candidate_spec_from_item(item, tier=tier)
                if spec is not None:
                    specs.append(spec)

    deduped: list[CandidateSpec] = []
    seen: set[str] = set()
    for spec in specs:
        provider, provider_model, normalized = split_model_selector(spec.model)
        dedupe_key = f"{provider}:{provider_model}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(
            CandidateSpec(
                model=normalized,
                tier=spec.tier,
                input_price_per_million=spec.input_price_per_million,
                output_price_per_million=spec.output_price_per_million,
                quality_score=spec.quality_score,
                metadata=spec.metadata,
            )
        )
    return deduped
