"""Shared pricing lookup utilities."""

from datetime import UTC, datetime

from any_llm import AnyLLM
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.models.entities import ModelPricing

_PRICE_UNIT = 1_000_000


def pricing_model_ref(provider: str | None, model: str) -> str:
    return f"{provider}:{model}" if provider else model


def legacy_pricing_model_ref(provider: str | None, model: str) -> str:
    return f"{provider}/{model}" if provider else model


def split_pricing_model_ref(model_ref: str) -> tuple[str, str]:
    provider, model = AnyLLM.split_model_provider(model_ref)
    return provider.value, model


def normalized_pricing_model_ref(model_ref: str) -> str:
    provider, model = split_pricing_model_ref(model_ref)
    return pricing_model_ref(provider, model)


def candidate_pricing_model_refs(model_ref: str) -> list[str]:
    """Return possible stored pricing keys for a provided model reference."""

    candidates = [model_ref]
    try:
        provider, model = split_pricing_model_ref(model_ref)
    except ValueError:
        return candidates

    for key in (pricing_model_ref(provider, model), legacy_pricing_model_ref(provider, model)):
        if key not in candidates:
            candidates.append(key)
    return candidates


def input_metered_cost(
    pricing: ModelPricing,
    *,
    units: float,
    price_divisor: float = _PRICE_UNIT,
) -> float:
    return (units / price_divisor) * pricing.input_price_per_million


def token_usage_cost(
    pricing: ModelPricing,
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    return input_metered_cost(pricing, units=prompt_tokens) + (
        completion_tokens / _PRICE_UNIT
    ) * pricing.output_price_per_million


def log_missing_pricing(provider: str | None, model: str) -> None:
    logger.warning(
        "No pricing configured for '%s'. Usage will be tracked without cost.",
        pricing_model_ref(provider, model),
    )


def normalize_effective_at(value: datetime | None) -> datetime:
    """Normalize a datetime to an aware UTC timestamp, defaulting to now."""

    normalized = value or datetime.now(UTC)
    if normalized.tzinfo is None:
        return normalized.replace(tzinfo=UTC)
    return normalized.astimezone(UTC)


async def find_model_pricing(
    db: AsyncSession,
    provider: str | None,
    model: str,
    *,
    as_of: datetime | None = None,
) -> ModelPricing | None:
    """Look up model pricing as of a timestamp, with legacy key fallback."""

    lookup_time = normalize_effective_at(as_of)
    model_keys = [pricing_model_ref(provider, model)]
    if provider:
        model_keys.append(legacy_pricing_model_ref(provider, model))

    for model_key in model_keys:
        stmt = (
            select(ModelPricing)
            .where(
                ModelPricing.model_key == model_key,
                ModelPricing.effective_at <= lookup_time,
            )
            .order_by(ModelPricing.effective_at.desc())
            .limit(1)
        )
        pricing = (await db.execute(stmt)).scalar_one_or_none()
        if pricing is not None:
            return pricing
    return None
