"""Helpers and response models for model catalog routes."""

import calendar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._model_catalog_models import (
    GatewayCatalogModel,
    GatewayCatalogPricing,
    GatewayVendorModelMetadata,
    GatewayVendorResponse,
    ModelObject,
    ModelPricingInfo,
)
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ModelPricing
from gateway.services.model_discovery_service import discover_all_models, get_model_cache

T = TypeVar("T")


@dataclass(frozen=True)
class CatalogRecord:
    model_key: str
    provider: str
    provider_model: str
    created: int
    created_at: str | None
    updated_at: str | None
    pricing: ModelPricingInfo | None


_VENDOR_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "azureopenai": "Azure OpenAI",
    "bedrock": "AWS Bedrock",
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "vertexai": "Google Vertex AI",
    "vertexaianthropic": "Vertex AI Anthropic",
}


def _split_model_key(model_key: str) -> tuple[str, str]:
    """Split a gateway model selector into provider/model parts."""
    if ":" in model_key:
        provider, model_name = model_key.split(":", 1)
        return provider or "unknown", model_name
    if "/" in model_key:
        provider, model_name = model_key.split("/", 1)
        return provider or "unknown", model_name
    return "unknown", model_key


def _canonical_model_id(model_key: str) -> str:
    """Return a provider/model ID for a stored model key."""
    provider, model_name = _split_model_key(model_key)
    return f"{provider}/{model_name}"


def _stored_model_key(selector: str) -> str:
    """Return the canonical storage key for either provider:model or provider/model."""
    provider, model_name = _split_model_key(selector)
    return f"{provider}:{model_name}"


def _model_pricing_info(pricing: ModelPricing | None) -> ModelPricingInfo | None:
    if pricing is None:
        return None
    return ModelPricingInfo(
        input_price_per_million=pricing.input_price_per_million,
        output_price_per_million=pricing.output_price_per_million,
    )


def _created_epoch(value: datetime | None) -> int:
    if value is None:
        return 0
    return int(calendar.timegm(value.utctimetuple()))


def _iso_from_epoch(epoch_seconds: int | None) -> str | None:
    if epoch_seconds is None or epoch_seconds <= 0:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat().replace("+00:00", "Z")


def _display_name(model_key: str) -> str:
    _provider, model_name = _split_model_key(model_key)
    words = model_name.replace("/", " / ").replace("-", " ").replace("_", " ").split()
    fixed: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in {"gpt", "llm", "glm", "mcp"}:
            fixed.append(lower.upper())
        elif lower.endswith("o") and any(char.isdigit() for char in lower):
            fixed.append(lower)
        else:
            fixed.append(word[:1].upper() + word[1:])
    return " ".join(fixed)


def _vendor_name(vendor: str) -> str:
    return _VENDOR_DISPLAY_NAMES.get(vendor, vendor.replace("_", " ").replace("-", " ").title())


def model_from_catalog_record(record: CatalogRecord) -> ModelObject:
    """Convert an internal catalog record to an OpenAI-compatible ModelObject."""
    return ModelObject(
        id=record.model_key,
        created=record.created,
        owned_by=record.provider,
        pricing=record.pricing,
    )


def gateway_model_from_catalog_record(record: CatalogRecord) -> GatewayCatalogModel:
    """Convert an internal catalog record to a gateway-catalog model object."""
    pricing = None
    if record.pricing is not None:
        pricing = GatewayCatalogPricing(
            input_per_million=record.pricing.input_price_per_million,
            output_per_million=record.pricing.output_price_per_million,
        )
    return GatewayCatalogModel(
        model=_canonical_model_id(record.model_key),
        provider=record.provider,
        display_name=_display_name(record.model_key),
        vendors={
            record.provider: GatewayVendorModelMetadata(
                launch_date=record.created_at[:10] if record.created_at else None,
                pricing=pricing,
            )
        },
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


async def _get_pricing_map(db: AsyncSession, provider_filter: str | None = None) -> dict[str, ModelPricing]:
    """Load latest pricing per model_key, optionally filtered by provider prefix."""
    latest_effective = (
        select(
            ModelPricing.model_key.label("model_key"),
            func.max(ModelPricing.effective_at).label("effective_at"),
        )
        .group_by(ModelPricing.model_key)
        .subquery()
    )

    stmt = select(ModelPricing).join(
        latest_effective,
        (ModelPricing.model_key == latest_effective.c.model_key)
        & (ModelPricing.effective_at == latest_effective.c.effective_at),
    )

    if provider_filter:
        stmt = stmt.where(ModelPricing.model_key.startswith(f"{provider_filter}:"))

    stmt = stmt.order_by(ModelPricing.model_key)
    result = await db.execute(stmt)
    pricings = result.scalars().all()
    return {p.model_key: p for p in pricings}


async def load_catalog_records(
    db: AsyncSession,
    config: GatewayConfig,
    provider_filter: str | None = None,
) -> list[CatalogRecord]:
    """Load latest models from discovery and pricing into one internal catalog."""
    pricing_map = await _get_pricing_map(db, provider_filter=provider_filter)
    records: dict[str, CatalogRecord] = {}

    if config.model_discovery:
        try:
            discovered = await discover_all_models(config, provider_filter=provider_filter)
        except Exception:
            logger.exception("Model discovery failed unexpectedly")
            discovered = []

        for provider_name, model in discovered:
            model_key = f"{provider_name}:{model.id}"
            pricing = pricing_map.pop(model_key, None)
            created_at = _iso_from_epoch(model.created)
            records[model_key] = CatalogRecord(
                model_key=model_key,
                provider=provider_name,
                provider_model=model.id,
                created=model.created,
                created_at=created_at,
                updated_at=pricing.updated_at.isoformat() if pricing else created_at,
                pricing=_model_pricing_info(pricing),
            )

    for model_key, pricing in pricing_map.items():
        if model_key in records:
            continue
        provider, provider_model = _split_model_key(model_key)
        records[model_key] = CatalogRecord(
            model_key=model_key,
            provider=provider,
            provider_model=provider_model,
            created=_created_epoch(pricing.created_at),
            created_at=pricing.created_at.isoformat() if pricing.created_at else None,
            updated_at=pricing.updated_at.isoformat() if pricing.updated_at else None,
            pricing=_model_pricing_info(pricing),
        )

    return sorted(records.values(), key=lambda record: record.model_key)


def paginate(items: list[T], *, cursor: str | None, limit: int) -> tuple[list[T], bool, str | None]:
    """Apply offset-cursor pagination to catalog responses."""
    try:
        offset = int(cursor or "0")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="cursor must be an integer offset",
        ) from None
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="cursor must be non-negative",
        )
    page = items[offset : offset + limit]
    next_offset = offset + limit
    has_more = next_offset < len(items)
    return page, has_more, str(next_offset) if has_more else None


def filter_catalog_records(
    records: list[CatalogRecord],
    *,
    provider: str | None,
    vendor: str | None,
    model: str | None,
) -> list[CatalogRecord]:
    """Filter catalog records using provider/vendor/model parameters."""
    selected = records
    if provider:
        selected = [record for record in selected if record.provider == provider]
    if vendor:
        selected = [record for record in selected if record.provider == vendor]
    if model:
        stored_model_key = _stored_model_key(model)
        selected = [record for record in selected if record.model_key == stored_model_key]
    return selected


def vendor_catalog_from_records(records: list[CatalogRecord], config: GatewayConfig) -> list[GatewayVendorResponse]:
    """Build execution-vendor objects from catalog records."""
    by_vendor: dict[str, set[str]] = {}
    for record in records:
        by_vendor.setdefault(record.provider, set()).add(_canonical_model_id(record.model_key))

    for provider_name in config.providers:
        by_vendor.setdefault(provider_name, set())

    vendors = [
        GatewayVendorResponse(
            vendor=vendor,
            name=_vendor_name(vendor),
            models=sorted(models),
            supports_byok=vendor in config.providers,
            availability_status="active" if models or vendor in config.providers else "unavailable",
        )
        for vendor, models in by_vendor.items()
    ]
    return sorted(vendors, key=lambda item: item.vendor)


async def load_model_object(db: AsyncSession, config: GatewayConfig, model_id: str) -> ModelObject:
    """Load one OpenAI-compatible model object from pricing or discovery cache."""
    normalized_model_id = _stored_model_key(model_id)
    stmt = (
        select(ModelPricing)
        .where(ModelPricing.model_key == normalized_model_id)
        .order_by(ModelPricing.effective_at.desc())
        .limit(1)
    )
    pricing = (await db.execute(stmt)).scalar_one_or_none()

    discovered_model = None
    discovered_provider = None
    if config.model_discovery and ":" in normalized_model_id:
        provider_prefix, model_name = normalized_model_id.split(":", 1)
        cache = get_model_cache()
        ttl = config.model_cache_ttl_seconds
        cached_models = cache.get(provider_prefix, ttl)
        if cached_models is not None:
            for model in cached_models:
                if model.id == model_name:
                    discovered_model = model
                    discovered_provider = provider_prefix
                    break

    if not pricing and not discovered_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found",
        )

    if discovered_model:
        assert discovered_provider is not None
        model_key = f"{discovered_provider}:{discovered_model.id}"
        return ModelObject(
            id=model_key,
            created=discovered_model.created,
            owned_by=discovered_provider,
            pricing=_model_pricing_info(pricing),
        )

    assert pricing is not None
    provider, provider_model = _split_model_key(pricing.model_key)
    return model_from_catalog_record(
        CatalogRecord(
            model_key=pricing.model_key,
            provider=provider,
            provider_model=provider_model,
            created=_created_epoch(pricing.created_at),
            created_at=pricing.created_at.isoformat() if pricing.created_at else None,
            updated_at=pricing.updated_at.isoformat() if pricing.updated_at else None,
            pricing=_model_pricing_info(pricing),
        )
    )
