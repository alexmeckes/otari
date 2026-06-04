from datetime import datetime

from pydantic import BaseModel, Field

from gateway.api.routes._response_datetime import datetime_isoformat
from gateway.models.entities import ModelPricing
from gateway.services.pricing_service import normalize_effective_at


class SetPricingRequest(BaseModel):
    """Request model for setting model pricing."""

    model_key: str = Field(description="Model identifier in format 'provider:model'")
    input_price_per_million: float = Field(ge=0, description="Price per 1M input tokens")
    output_price_per_million: float = Field(ge=0, description="Price per 1M output tokens")
    effective_at: datetime | None = Field(
        default=None,
        description="ISO 8601 datetime from which this price applies. Defaults to now if omitted.",
    )


class PricingResponse(BaseModel):
    """Response model for model pricing."""

    model_key: str
    effective_at: str
    input_price_per_million: float
    output_price_per_million: float
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, pricing: ModelPricing) -> "PricingResponse":
        """Create a PricingResponse from a ModelPricing ORM model."""
        return cls(
            model_key=pricing.model_key,
            effective_at=datetime_isoformat(normalize_effective_at(pricing.effective_at)),
            input_price_per_million=pricing.input_price_per_million,
            output_price_per_million=pricing.output_price_per_million,
            created_at=datetime_isoformat(pricing.created_at),
            updated_at=datetime_isoformat(pricing.updated_at),
        )
