"""Response models for model catalog routes."""

from pydantic import BaseModel, Field


class ModelPricingInfo(BaseModel):
    """Pricing information for a model."""

    input_price_per_million: float
    output_price_per_million: float


class ModelObject(BaseModel):
    """OpenAI-compatible model object."""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    pricing: ModelPricingInfo | None = None


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: list[ModelObject]


class GatewayCatalogPricing(BaseModel):
    """Gateway-catalog pricing metadata for a vendor/model pair."""

    currency: str = "USD"
    input_per_million: float
    output_per_million: float


class GatewayCatalogCapabilities(BaseModel):
    """Best-effort capability metadata for a vendor/model pair."""

    input: list[str] = Field(default_factory=lambda: ["text"])
    output: list[str] = Field(default_factory=lambda: ["text", "tool_use"])
    supports_tool_calling: bool = True
    supports_tool_choice: bool = True
    supports_structured_outputs: bool = True
    streaming: bool = True


class GatewayVendorModelMetadata(BaseModel):
    """Execution metadata for a model served by one vendor."""

    launch_date: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    availability_status: str = "available"
    zero_data_retention: bool = False
    capabilities: GatewayCatalogCapabilities = Field(default_factory=GatewayCatalogCapabilities)
    pricing: GatewayCatalogPricing | None = None


class GatewayCatalogModel(BaseModel):
    """Canonical gateway-catalog model object."""

    model: str
    provider: str
    display_name: str
    vendors: dict[str, GatewayVendorModelMetadata]
    availability_status: str = "available"
    created_at: str | None = None
    updated_at: str | None = None


class GatewayCatalogListResponse(BaseModel):
    """Paginated gateway-catalog model response."""

    object: str = "list"
    data: list[GatewayCatalogModel]
    has_more: bool = False
    next_cursor: str | None = None


class GatewayVendorResponse(BaseModel):
    """Gateway-catalog execution vendor object."""

    vendor: str
    name: str
    models: list[str]
    supports_zdr: bool = False
    supports_byok: bool = True
    availability_status: str = "active"


class GatewayVendorListResponse(BaseModel):
    """Paginated gateway-catalog vendor response."""

    object: str = "list"
    data: list[GatewayVendorResponse]
    has_more: bool = False
    next_cursor: str | None = None
