from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gateway.services.routing_policy_service import (
    DEFAULT_ROUTING_MODEL,
    RoutingCandidate,
    RoutingPlan,
    require_routing_model_selector,
)


class ResolveRoutingRequest(BaseModel):
    """Request model for previewing one routing decision."""

    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_ROUTING_MODEL
    messages: list[dict[str, Any]] = Field(min_length=1)
    policy_id: str | None = None
    project_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    tools: list[dict[str, Any]] | None = None

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, v: Any) -> str:
        """Accept omitted/null/case-insensitive default_routing sentinels."""
        return require_routing_model_selector(v)


class ResolvedRoutingCandidateResponse(BaseModel):
    """Candidate returned from a dry-run routing resolution."""

    model: str
    provider: str
    provider_model: str
    position: int
    tier: str | None
    estimated_cost: float | None
    input_price_per_million: float | None
    output_price_per_million: float | None
    quality_score: float | None
    average_latency_ms: float | None
    latency_sample_count: int
    routing_score: float | None
    score_components: dict[str, float] | None
    provider_health: dict[str, Any] | None
    metadata: dict[str, Any]

    @classmethod
    def from_candidate(cls, candidate: RoutingCandidate) -> "ResolvedRoutingCandidateResponse":
        """Create a response from a routing candidate."""
        return cls(**candidate.to_trace_dict())


class RejectedRoutingCandidateResponse(BaseModel):
    """Candidate rejected by routing constraints during dry-run resolution."""

    model: str
    provider: str
    reason: str
    estimated_cost: float | None
    regions: list[str] = Field(default_factory=list)
    provider_health: dict[str, Any] | None = None


class ResolveRoutingResponse(BaseModel):
    """Dry-run routing decision response."""

    requested_model: str
    selected_model: str
    selected_provider: str
    selected_provider_model: str
    project_id: str | None
    policy_id: str
    policy_name: str
    policy_status: str
    policy_source: str
    policy_rollout: dict[str, Any] | None
    strategy: str
    target_tier: str
    prompt_tokens: int
    output_tokens: int
    estimated_cost: float | None
    fallback_enabled: bool
    reason: str
    tags: dict[str, str]
    guardrails: dict[str, Any] | None
    context: dict[str, Any] | None
    candidates: list[ResolvedRoutingCandidateResponse]
    rejected_candidates: list[RejectedRoutingCandidateResponse]

    @classmethod
    def from_plan(cls, plan: RoutingPlan) -> "ResolveRoutingResponse":
        """Create a response from a resolved routing plan."""
        selected = plan.selected_candidate
        return cls(
            requested_model=plan.requested_model,
            selected_model=selected.model,
            selected_provider=selected.provider,
            selected_provider_model=selected.provider_model,
            project_id=plan.project_id,
            policy_id=plan.policy.policy_id,
            policy_name=plan.policy.name,
            policy_status=plan.policy.status,
            policy_source=plan.policy_source,
            policy_rollout=plan.policy_rollout,
            strategy=plan.strategy,
            target_tier=plan.target_tier,
            prompt_tokens=plan.prompt_tokens,
            output_tokens=plan.output_tokens,
            estimated_cost=selected.estimated_cost,
            fallback_enabled=plan.fallback_enabled,
            reason=plan.reason,
            tags=plan.tags,
            guardrails=plan.guardrails,
            context=plan.context,
            candidates=[
                ResolvedRoutingCandidateResponse.from_candidate(candidate)
                for candidate in plan.candidates
            ],
            rejected_candidates=[
                RejectedRoutingCandidateResponse(**candidate)
                for candidate in plan.rejected_candidates
            ],
        )
