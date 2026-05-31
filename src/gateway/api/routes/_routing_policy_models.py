from typing import Any

from pydantic import BaseModel, Field

from gateway.models.entities import RoutingPolicy, RoutingPolicyRevision
from gateway.services import routing_policy_eval_scores, routing_policy_shape
from gateway.services.routing_policy_service import ACTIVE_ROUTING_POLICY_STATUS


class CreateRoutingPolicyRequest(BaseModel):
    """Request model for creating a routing policy."""

    name: str = Field(min_length=1)
    strategy: str = Field(default="cost_tier")
    config: dict[str, Any] = Field(default_factory=dict)
    default_strategy: dict[str, Any] | None = None
    is_default: bool = False
    status: str = ACTIVE_ROUTING_POLICY_STATUS
    change_note: str | None = None


class CloneRoutingPolicyRequest(BaseModel):
    """Request model for cloning a routing policy into a draft."""

    name: str | None = Field(default=None, min_length=1)
    change_note: str | None = None


class ApplyRoutingPolicyRevisionRequest(BaseModel):
    """Request model for applying a previous routing policy revision."""

    change_note: str | None = None


class RoutingPolicyEvalScoreItem(BaseModel):
    """One uploaded eval or benchmark score for a candidate model."""

    model: str = Field(min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    score: float | None = Field(default=None, ge=0, le=100)
    quality_score: float | None = Field(default=None, ge=0, le=100)
    benchmark_score: float | None = Field(default=None, ge=0, le=100)
    metric: str | None = Field(default=None, min_length=1)
    sample_count: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApplyRoutingPolicyEvalScoresRequest(BaseModel):
    """Request model for applying eval scores to weighted routing candidates."""

    scores: list[RoutingPolicyEvalScoreItem] = Field(min_length=1)
    change_note: str | None = None


class UpdateRoutingPolicyRequest(BaseModel):
    """Request model for updating a routing policy."""

    name: str | None = Field(default=None, min_length=1)
    strategy: str | None = None
    config: dict[str, Any] | None = None
    default_strategy: dict[str, Any] | None = None
    is_default: bool | None = None
    status: str | None = None
    change_note: str | None = None


class RoutingPolicyResponse(BaseModel):
    """Response model for routing policy information."""

    policy_id: str
    name: str
    strategy: str
    config: dict[str, Any]
    default_strategy: dict[str, Any] | None
    is_default: bool
    revision: int
    status: str
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, policy: RoutingPolicy) -> "RoutingPolicyResponse":
        """Create a response from an ORM model."""
        config = policy.config_dict()
        return cls(
            policy_id=policy.policy_id,
            name=policy.name,
            strategy=policy.strategy,
            config=config,
            default_strategy=routing_policy_shape.default_strategy_from_internal(policy.strategy, config),
            is_default=bool(policy.is_default),
            revision=int(policy.revision or 0),
            status=policy.status,
            created_at=policy.created_at.isoformat(),
            updated_at=policy.updated_at.isoformat(),
        )


class RoutingPolicyRevisionResponse(BaseModel):
    """Response model for routing policy revision history."""

    revision_id: str
    policy_id: str
    revision: int
    action: str
    name: str
    strategy: str
    config: dict[str, Any]
    default_strategy: dict[str, Any] | None
    is_default: bool
    status: str
    change_note: str | None
    created_at: str

    @classmethod
    def from_model(cls, revision: RoutingPolicyRevision) -> "RoutingPolicyRevisionResponse":
        """Create a response from an ORM model."""
        config = revision.config_dict()
        return cls(
            revision_id=revision.revision_id,
            policy_id=revision.policy_id,
            revision=revision.revision,
            action=revision.action,
            name=revision.name,
            strategy=revision.strategy,
            config=config,
            default_strategy=routing_policy_shape.default_strategy_from_internal(revision.strategy, config),
            is_default=bool(revision.is_default),
            status=revision.status,
            change_note=revision.change_note,
            created_at=revision.created_at.isoformat(),
        )


class AppliedRoutingPolicyEvalScoreResponse(BaseModel):
    """Candidate score update returned after eval ingestion."""

    model: str
    previous_quality_score: float | None
    quality_score: float
    sample_count: int
    metrics: list[str]


class ApplyRoutingPolicyEvalScoresResponse(BaseModel):
    """Response model for eval score ingestion."""

    policy: RoutingPolicyResponse
    applied_count: int
    unmatched_models: list[str]
    applied_scores: list[AppliedRoutingPolicyEvalScoreResponse]


def eval_score_input(item: RoutingPolicyEvalScoreItem) -> routing_policy_eval_scores.EvalScoreInput:
    return routing_policy_eval_scores.EvalScoreInput(
        model=item.model,
        provider=item.provider,
        score=item.score,
        quality_score=item.quality_score,
        benchmark_score=item.benchmark_score,
        metric=item.metric,
        sample_count=item.sample_count,
        metadata=dict(item.metadata),
    )
