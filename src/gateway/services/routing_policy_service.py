"""Database-backed routing policy resolution for standalone gateway mode."""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from any_llm import AnyLLM
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.models.entities import Project, RouteTrace, RoutingPolicy
from gateway.services import routing_constraints as _routing_constraints
from gateway.services import routing_guardrails as _routing_guardrails
from gateway.services import routing_policy_match as _routing_policy_match
from gateway.services import routing_request_analysis as _routing_request_analysis
from gateway.services import routing_weighted_scoring as _routing_weighted_scoring
from gateway.services.pricing_service import find_model_pricing
from gateway.services.routing_context_policy import apply_context_policy as apply_context_policy

DEFAULT_ROUTING_MODEL = "default_routing"
DEFAULT_ROUTE_TRACE_ENDPOINT = "/v1/chat/completions"
ROUTING_STRATEGIES = {
    "single",
    "priority",
    "lowest_cost",
    "least_latency",
    "cost_tier",
    "intelligent",
    "weighted_score",
}
ACTIVE_ROUTING_POLICY_STATUS = "active"

_TIER_ORDER = ("simple", "medium", "complex", "reasoning")
_INFERRED_TIER_BY_OUTPUT_PRICE = (
    (0.10, "simple"),
    (1.50, "simple"),
    (2.00, "medium"),
    (5.00, "complex"),
)
_HEALTH_MODES = {"observe", "downrank", "skip_unhealthy"}
_HEALTH_RANK = {"healthy": 0, "unknown": 1, "degraded": 2, "unhealthy": 3}
_bool_config = _routing_request_analysis.bool_config
_int_config = _routing_request_analysis.int_config
_jsonable_text = _routing_request_analysis.jsonable_text
classify_request_tier = _routing_request_analysis.classify_request_tier
estimate_output_tokens = _routing_request_analysis.estimate_output_tokens
estimate_prompt_tokens = _routing_request_analysis.estimate_prompt_tokens
_guardrail_action = _routing_guardrails.guardrail_action
apply_guardrail_redactions = _routing_guardrails.apply_guardrail_redactions


def normalize_routing_model_selector(model: Any) -> str:
    """Normalize default-routing sentinels while preserving direct model selectors."""
    if model is None:
        return DEFAULT_ROUTING_MODEL
    if not isinstance(model, str):
        return str(model)
    normalized = model.strip()
    if normalized.lower() == DEFAULT_ROUTING_MODEL:
        return DEFAULT_ROUTING_MODEL
    return normalized


class RoutingPolicyError(Exception):
    """Error that can be surfaced as an HTTP routing-policy failure."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class _CandidateSpec:
    model: str
    tier: str | None
    input_price_per_million: float | None
    output_price_per_million: float | None
    quality_score: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProviderHealth:
    """Passive provider health summary derived from recent route traces."""

    provider: str
    status: str
    sample_count: int
    success_count: int
    error_count: int
    failure_rate: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe health representation."""
        return {
            "provider": self.provider,
            "status": self.status,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "failure_rate": self.failure_rate,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RoutingCandidate:
    """Resolved candidate model with pricing and ordering metadata."""

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
    provider_health: ProviderHealth | None
    metadata: dict[str, Any]

    def to_trace_dict(self) -> dict[str, Any]:
        """Return the JSON-safe trace representation."""
        return {
            "model": self.model,
            "provider": self.provider,
            "provider_model": self.provider_model,
            "position": self.position,
            "tier": self.tier,
            "estimated_cost": self.estimated_cost,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "quality_score": self.quality_score,
            "average_latency_ms": self.average_latency_ms,
            "latency_sample_count": self.latency_sample_count,
            "routing_score": self.routing_score,
            "score_components": self.score_components,
            "provider_health": self.provider_health.to_dict() if self.provider_health else None,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RoutingPlan:
    """Resolved model-routing plan for one chat completion request."""

    requested_model: str
    project: Project | None
    policy: RoutingPolicy
    strategy: str
    target_tier: str
    prompt_tokens: int
    output_tokens: int
    candidates: list[RoutingCandidate]
    fallback_enabled: bool
    policy_source: str
    reason: str
    tags: dict[str, str]
    rejected_candidates: list[dict[str, Any]]
    policy_rollout: dict[str, Any] | None
    guardrails: dict[str, Any] | None
    context: dict[str, Any] | None

    @property
    def selected_candidate(self) -> RoutingCandidate:
        """Return the first candidate the gateway should try."""
        return self.candidates[0]

    @property
    def project_id(self) -> str | None:
        """Return the project id attached to the request."""
        return self.project.project_id if self.project is not None else None


@dataclass(frozen=True)
class _PolicyMatch:
    policy: RoutingPolicy
    source: str
    rollout: dict[str, Any] | None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _non_negative_float_or_none(value: Any) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _score_or_none(value: Any) -> float | None:
    parsed = _non_negative_float_or_none(value)
    if parsed is None:
        return None
    if parsed <= 1.0:
        return parsed
    if parsed <= 100.0:
        return parsed / 100.0
    return None


def _normalize_tier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _TIER_ORDER else None


def _candidate_spec_from_item(item: Any, *, tier: str | None) -> _CandidateSpec | None:
    if isinstance(item, str):
        model = item.strip()
        if not model:
            return None
        return _CandidateSpec(
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

    metadata_value = item.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
    for key in ("region", "regions"):
        if key in item and key not in metadata:
            metadata[key] = item[key]
    quality_score = _score_or_none(item.get("quality_score"))
    if quality_score is None:
        for key in ("benchmark_score", "score", "intelligence_score"):
            quality_score = _score_or_none(item.get(key))
            if quality_score is not None:
                break
    if quality_score is None:
        for key in ("quality_score", "benchmark_score", "score", "intelligence_score"):
            quality_score = _score_or_none(metadata.get(key))
            if quality_score is not None:
                break
    return _CandidateSpec(
        model=model_value.strip(),
        tier=_normalize_tier(item.get("tier")) or tier,
        input_price_per_million=_float_or_none(item.get("input_price_per_million")),
        output_price_per_million=_float_or_none(item.get("output_price_per_million")),
        quality_score=quality_score,
        metadata=metadata,
    )


def _infer_tier_from_output_price(output_price_per_million: float | None) -> str | None:
    """Infer an internal complexity tier from output-token pricing."""
    if output_price_per_million is None:
        return None
    for max_output_price, tier in _INFERRED_TIER_BY_OUTPUT_PRICE:
        if output_price_per_million < max_output_price:
            return tier
    return "reasoning"


def _configured_candidate_specs(config: Mapping[str, Any]) -> list[_CandidateSpec]:
    specs: list[_CandidateSpec] = []

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

    deduped: list[_CandidateSpec] = []
    seen: set[str] = set()
    for spec in specs:
        provider, provider_model, normalized = split_model_selector(spec.model)
        dedupe_key = f"{provider}:{provider_model}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(
            _CandidateSpec(
                model=normalized,
                tier=spec.tier,
                input_price_per_million=spec.input_price_per_million,
                output_price_per_million=spec.output_price_per_million,
                quality_score=spec.quality_score,
                metadata=spec.metadata,
            )
        )
    return deduped


def split_model_selector(model_selector: str) -> tuple[str, str, str]:
    """Split and normalize a provider:model selector."""
    provider, model_name = AnyLLM.split_model_provider(model_selector)
    provider_name = provider.value
    return provider_name, model_name, f"{provider_name}:{model_name}"


async def _build_candidates(
    db: AsyncSession,
    specs: Sequence[_CandidateSpec],
    *,
    prompt_tokens: int,
    output_tokens: int,
) -> list[RoutingCandidate]:
    candidates: list[RoutingCandidate] = []
    for index, spec in enumerate(specs, start=1):
        provider, provider_model, normalized_model = split_model_selector(spec.model)
        pricing = await find_model_pricing(db, provider, provider_model)
        input_price = spec.input_price_per_million
        output_price = spec.output_price_per_million
        if pricing is not None:
            if input_price is None:
                input_price = pricing.input_price_per_million
            if output_price is None:
                output_price = pricing.output_price_per_million

        estimated_cost: float | None = None
        if input_price is not None and output_price is not None:
            estimated_cost = (prompt_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price

        candidates.append(
            RoutingCandidate(
                model=normalized_model,
                provider=provider,
                provider_model=provider_model,
                position=index,
                tier=spec.tier or _infer_tier_from_output_price(output_price),
                estimated_cost=estimated_cost,
                input_price_per_million=input_price,
                output_price_per_million=output_price,
                quality_score=spec.quality_score,
                average_latency_ms=None,
                latency_sample_count=0,
                routing_score=None,
                score_components=None,
                provider_health=None,
                metadata=spec.metadata,
            )
        )
    return candidates


def _by_cost(candidate: RoutingCandidate) -> tuple[bool, float, int]:
    return (candidate.estimated_cost is None, candidate.estimated_cost or 0.0, candidate.position)


def _by_latency(candidate: RoutingCandidate) -> tuple[bool, float, int]:
    return (candidate.average_latency_ms is None, candidate.average_latency_ms or 0.0, candidate.position)


def _by_weighted_score(candidate: RoutingCandidate) -> tuple[bool, float, int]:
    return (candidate.routing_score is None, -(candidate.routing_score or 0.0), candidate.position)


def _by_health(candidate: RoutingCandidate) -> tuple[int, int]:
    status_value = candidate.provider_health.status if candidate.provider_health else "unknown"
    return (_HEALTH_RANK.get(status_value, _HEALTH_RANK["unknown"]), candidate.position)


def _tier_fallback_order(target_tier: str) -> list[str]:
    target_index = _TIER_ORDER.index(target_tier)
    return [*_TIER_ORDER[target_index:], *reversed(_TIER_ORDER[:target_index])]


def _order_candidates(
    candidates: Sequence[RoutingCandidate],
    *,
    strategy: str,
    target_tier: str,
) -> list[RoutingCandidate]:
    if strategy in {"single", "priority"}:
        return list(candidates)
    if strategy == "lowest_cost":
        return sorted(candidates, key=_by_cost)
    if strategy == "least_latency":
        return sorted(candidates, key=_by_latency)
    if strategy == "weighted_score":
        return sorted(candidates, key=_by_weighted_score)

    ordered: list[RoutingCandidate] = []
    consumed: set[str] = set()
    for tier in _tier_fallback_order(target_tier):
        tier_candidates = [candidate for candidate in candidates if candidate.tier == tier]
        for candidate in sorted(tier_candidates, key=_by_cost):
            ordered.append(candidate)
            consumed.add(candidate.model)

    remaining = [candidate for candidate in candidates if candidate.model not in consumed]
    ordered.extend(sorted(remaining, key=_by_cost))
    return ordered


def _fallback_enabled(config: Mapping[str, Any], *, strategy: str) -> bool:
    if strategy == "single":
        return False
    return _bool_config(config.get("fallback_enabled"), True)


def _attempt_model_key(attempt: Mapping[str, Any]) -> str | None:
    model_key = attempt.get("model_key")
    if isinstance(model_key, str) and model_key:
        return model_key

    provider = attempt.get("provider")
    model = attempt.get("model")
    if isinstance(provider, str) and provider and isinstance(model, str) and model:
        return f"{provider}:{model}"
    return None


def _attempt_duration_ms(attempt: Mapping[str, Any]) -> float | None:
    duration = attempt.get("duration_ms")
    if isinstance(duration, bool):
        return None
    if isinstance(duration, int | float) and duration >= 0:
        return float(duration)
    return None


def _attempt_provider(attempt: Mapping[str, Any]) -> str | None:
    provider = attempt.get("provider")
    return provider if isinstance(provider, str) and provider else None


def _attempt_outcome(attempt: Mapping[str, Any]) -> str | None:
    status = attempt.get("status")
    if status in {"success", "error"}:
        return str(status)
    return None


async def _attach_latency_stats(
    db: AsyncSession,
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
) -> list[RoutingCandidate]:
    candidate_models = {candidate.model for candidate in candidates}
    if not candidate_models:
        return list(candidates)

    sample_limit = _int_config(config.get("latency_sample_limit"), 200)
    min_samples = _int_config(config.get("latency_min_samples"), 1)
    result = await db.execute(
        select(RouteTrace)
        .where(RouteTrace.status == "success")
        .order_by(RouteTrace.timestamp.desc())
        .limit(sample_limit)
    )
    durations_by_model: dict[str, list[float]] = {model: [] for model in candidate_models}
    for trace in result.scalars().all():
        attempts = trace.attempts if isinstance(trace.attempts, list) else []
        for attempt in attempts:
            if not isinstance(attempt, dict) or attempt.get("status") != "success":
                continue
            model_key = _attempt_model_key(attempt)
            duration_ms = _attempt_duration_ms(attempt)
            if model_key in durations_by_model and duration_ms is not None:
                durations_by_model[model_key].append(duration_ms)

    enriched: list[RoutingCandidate] = []
    for candidate in candidates:
        durations = durations_by_model.get(candidate.model, [])
        if len(durations) < min_samples:
            enriched.append(candidate)
            continue
        enriched.append(
            replace(
                candidate,
                average_latency_ms=sum(durations) / len(durations),
                latency_sample_count=len(durations),
            )
        )
    return enriched


def _provider_health_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    health = config.get("health")
    return health if isinstance(health, dict) else {}


def _provider_health_enabled(config: Mapping[str, Any]) -> bool:
    return _bool_config(_provider_health_config(config).get("enabled"), False)


def _provider_health_mode(config: Mapping[str, Any]) -> str:
    mode = _provider_health_config(config).get("mode")
    return mode if isinstance(mode, str) and mode in _HEALTH_MODES else "downrank"


def _provider_health_rate(config: Mapping[str, Any], key: str, default: float) -> float:
    rate = _non_negative_float_or_none(_provider_health_config(config).get(key))
    if rate is None:
        return default
    return min(rate, 1.0)


def _provider_health_from_counts(
    provider: str,
    *,
    success_count: int,
    error_count: int,
    config: Mapping[str, Any],
) -> ProviderHealth:
    sample_count = success_count + error_count
    min_samples = _int_config(_provider_health_config(config).get("min_samples"), 3)
    degraded_rate = _provider_health_rate(config, "degraded_failure_rate", 0.25)
    unhealthy_rate = _provider_health_rate(config, "unhealthy_failure_rate", 0.50)
    failure_rate = None if sample_count == 0 else error_count / sample_count

    if sample_count < min_samples or failure_rate is None:
        return ProviderHealth(
            provider=provider,
            status="unknown",
            sample_count=sample_count,
            success_count=success_count,
            error_count=error_count,
            failure_rate=failure_rate,
            reason="insufficient_samples",
        )
    if failure_rate >= unhealthy_rate:
        return ProviderHealth(
            provider=provider,
            status="unhealthy",
            sample_count=sample_count,
            success_count=success_count,
            error_count=error_count,
            failure_rate=failure_rate,
            reason="failure_rate_exceeds_unhealthy_threshold",
        )
    if failure_rate >= degraded_rate:
        return ProviderHealth(
            provider=provider,
            status="degraded",
            sample_count=sample_count,
            success_count=success_count,
            error_count=error_count,
            failure_rate=failure_rate,
            reason="failure_rate_exceeds_degraded_threshold",
        )
    return ProviderHealth(
        provider=provider,
        status="healthy",
        sample_count=sample_count,
        success_count=success_count,
        error_count=error_count,
        failure_rate=failure_rate,
        reason="failure_rate_below_threshold",
    )


async def _attach_provider_health(
    db: AsyncSession,
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
) -> list[RoutingCandidate]:
    if not _provider_health_enabled(config):
        return list(candidates)

    candidate_providers = {candidate.provider for candidate in candidates}
    if not candidate_providers:
        return list(candidates)

    sample_limit = _int_config(_provider_health_config(config).get("sample_limit"), 200)
    counts_by_provider = {
        provider: {"success": 0, "error": 0}
        for provider in candidate_providers
    }
    result = await db.execute(
        select(RouteTrace)
        .order_by(RouteTrace.timestamp.desc())
        .limit(sample_limit)
    )
    for trace in result.scalars().all():
        attempts = trace.attempts if isinstance(trace.attempts, list) else []
        if attempts:
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                provider = _attempt_provider(attempt)
                outcome = _attempt_outcome(attempt)
                if provider in counts_by_provider and outcome is not None:
                    counts_by_provider[provider][outcome] += 1
            continue

        if trace.selected_provider in counts_by_provider and trace.status in {"success", "error"}:
            counts_by_provider[trace.selected_provider][trace.status] += 1

    health_by_provider = {
        provider: _provider_health_from_counts(
            provider,
            success_count=counts["success"],
            error_count=counts["error"],
            config=config,
        )
        for provider, counts in counts_by_provider.items()
    }
    return [
        replace(candidate, provider_health=health_by_provider.get(candidate.provider))
        for candidate in candidates
    ]


def _apply_provider_health_gate(
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
) -> tuple[list[RoutingCandidate], list[dict[str, Any]]]:
    if not _provider_health_enabled(config) or _provider_health_mode(config) != "skip_unhealthy":
        return list(candidates), []

    allowed: list[RoutingCandidate] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.provider_health is None or candidate.provider_health.status != "unhealthy":
            allowed.append(candidate)
            continue
        rejected.append(
            {
                "model": candidate.model,
                "provider": candidate.provider,
                "reason": "provider_unhealthy",
                "estimated_cost": candidate.estimated_cost,
                "provider_health": candidate.provider_health.to_dict(),
            }
        )
    return allowed, rejected


def _apply_provider_health_order(
    candidates: Sequence[RoutingCandidate],
    *,
    config: Mapping[str, Any],
) -> list[RoutingCandidate]:
    if not _provider_health_enabled(config) or _provider_health_mode(config) != "downrank":
        return list(candidates)
    return sorted(candidates, key=_by_health)


async def _post_external_guardrail_classifier(
    *,
    url: str,
    request_text: str,
    timeout_seconds: float,
    headers: dict[str, str] | None,
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    return await _routing_guardrails.post_external_guardrail_classifier(
        url=url,
        request_text=request_text,
        timeout_seconds=timeout_seconds,
        headers=headers,
    )


async def _evaluate_guardrails(config: Mapping[str, Any], request_body: Mapping[str, Any]) -> dict[str, Any] | None:
    return await _routing_guardrails.evaluate_guardrails(
        config,
        request_body,
        post_classifier=_post_external_guardrail_classifier,
    )


async def _default_policy(db: AsyncSession) -> RoutingPolicy | None:
    result = await db.execute(
        select(RoutingPolicy)
        .where(RoutingPolicy.is_default.is_(True))
        .where(RoutingPolicy.status == ACTIVE_ROUTING_POLICY_STATUS)
        .order_by(RoutingPolicy.updated_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _matching_policy(db: AsyncSession, tags: Mapping[str, str]) -> _PolicyMatch | None:
    if not tags:
        return None

    result = await db.execute(
        select(RoutingPolicy)
        .where(RoutingPolicy.is_default.is_(False))
        .where(RoutingPolicy.status == ACTIVE_ROUTING_POLICY_STATUS)
    )
    policies = result.scalars().all()
    matches: list[_PolicyMatch] = []
    for policy in policies:
        config = policy.config_ or {}
        if not _routing_policy_match.matches_policy_match_config(config, tags):
            continue
        rollout = _routing_policy_match.policy_rollout_info(
            policy_id=policy.policy_id,
            config=config,
            request_tags=tags,
        )
        if rollout is not None and not rollout["matched"]:
            continue
        source = "canary_match" if rollout is not None else "tag_match"
        matches.append(_PolicyMatch(policy=policy, source=source, rollout=rollout))
    if not matches:
        return None

    return sorted(
        matches,
        key=lambda match: (
            _routing_policy_match.policy_match_priority(match.policy.config_ or {}),
            match.policy.updated_at,
        ),
        reverse=True,
    )[0]


async def resolve_routing_plan(
    db: AsyncSession,
    *,
    request_body: Mapping[str, Any],
    project_id: str | None,
    tags: Mapping[str, str] | None,
    policy_id: str | None = None,
    allow_inactive_policy: bool = False,
) -> RoutingPlan:
    """Resolve a routing policy into an ordered model-attempt plan."""
    project: Project | None = None
    policy: RoutingPolicy | None = None
    request_tags = dict(tags or {})
    policy_source = "default"
    policy_rollout: dict[str, Any] | None = None

    if project_id:
        project = await db.get(Project, project_id)
        if project is None:
            raise RoutingPolicyError(404, f"Project '{project_id}' not found")
        if not project.is_active:
            raise RoutingPolicyError(403, f"Project '{project_id}' is inactive")
        if project.routing_policy_id and policy_id is None:
            policy = await db.get(RoutingPolicy, project.routing_policy_id)
            policy_source = "project"

    if policy_id is not None:
        policy = await db.get(RoutingPolicy, policy_id)
        if policy is None:
            raise RoutingPolicyError(404, f"Routing policy '{policy_id}' not found")
        policy_source = "policy_override"
        policy_rollout = _routing_policy_match.policy_rollout_info(
            policy_id=policy.policy_id,
            config=policy.config_ or {},
            request_tags=request_tags,
        )

    if policy is None:
        match = await _matching_policy(db, request_tags)
        if match is not None:
            policy = match.policy
            policy_source = match.source
            policy_rollout = match.rollout

    if policy is None:
        policy = await _default_policy(db)
        policy_source = "default"
        policy_rollout = None
    if policy is None:
        raise RoutingPolicyError(404, "No routing policy is configured")
    if policy.status != ACTIVE_ROUTING_POLICY_STATUS and not allow_inactive_policy:
        raise RoutingPolicyError(422, f"Routing policy '{policy.policy_id}' is not active")

    strategy = policy.strategy
    if strategy not in ROUTING_STRATEGIES:
        raise RoutingPolicyError(422, f"Unsupported routing strategy '{strategy}'")

    config = policy.config_ or {}
    guardrails = await _evaluate_guardrails(config, request_body)
    if guardrails is not None and guardrails["status"] == "blocked":
        first_violation = guardrails["violations"][0] if guardrails["violations"] else {}
        violation_type = first_violation.get("type", "guardrail")
        violation_rule = first_violation.get("rule", "configured_rule")
        raise RoutingPolicyError(
            403,
            f"Routing policy '{policy.policy_id}' guardrail blocked request: {violation_type}:{violation_rule}",
        )
    try:
        specs = _configured_candidate_specs(config)
    except ValueError as exc:
        raise RoutingPolicyError(422, f"Invalid routing policy candidate: {exc}") from exc
    if not specs:
        raise RoutingPolicyError(422, f"Routing policy '{policy.policy_id}' has no candidates")

    redacted_request_body, redactions = apply_guardrail_redactions(config, request_body)
    if redactions is not None:
        if guardrails is None:
            guardrails = {"enabled": True, "status": "passed", "action": _guardrail_action(config), "violations": []}
        guardrails["redactions"] = redactions

    effective_request_body, context = apply_context_policy(config, redacted_request_body)
    prompt_tokens = estimate_prompt_tokens(effective_request_body)
    output_tokens = estimate_output_tokens(effective_request_body)
    target_tier = classify_request_tier(effective_request_body, prompt_tokens=prompt_tokens, config=config)
    try:
        candidates = await _build_candidates(
            db,
            specs,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
    except ValueError as exc:
        raise RoutingPolicyError(422, f"Invalid routing policy candidate: {exc}") from exc
    constrained_candidates, rejected_candidates = _routing_constraints.apply_constraints(
        candidates,
        config=config,
        tags=request_tags,
    )
    candidates = cast(list[RoutingCandidate], constrained_candidates)
    if not candidates:
        detail = f"Routing policy '{policy.policy_id}' has no candidates after constraints"
        if rejected_candidates:
            reasons = sorted({str(item["reason"]) for item in rejected_candidates})
            detail = f"{detail}: {', '.join(reasons)}"
        raise RoutingPolicyError(422, detail)
    candidates = await _attach_provider_health(db, candidates, config=config)
    candidates, health_rejected_candidates = _apply_provider_health_gate(candidates, config=config)
    rejected_candidates.extend(health_rejected_candidates)
    if not candidates:
        detail = f"Routing policy '{policy.policy_id}' has no candidates after provider health gate"
        if health_rejected_candidates:
            reasons = sorted({str(item["reason"]) for item in health_rejected_candidates})
            detail = f"{detail}: {', '.join(reasons)}"
        raise RoutingPolicyError(422, detail)
    if strategy in {"least_latency", "weighted_score"}:
        candidates = await _attach_latency_stats(db, candidates, config=config)
    if strategy == "weighted_score":
        candidates = cast(
            list[RoutingCandidate],
            _routing_weighted_scoring.attach_weighted_scores(candidates, config=config),
        )
    ordered_candidates = _order_candidates(candidates, strategy=strategy, target_tier=target_tier)
    ordered_candidates = _apply_provider_health_order(ordered_candidates, config=config)
    fallback_enabled = _fallback_enabled(config, strategy=strategy)
    if not fallback_enabled:
        ordered_candidates = ordered_candidates[:1]
    if not ordered_candidates:
        raise RoutingPolicyError(422, f"Routing policy '{policy.policy_id}' has no usable candidates")

    reason = f"{policy_source} policy {strategy} selected {ordered_candidates[0].model} for {target_tier} request"
    if not fallback_enabled:
        reason += " with fallback disabled"

    return RoutingPlan(
        requested_model=DEFAULT_ROUTING_MODEL,
        project=project,
        policy=policy,
        strategy=strategy,
        target_tier=target_tier,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        candidates=ordered_candidates,
        fallback_enabled=fallback_enabled,
        policy_source=policy_source,
        reason=reason,
        tags=request_tags,
        rejected_candidates=rejected_candidates,
        policy_rollout=policy_rollout,
        guardrails=guardrails,
        context=context,
    )


async def record_route_trace(
    db: AsyncSession,
    *,
    plan: RoutingPlan,
    api_key_id: str | None,
    user_id: str | None,
    status: str,
    attempts: list[dict[str, Any]],
    error_message: str | None,
    selected_candidate: RoutingCandidate | None,
    endpoint: str = DEFAULT_ROUTE_TRACE_ENDPOINT,
) -> str:
    """Persist a best-effort route trace and return its generated id."""
    trace_id = str(uuid.uuid4())
    candidate = selected_candidate
    trace = RouteTrace(
        trace_id=trace_id,
        api_key_id=api_key_id,
        user_id=user_id,
        project_id=plan.project_id,
        policy_id=plan.policy.policy_id,
        requested_model=plan.requested_model,
        endpoint=endpoint,
        selected_model=candidate.model if candidate else None,
        selected_provider=candidate.provider if candidate else None,
        strategy=plan.strategy,
        status=status,
        error_message=error_message,
        selected_reason=plan.reason if candidate else None,
        estimated_prompt_tokens=plan.prompt_tokens,
        estimated_output_tokens=plan.output_tokens,
        estimated_cost=candidate.estimated_cost if candidate else None,
        fallback_enabled=plan.fallback_enabled,
        policy_source=plan.policy_source,
        tags=plan.tags,
        guardrails=plan.guardrails,
        context=plan.context,
        candidates=[candidate_item.to_trace_dict() for candidate_item in plan.candidates],
        attempts=attempts,
    )
    db.add(trace)
    try:
        await db.commit()
    except SQLAlchemyError as exc:
        await db.rollback()
        logger.warning("Failed to record route trace %s: %s", trace_id, exc)
    return trace_id
