from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from gateway.services import routing_policy_shape
from gateway.services.routing_config_values import dict_or_empty, score_or_none, string_or_none
from gateway.services.routing_quality_scores import candidate_quality_score


class RoutingPolicyEvalScoreError(ValueError):
    """Raised when eval scores cannot be applied to a routing policy config."""


@dataclass(frozen=True)
class EvalScoreInput:
    model: str
    provider: str | None = None
    score: float | None = None
    quality_score: float | None = None
    benchmark_score: float | None = None
    metric: str | None = None
    sample_count: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalScoreAggregate:
    model: str
    quality_score: float
    sample_count: int
    metrics: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AppliedEvalScore:
    model: str
    previous_quality_score: float | None
    quality_score: float
    sample_count: int
    metrics: list[str]


@dataclass(frozen=True)
class EvalScoreApplication:
    config: dict[str, Any]
    applied_scores: list[AppliedEvalScore]
    unmatched_models: list[str]


def aggregate_eval_scores(items: Iterable[EvalScoreInput]) -> dict[str, EvalScoreAggregate]:
    totals: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    metrics: dict[str, set[str]] = {}
    metadata_by_model: dict[str, dict[str, Any]] = {}

    for item in items:
        model_key = routing_policy_shape.normalized_model_selector(item.provider, item.model)
        score: float | None = None
        for value in (item.quality_score, item.score, item.benchmark_score):
            score = score_or_none(value)
            if score is not None:
                break
        if score is None:
            raise RoutingPolicyEvalScoreError("Each eval score must include score, quality_score, or benchmark_score")
        sample_count = item.sample_count or 1
        totals[model_key] = totals.get(model_key, 0.0) + (score * sample_count)
        sample_counts[model_key] = sample_counts.get(model_key, 0) + sample_count
        if item.metric is not None:
            metrics.setdefault(model_key, set()).add(item.metric)
        if item.metadata:
            metadata = metadata_by_model.setdefault(model_key, {})
            metadata.update(item.metadata)

    return {
        model_key: EvalScoreAggregate(
            model=model_key,
            quality_score=totals[model_key] / sample_counts[model_key],
            sample_count=sample_counts[model_key],
            metrics=sorted(metrics.get(model_key, set())),
            metadata=metadata_by_model.get(model_key, {}),
        )
        for model_key in totals
    }


def _apply_eval_score_to_candidate(
    candidate: Any,
    *,
    scores_by_model: Mapping[str, EvalScoreAggregate],
    applied_scores: list[AppliedEvalScore],
    applied_model_keys: set[str],
    updated_at: str,
) -> Any:
    model_key: str | None
    if isinstance(candidate, str):
        model_key = routing_policy_shape.normalized_model_selector(None, candidate)
    elif isinstance(candidate, dict):
        model = string_or_none(candidate.get("model"))
        model_key = routing_policy_shape.normalized_model_selector(None, model) if model is not None else None
    else:
        model_key = None
    if model_key is None or model_key not in scores_by_model:
        return candidate

    aggregate = scores_by_model[model_key]
    previous_quality_score = candidate_quality_score(candidate) if isinstance(candidate, dict) else None
    updated_candidate = {"model": candidate} if isinstance(candidate, str) else dict(candidate)
    metadata = dict_or_empty(updated_candidate.get("metadata"), copy_value=True)
    metadata["eval_score"] = {
        "quality_score": aggregate.quality_score,
        "sample_count": aggregate.sample_count,
        "metrics": aggregate.metrics,
        "metadata": aggregate.metadata,
        "updated_at": updated_at,
    }
    updated_candidate["quality_score"] = aggregate.quality_score
    updated_candidate["metadata"] = metadata
    applied_model_keys.add(model_key)
    applied_scores.append(
        AppliedEvalScore(
            model=model_key,
            previous_quality_score=previous_quality_score,
            quality_score=aggregate.quality_score,
            sample_count=aggregate.sample_count,
            metrics=aggregate.metrics,
        )
    )
    return updated_candidate


def apply_eval_scores_to_policy_config(
    config: Mapping[str, Any],
    scores: Iterable[EvalScoreInput],
    *,
    updated_at: str | None = None,
) -> EvalScoreApplication:
    scores_by_model = aggregate_eval_scores(scores)
    updated_config = dict(config)
    applied_scores: list[AppliedEvalScore] = []
    applied_model_keys: set[str] = set()
    updated_at_value = updated_at or datetime.now(UTC).isoformat()

    candidates = updated_config.get("candidates")
    if isinstance(candidates, list):
        updated_config["candidates"] = [
            _apply_eval_score_to_candidate(
                candidate,
                scores_by_model=scores_by_model,
                applied_scores=applied_scores,
                applied_model_keys=applied_model_keys,
                updated_at=updated_at_value,
            )
            for candidate in candidates
        ]

    tiers = updated_config.get("tiers")
    if isinstance(tiers, dict):
        updated_tiers: dict[str, Any] = {}
        for tier_name, tier_candidates in tiers.items():
            if not isinstance(tier_candidates, list):
                updated_tiers[str(tier_name)] = tier_candidates
                continue
            updated_tiers[str(tier_name)] = [
                _apply_eval_score_to_candidate(
                    candidate,
                    scores_by_model=scores_by_model,
                    applied_scores=applied_scores,
                    applied_model_keys=applied_model_keys,
                    updated_at=updated_at_value,
                )
                for candidate in tier_candidates
            ]
        updated_config["tiers"] = updated_tiers

    unmatched_models = sorted(set(scores_by_model) - applied_model_keys)
    return EvalScoreApplication(
        config=updated_config,
        applied_scores=applied_scores,
        unmatched_models=unmatched_models,
    )
