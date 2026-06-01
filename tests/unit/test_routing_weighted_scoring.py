import pytest

from gateway.services.routing_policy_service import RoutingCandidate
from gateway.services.routing_weighted_scoring import attach_weighted_scores


def _candidate(
    *,
    quality_score: float | None = None,
    estimated_cost: float | None = None,
    average_latency_ms: float | None = None,
) -> RoutingCandidate:
    return RoutingCandidate(
        model="openai:gpt-4o-mini",
        provider="openai",
        provider_model="gpt-4o-mini",
        position=1,
        tier=None,
        estimated_cost=estimated_cost,
        input_price_per_million=None,
        output_price_per_million=None,
        quality_score=quality_score,
        average_latency_ms=average_latency_ms,
        latency_sample_count=0,
        routing_score=None,
        score_components=None,
        provider_health=None,
        metadata={},
    )


def test_attach_weighted_scores_uses_default_weights_and_unknown_scores() -> None:
    scored = attach_weighted_scores([_candidate()], config={})

    candidate = scored[0]
    assert candidate.routing_score == pytest.approx(0.35)
    assert candidate.score_components == {
        "quality": 0.5,
        "cost": 0.0,
        "latency": 0.5,
        "quality_weight": 0.5,
        "cost_weight": 0.3,
        "latency_weight": 0.2,
    }


def test_attach_weighted_scores_falls_back_when_configured_weights_sum_to_zero() -> None:
    scored = attach_weighted_scores(
        [_candidate(quality_score=1.0, estimated_cost=1.0, average_latency_ms=20.0)],
        config={"scoring": {"weights": {"quality": 0, "cost": 0, "latency": 0}}},
    )

    assert scored[0].score_components is not None
    assert scored[0].score_components["quality_weight"] == 0.5
    assert scored[0].score_components["cost_weight"] == 0.3
    assert scored[0].score_components["latency_weight"] == 0.2


def test_attach_weighted_scores_uses_score_weights_fallback_when_scoring_is_not_dict() -> None:
    scored = attach_weighted_scores(
        [_candidate(quality_score=1.0, estimated_cost=1.0, average_latency_ms=20.0)],
        config={
            "scoring": "weighted",
            "score_weights": {
                "quality_weight": 1.0,
                "cost_weight": 0.0,
                "latency_weight": 0.0,
            },
        },
    )

    assert scored[0].routing_score == pytest.approx(1.0)
    assert scored[0].score_components is not None
    assert scored[0].score_components["quality_weight"] == 1.0


def test_attach_weighted_scores_prefers_nested_weight_alias_over_top_level_alias() -> None:
    scored = attach_weighted_scores(
        [_candidate(quality_score=1.0)],
        config={
            "scoring": {
                "weights": {
                    "quality_weight": 0.25,
                    "cost_weight": 0.75,
                    "latency_weight": 0.0,
                },
                "quality_weight": 1.0,
                "cost_weight": 0.0,
                "latency_weight": 0.0,
            }
        },
    )

    assert scored[0].routing_score == pytest.approx(0.25)
    assert scored[0].score_components is not None
    assert scored[0].score_components["quality_weight"] == 0.25
    assert scored[0].score_components["cost_weight"] == 0.75


def test_attach_weighted_scores_supports_top_level_metric_name_aliases() -> None:
    scored = attach_weighted_scores(
        [_candidate(quality_score=1.0, estimated_cost=1.0, average_latency_ms=20.0)],
        config={
            "scoring": {
                "quality": 0.0,
                "cost": 1.0,
                "latency": 0.0,
            }
        },
    )

    assert scored[0].routing_score == pytest.approx(1.0)
    assert scored[0].score_components is not None
    assert scored[0].score_components["quality_weight"] == 0.0
    assert scored[0].score_components["cost_weight"] == 1.0
