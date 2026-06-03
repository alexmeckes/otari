from gateway.services.routing_policy_service import (
    ROUTING_STRATEGIES,
    RoutingCandidate,
    _no_candidates_detail,
    _order_candidates,
)
from gateway.services.routing_request_analysis import (
    classify_request_tier,
    estimate_output_tokens,
    estimate_prompt_tokens,
    stable_json_text,
)


class _UnserializableToolMarker:
    def __str__(self) -> str:
        return "custom-tool-marker"


def test_classify_request_tier_uses_complexity_hints() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Design an architecture and migration plan for this gateway.",
            }
        ]
    }

    tier = classify_request_tier(request, prompt_tokens=12, config={})

    assert tier == "complex"


def test_classify_request_tier_prefers_reasoning_hints() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Prove this theorem, then outline the architecture migration.",
            }
        ]
    }

    tier = classify_request_tier(request, prompt_tokens=12, config={})

    assert tier == "reasoning"


def test_classify_request_tier_respects_configured_thresholds() -> None:
    request = {"messages": [{"role": "user", "content": "short prompt"}]}

    tier = classify_request_tier(
        request,
        prompt_tokens=120,
        config={"tier_thresholds": {"medium": 100, "complex": 500, "reasoning": 1000}},
    )

    assert tier == "medium"


def test_classify_request_tier_uses_default_for_invalid_thresholds() -> None:
    request = {"messages": [{"role": "user", "content": "short prompt"}]}

    tier = classify_request_tier(
        request,
        prompt_tokens=120,
        config={"tier_thresholds": {"medium": 0, "complex": -1, "reasoning": "100"}},
    )

    assert tier == "simple"


def test_estimates_prompt_and_output_tokens_from_request() -> None:
    request = {
        "messages": [{"role": "user", "content": "abcd" * 20}],
        "max_completion_tokens": 123,
    }

    assert estimate_prompt_tokens(request) >= 20
    assert estimate_output_tokens(request) == 123


def test_stable_json_text_sorts_keys_for_json_payloads() -> None:
    assert stable_json_text({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'


def test_stable_json_text_falls_back_to_jsonable_text_for_unserializable_values() -> None:
    assert stable_json_text({"tool": _UnserializableToolMarker()}) == "custom-tool-marker"


def test_supported_strategies_include_default_strategy_names() -> None:
    assert {
        "single",
        "priority",
        "lowest_cost",
        "least_latency",
        "cost_tier",
        "intelligent",
        "weighted_score",
    } <= ROUTING_STRATEGIES


def _candidate(
    model: str,
    *,
    position: int,
    tier: str | None = None,
    estimated_cost: float | None = None,
    average_latency_ms: float | None = None,
    routing_score: float | None = None,
) -> RoutingCandidate:
    return RoutingCandidate(
        model=model,
        provider="openai",
        provider_model=model,
        position=position,
        tier=tier,
        estimated_cost=estimated_cost,
        input_price_per_million=None,
        output_price_per_million=None,
        quality_score=None,
        average_latency_ms=average_latency_ms,
        latency_sample_count=0,
        routing_score=routing_score,
        score_components=None,
        provider_health=None,
        metadata={},
    )


def test_order_candidates_sorts_unknown_cost_and_latency_last() -> None:
    cost_order = _order_candidates(
        [
            _candidate("unknown", position=1),
            _candidate("cheap", position=2, estimated_cost=0.01),
        ],
        strategy="lowest_cost",
        target_tier="simple",
    )
    latency_order = _order_candidates(
        [
            _candidate("unknown", position=1),
            _candidate("fast", position=2, average_latency_ms=20.0),
        ],
        strategy="least_latency",
        target_tier="simple",
    )

    assert [candidate.model for candidate in cost_order] == ["cheap", "unknown"]
    assert [candidate.model for candidate in latency_order] == ["fast", "unknown"]


def test_order_candidates_sorts_weighted_scores_descending_with_unknown_last() -> None:
    candidates = _order_candidates(
        [
            _candidate("unknown", position=1),
            _candidate("lower", position=2, routing_score=0.2),
            _candidate("higher", position=3, routing_score=0.9),
        ],
        strategy="weighted_score",
        target_tier="simple",
    )

    assert [candidate.model for candidate in candidates] == ["higher", "lower", "unknown"]


def test_order_candidates_uses_target_tier_then_fallback_tiers_and_cost() -> None:
    candidates = _order_candidates(
        [
            _candidate("simple", position=1, tier="simple", estimated_cost=0.01),
            _candidate("medium", position=2, tier="medium", estimated_cost=0.02),
            _candidate("complex-expensive", position=3, tier="complex", estimated_cost=0.03),
            _candidate("complex-cheap", position=4, tier="complex", estimated_cost=0.01),
            _candidate("reasoning", position=5, tier="reasoning", estimated_cost=0.05),
        ],
        strategy="intelligent",
        target_tier="complex",
    )

    assert [candidate.model for candidate in candidates] == [
        "complex-cheap",
        "complex-expensive",
        "reasoning",
        "medium",
        "simple",
    ]


def test_no_candidates_detail_includes_sorted_unique_reasons() -> None:
    detail = _no_candidates_detail(
        "policy-a",
        "provider health gate",
        [
            {"reason": "provider_unhealthy"},
            {"reason": "provider_blocked"},
            {"reason": "provider_unhealthy"},
        ],
    )

    assert detail == (
        "Routing policy 'policy-a' has no candidates after provider health gate: "
        "provider_blocked, provider_unhealthy"
    )


def test_no_candidates_detail_omits_reasons_when_none_rejected() -> None:
    detail = _no_candidates_detail("policy-a", "constraints", [])

    assert detail == "Routing policy 'policy-a' has no candidates after constraints"
