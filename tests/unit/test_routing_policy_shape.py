import pytest

from gateway.services import routing_policy_shape
from gateway.services.routing_config_values import float_or_none, score_or_none


def test_number_value_reuses_shared_float_parser() -> None:
    assert routing_policy_shape.number_value is float_or_none
    assert routing_policy_shape.number_value(1.25) == 1.25
    assert routing_policy_shape.number_value(True) is None


def test_score_value_reuses_shared_score_parser() -> None:
    assert routing_policy_shape.score_value is score_or_none
    assert routing_policy_shape.score_value(72) == 0.72
    assert routing_policy_shape.score_value(101) is None


def test_default_strategy_parsing_trims_strategy_axis_provider_and_model() -> None:
    strategy, config = routing_policy_shape.config_from_default_strategy(
        {
            "type": " Intelligent ",
            "axis": " Intelligence ",
            "providers": [{"provider": " openai ", "model": " gpt-4o-mini "}],
        }
    )

    assert strategy == "intelligent"
    assert config["axis"] == "intelligence"
    assert config["candidates"] == [{"model": "openai:gpt-4o-mini"}]


def test_default_strategy_parsing_preserves_controls_and_weighted_scoring_aliases() -> None:
    strategy, config = routing_policy_shape.config_from_default_strategy(
        {
            "type": "weighted_score",
            "providers": [{"provider": "openai", "model": "gpt-4o"}],
            "constraints": {"allowed_providers": ["openai"]},
            "health": {"enabled": True},
            "match": {"tags": {"tier": "prod"}},
            "tier_thresholds": {"medium": 100, "complex": 500, "reasoning": 1000},
            "scoring": {"weights": {"quality": 0.5, "cost": 0.5}},
            "quality_weight": 0.8,
            "unknown_cost_score": 0.25,
        }
    )

    assert strategy == "weighted_score"
    assert config["constraints"] == {"allowed_providers": ["openai"]}
    assert config["health"] == {"enabled": True}
    assert config["match"] == {"tags": {"tier": "prod"}}
    assert config["tier_thresholds"] == {"medium": 100, "complex": 500, "reasoning": 1000}
    assert config["scoring"] == {
        "weights": {"quality": 0.5, "cost": 0.5},
        "quality_weight": 0.8,
        "unknown_cost_score": 0.25,
    }


def test_default_strategy_provider_validation_preserves_error_messages() -> None:
    with pytest.raises(
        routing_policy_shape.RoutingPolicyShapeError,
        match="default_strategy.providers entries must include a non-empty model",
    ):
        routing_policy_shape.config_from_default_strategy(
            {"type": "fallback", "providers": [{"provider": "openai", "model": " "}]}
        )

    with pytest.raises(
        routing_policy_shape.RoutingPolicyShapeError,
        match="default_strategy.providers entries must include a non-empty provider when set",
    ):
        routing_policy_shape.config_from_default_strategy(
            {"type": "fallback", "providers": [{"provider": " ", "model": "gpt-4o"}]}
        )


def test_default_strategy_from_internal_trims_candidate_models_and_skips_blanks() -> None:
    default_strategy = routing_policy_shape.default_strategy_from_internal(
        "priority",
        {
            "candidates": [
                " openai:gpt-4o-mini ",
                " ",
                {"model": " anthropic:claude-3-5-haiku-latest ", "metadata": {"priority": 3}},
            ]
        },
    )

    assert default_strategy is not None
    assert default_strategy["providers"] == [
        {"provider": "openai", "model": "gpt-4o-mini", "priority": 1},
        {"provider": "anthropic", "model": "claude-3-5-haiku-latest", "priority": 3},
    ]


def test_default_strategy_from_internal_forces_single_fallback_disabled() -> None:
    default_strategy = routing_policy_shape.default_strategy_from_internal(
        "single",
        {"candidates": ["openai:gpt-4o"], "fallback_enabled": True},
    )

    assert default_strategy is not None
    assert default_strategy["fallback_enabled"] is False


def test_default_strategy_from_internal_preserves_fallback_enabled_flags_and_defaults() -> None:
    priority = routing_policy_shape.default_strategy_from_internal(
        "priority",
        {"candidates": ["openai:gpt-4o"], "fallback_enabled": False},
    )
    intelligent = routing_policy_shape.default_strategy_from_internal(
        "intelligent",
        {"candidates": ["openai:gpt-4o"]},
    )
    weighted_score = routing_policy_shape.default_strategy_from_internal(
        "weighted_score",
        {"candidates": ["openai:gpt-4o"], "fallback_enabled": False, "scoring": {"quality_weight": 1.0}},
    )
    lowest_cost = routing_policy_shape.default_strategy_from_internal(
        "lowest_cost",
        {"candidates": ["openai:gpt-4o"], "fallback_enabled": False},
    )

    assert priority is not None
    assert intelligent is not None
    assert weighted_score is not None
    assert lowest_cost is not None
    assert priority["fallback_enabled"] is False
    assert intelligent["fallback_enabled"] is True
    assert weighted_score["fallback_enabled"] is False
    assert lowest_cost["fallback_enabled"] is False
