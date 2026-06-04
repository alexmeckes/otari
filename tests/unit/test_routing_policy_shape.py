import pytest

from gateway.services import routing_policy_shape


def test_default_strategy_provider_priority_rejects_bool_values() -> None:
    _strategy, config = routing_policy_shape.config_from_default_strategy(
        {
            "type": "fallback",
            "providers": [
                {"provider": "anthropic", "model": "claude-3-5-sonnet-latest", "priority": 1.5},
                {"provider": "openai", "model": "gpt-4o", "priority": True},
            ],
        }
    )

    assert [candidate["model"] for candidate in config["candidates"]] == [
        "anthropic:claude-3-5-sonnet-latest",
        "openai:gpt-4o",
    ]


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


def test_default_strategy_candidate_ordering_preserves_strategy_semantics() -> None:
    providers = [
        {"provider": "openai", "model": "gpt-4o", "priority": 2},
        {"provider": "anthropic", "model": "claude-3-5-sonnet-latest", "priority": 1},
    ]

    fallback_strategy, fallback_config = routing_policy_shape.config_from_default_strategy(
        {"type": "fallback", "providers": providers}
    )
    weighted_strategy, weighted_config = routing_policy_shape.config_from_default_strategy(
        {"type": "weighted_score", "providers": providers}
    )
    intelligent_strategy, intelligent_config = routing_policy_shape.config_from_default_strategy(
        {"type": "intelligent", "providers": providers}
    )

    assert fallback_strategy == "priority"
    assert [candidate["model"] for candidate in fallback_config["candidates"]] == [
        "anthropic:claude-3-5-sonnet-latest",
        "openai:gpt-4o",
    ]
    assert weighted_strategy == "weighted_score"
    assert intelligent_strategy == "intelligent"
    assert [candidate["model"] for candidate in weighted_config["candidates"]] == [
        "openai:gpt-4o",
        "anthropic:claude-3-5-sonnet-latest",
    ]
    assert [candidate["model"] for candidate in intelligent_config["candidates"]] == [
        "openai:gpt-4o",
        "anthropic:claude-3-5-sonnet-latest",
    ]


@pytest.mark.parametrize("strategy_type", ["fallback", "weighted_score", "intelligent"])
def test_default_strategy_candidate_construction_preserves_provider_fields(strategy_type: str) -> None:
    _strategy, config = routing_policy_shape.config_from_default_strategy(
        {
            "type": strategy_type,
            "providers": [
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "priority": 2,
                    "tier": "medium",
                    "input_price_per_million": 0.15,
                    "output_price_per_million": 0.6,
                    "region": "us",
                }
            ],
        }
    )

    assert config["candidates"] == [
        {
            "model": "openai:gpt-4o-mini",
            "tier": "medium",
            "input_price_per_million": 0.15,
            "output_price_per_million": 0.6,
            "metadata": {"priority": 2, "region": "us"},
        }
    ]


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


@pytest.mark.parametrize(
    ("providers", "message"),
    [
        (None, "default_strategy.providers must be a non-empty list"),
        ([], "default_strategy.providers must be a non-empty list"),
        ("openai", "default_strategy.providers must be a non-empty list"),
        (["openai:gpt-4o"], "default_strategy.providers entries must be objects"),
    ],
)
def test_default_strategy_provider_list_validation_preserves_error_messages(
    providers: object,
    message: str,
) -> None:
    default_strategy: dict[str, object] = {"type": "fallback"}
    if providers is not None:
        default_strategy["providers"] = providers

    with pytest.raises(routing_policy_shape.RoutingPolicyShapeError, match=message):
        routing_policy_shape.config_from_default_strategy(default_strategy)


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


def test_default_strategy_from_internal_preserves_tier_candidates() -> None:
    default_strategy = routing_policy_shape.default_strategy_from_internal(
        "priority",
        {
            "candidates": ["openai:gpt-4o"],
            "tiers": {
                "complex": [
                    {
                        "model": " anthropic:claude-3-5-sonnet-latest ",
                        "metadata": {"priority": 7, "region": "us"},
                        "input_price_per_million": 3.0,
                        "output_price_per_million": 15.0,
                    }
                ],
                1: ["openai:gpt-4o-mini"],
                "simple": "openai:gpt-4o-mini",
            },
        },
    )

    assert default_strategy is not None
    assert default_strategy["providers"] == [
        {"provider": "openai", "model": "gpt-4o", "priority": 1},
        {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-latest",
            "priority": 7,
            "tier": "complex",
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
            "region": "us",
        },
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
