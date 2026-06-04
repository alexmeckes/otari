from gateway.models.entities import RoutingPolicy, RoutingPolicyRevision


def test_routing_policy_config_dict_returns_config_copy() -> None:
    config = {"candidates": ["openai:gpt-4o-mini"]}
    policy = RoutingPolicy(config_=config)

    config_copy = policy.config_dict()
    config_copy["candidates"] = []

    assert policy.config_dict() == {"candidates": ["openai:gpt-4o-mini"]}
    assert policy.to_dict()["config"] == {"candidates": ["openai:gpt-4o-mini"]}


def test_routing_policy_config_dict_returns_empty_dict_for_invalid_config() -> None:
    policy = RoutingPolicy(config_=None)  # type: ignore[arg-type]

    assert policy.config_dict() == {}
    assert policy.to_dict()["config"] == {}


def test_routing_policy_revision_config_dict_returns_config_copy() -> None:
    config = {"candidates": ["anthropic:claude-3-5-haiku-latest"]}
    revision = RoutingPolicyRevision(config_=config)

    config_copy = revision.config_dict()
    config_copy["candidates"] = []

    assert revision.config_dict() == {"candidates": ["anthropic:claude-3-5-haiku-latest"]}
    assert revision.to_dict()["config"] == {"candidates": ["anthropic:claude-3-5-haiku-latest"]}


def test_routing_policy_revision_config_dict_returns_empty_dict_for_invalid_config() -> None:
    revision = RoutingPolicyRevision(config_=None)  # type: ignore[arg-type]

    assert revision.config_dict() == {}
    assert revision.to_dict()["config"] == {}
