from gateway.services.routing_policy_match import (
    matches_policy_match_config,
    matches_tag_condition,
    policy_match_bucket,
    policy_match_bucket_key,
    policy_match_priority,
    policy_match_rollout_percentage,
    policy_match_tags,
    policy_rollout_info,
)


def test_policy_match_helpers_read_shared_match_config_values() -> None:
    config = {
        "match": {
            "tags": {"tenant": "vip"},
            "priority": 20,
            "rollout_percentage": 250,
            "bucket_by": "tenant",
            "salt": "canary",
        }
    }
    request_tags = {"tenant": "vip"}

    assert policy_match_tags(config) == {"tenant": "vip"}
    assert policy_match_priority(config) == 20
    assert policy_match_rollout_percentage(config) == 100.0
    assert policy_match_bucket_key(config, request_tags) == "tenant:vip"
    assert policy_match_bucket(config, "policy-a", request_tags) == 47.52
    assert policy_rollout_info(policy_id="policy-a", config=config, request_tags=request_tags)["matched"] is True


def test_policy_match_rollout_alias_and_defaults_are_preserved() -> None:
    assert policy_match_rollout_percentage({"match": {"percentage": 25}}) == 25.0
    assert policy_match_rollout_percentage({"match": {"rollout_percentage": -5}}) == 0.0
    assert policy_match_rollout_percentage({"match": {"rollout_percentage": "25"}}) == 100.0
    assert policy_match_rollout_percentage({}) == 100.0


def test_policy_match_bucket_key_defaults_to_sorted_tags_or_empty() -> None:
    assert (
        policy_match_bucket_key(
            {"match": {"bucket_by": "missing"}},
            {"tenant": "vip", "tier": "prod"},
        )
        == '[["tenant","vip"],["tier","prod"]]'
    )
    assert policy_match_bucket_key({"match": {"bucket_by": "missing"}}, {}) == "__empty__"


def test_policy_match_priority_ignores_bool_values() -> None:
    assert policy_match_priority({"match": {"priority": True}}) == 0


def test_matches_tag_condition_trims_and_lowers_operator() -> None:
    assert matches_tag_condition(
        {"tag": "tier", "operator": " EQ ", "value": "gold"},
        {"tier": "gold"},
    )


def test_matches_tag_condition_preserves_missing_and_non_string_operator_behavior() -> None:
    assert matches_tag_condition(
        {"tag": "tier", "value": "gold"},
        {"tier": "gold"},
    )
    assert not matches_tag_condition(
        {"tag": "tier", "operator": None, "value": "gold"},
        {"tier": "gold"},
    )


def test_matches_policy_match_config_trims_and_lowers_condition_logic() -> None:
    config = {
        "match": {
            "conditions": [
                {"tag": "tier", "value": "gold"},
                {"tag": "team", "value": "platform"},
            ],
            "logic": " OR ",
        }
    }

    assert matches_policy_match_config(config, {"tier": "silver", "team": "platform"})


def test_condition_group_aliases_preserve_order_for_nested_and_policy_configs() -> None:
    condition = {
        "any": [{"tag": "tier", "value": "gold"}],
        "all": [{"tag": "team", "value": "platform"}],
    }
    request_tags = {"team": "platform"}

    assert not matches_tag_condition(condition, request_tags)
    assert not matches_policy_match_config({"match": condition}, request_tags)
    assert matches_tag_condition({"or": [{"tag": "team", "value": "platform"}]}, request_tags)
    assert matches_policy_match_config({"match": {"and": [{"tag": "team", "value": "platform"}]}}, request_tags)


def test_matches_policy_match_config_preserves_non_or_logic_as_and() -> None:
    config = {
        "match": {
            "conditions": [
                {"tag": "tier", "value": "gold"},
                {"tag": "team", "value": "platform"},
            ],
            "logic": True,
        }
    }

    assert not matches_policy_match_config(config, {"tier": "silver", "team": "platform"})
