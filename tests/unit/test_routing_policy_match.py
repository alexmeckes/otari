from gateway.services.routing_policy_match import matches_policy_match_config, matches_tag_condition


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
