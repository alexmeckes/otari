from gateway.models.entities import Budget


def test_budget_match_tag_dict_returns_match_tags_when_dict() -> None:
    budget = Budget(match_tags={"team": "platform"})

    assert budget.match_tag_dict() == {"team": "platform"}
    assert budget.to_dict()["match_tags"] == {"team": "platform"}


def test_budget_match_tag_dict_returns_empty_dict_for_invalid_match_tags() -> None:
    budget = Budget(match_tags=None)

    assert budget.match_tag_dict() == {}
    assert budget.to_dict()["match_tags"] == {}


def test_budget_alert_threshold_list_returns_thresholds_when_list() -> None:
    budget = Budget(alert_thresholds=[0.5, 0.8])

    assert budget.alert_threshold_list() == [0.5, 0.8]
    assert budget.to_dict()["alert_thresholds"] == [0.5, 0.8]


def test_budget_alert_threshold_list_returns_empty_list_for_invalid_thresholds() -> None:
    budget = Budget(alert_thresholds=None)

    assert budget.alert_threshold_list() == []
    assert budget.to_dict()["alert_thresholds"] == []
