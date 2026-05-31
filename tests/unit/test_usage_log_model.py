from gateway.models.entities import UsageLog


def test_usage_log_tag_dict_returns_tags_when_dict() -> None:
    log = UsageLog(tags={"team": "platform"})

    assert log.tag_dict() == {"team": "platform"}
    assert log.to_dict()["tags"] == {"team": "platform"}


def test_usage_log_tag_dict_returns_empty_dict_for_invalid_tags() -> None:
    log = UsageLog(tags=None)

    assert log.tag_dict() == {}
    assert log.to_dict()["tags"] == {}
