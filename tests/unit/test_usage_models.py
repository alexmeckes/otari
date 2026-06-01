from datetime import UTC, datetime

from gateway.api.routes._usage_models import UsageEntry, summarize_usage_logs
from gateway.api.routes._user_models import UsageLogResponse
from gateway.models.entities import UsageLog


def _usage_log() -> UsageLog:
    return UsageLog(
        id="usage-1",
        user_id="user-1",
        api_key_id="key-1",
        project_id="project-1",
        timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost=0.02,
        status="success",
        tags={"team": "platform"},
    )


def test_user_usage_log_response_reuses_usage_entry_serializer() -> None:
    log = _usage_log()

    assert UsageLogResponse.from_model(log).model_dump() == UsageEntry.from_model(log).model_dump()
    assert isinstance(UsageLogResponse.from_model(log), UsageLogResponse)


def test_summarize_usage_logs_groups_totals_tags_unknowns_and_sorts_buckets() -> None:
    logs = [
        _usage_log(),
        UsageLog(
            id="usage-2",
            user_id="user-2",
            project_id="project-1",
            model="gpt-4o-mini",
            provider="openai",
            endpoint="/v1/responses",
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            cost=0.01,
            status="success",
            tags={"team": "platform", "surface": "responses"},
        ),
        UsageLog(
            id="usage-3",
            model="claude-3-5-haiku-latest",
            endpoint="/v1/messages",
            prompt_tokens=None,
            completion_tokens=1,
            total_tokens=1,
            cost=None,
            status="error",
            tags={"team": "research"},
        ),
    ]

    summary = summarize_usage_logs(logs)

    assert summary.total_count == 3
    assert summary.success_count == 2
    assert summary.error_count == 1
    assert summary.prompt_tokens == 13
    assert summary.completion_tokens == 8
    assert summary.total_tokens == 21
    assert summary.cost == 0.03
    assert [bucket.key for bucket in summary.by_project] == ["project-1", "unknown"]
    assert summary.by_project[0].count == 2
    assert summary.by_project[0].cost == 0.03
    assert summary.by_project[1].error_count == 1
    assert [bucket.key for bucket in summary.by_user] == ["user-1", "user-2", "unknown"]
    assert [bucket.key for bucket in summary.by_tag] == ["team=platform", "surface=responses", "team=research"]
    assert summary.by_tag[0].count == 2
