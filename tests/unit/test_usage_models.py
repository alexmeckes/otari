from datetime import UTC, datetime

from gateway.api.routes._usage_models import UsageEntry
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
