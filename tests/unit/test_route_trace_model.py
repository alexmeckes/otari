from gateway.api.routes._route_trace_models import summarize_route_trace_logs
from gateway.models.entities import RouteTrace


def test_route_trace_dict_helpers_return_dict_values() -> None:
    trace = RouteTrace(
        tags={"team": "platform"},
        guardrails={"blocked": False},
        context={"region": "us"},
    )

    assert trace.tag_dict() == {"team": "platform"}
    assert trace.guardrail_dict() == {"blocked": False}
    assert trace.context_dict() == {"region": "us"}
    assert trace.to_dict()["tags"] == {"team": "platform"}
    assert trace.to_dict()["guardrails"] == {"blocked": False}
    assert trace.to_dict()["context"] == {"region": "us"}


def test_route_trace_list_helpers_return_list_values() -> None:
    candidates = [{"model": "openai:gpt-4o-mini"}]
    attempts = [{"status": "success"}]
    trace = RouteTrace(candidates=candidates, attempts=attempts)

    assert trace.candidate_list() == candidates
    assert trace.attempt_list() == attempts
    assert trace.to_dict()["candidates"] == candidates
    assert trace.to_dict()["attempts"] == attempts


def test_route_trace_dict_helpers_return_empty_dict_for_invalid_values() -> None:
    trace = RouteTrace(tags=None, guardrails=None, context=None)  # type: ignore[arg-type]

    assert trace.tag_dict() == {}
    assert trace.guardrail_dict() == {}
    assert trace.context_dict() == {}
    assert trace.to_dict()["tags"] == {}
    assert trace.to_dict()["guardrails"] == {}
    assert trace.to_dict()["context"] == {}


def test_route_trace_list_helpers_return_empty_list_for_invalid_values() -> None:
    trace = RouteTrace(candidates=None, attempts=None)  # type: ignore[arg-type]

    assert trace.candidate_list() == []
    assert trace.attempt_list() == []
    assert trace.to_dict()["candidates"] == []
    assert trace.to_dict()["attempts"] == []


def test_summarize_route_trace_logs_groups_counts_cost_and_latency() -> None:
    traces = [
        RouteTrace(
            requested_model="gpt-4o",
            selected_model="openai:gpt-4o",
            selected_provider="openai",
            policy_id="policy-a",
            policy_source="default",
            endpoint="/v1/chat/completions",
            strategy="priority",
            status="success",
            estimated_cost=0.001,
            attempts=[{"status": "success", "duration_ms": 40}],
        ),
        RouteTrace(
            requested_model="gpt-4o",
            selected_model="openai:gpt-4o",
            selected_provider="openai",
            policy_id="policy-a",
            policy_source="default",
            endpoint="/v1/chat/completions",
            strategy="priority",
            status="success",
            estimated_cost=0.003,
            attempts=[{"status": "success", "duration_ms": 60}],
        ),
        RouteTrace(
            requested_model="claude",
            selected_model="anthropic:claude-3-5-sonnet-latest",
            selected_provider="anthropic",
            policy_id="policy-b",
            policy_source="canary_match",
            endpoint="/v1/messages",
            strategy="weighted_score",
            status="error",
            estimated_cost=0.0,
            attempts=[{"status": "error", "duration_ms": 100}],
        ),
    ]

    summary = summarize_route_trace_logs(traces)

    assert summary.total_count == 3
    assert summary.success_count == 2
    assert summary.error_count == 1
    assert summary.estimated_cost == 0.004
    assert summary.average_latency_ms == 50.0
    assert [bucket.key for bucket in summary.by_model] == [
        "openai:gpt-4o",
        "anthropic:claude-3-5-sonnet-latest",
    ]
    assert summary.by_model[0].count == 2
    assert summary.by_model[0].average_latency_ms == 50.0
    assert summary.by_model[1].error_count == 1
    assert summary.by_model[1].average_latency_ms is None
    assert [bucket.key for bucket in summary.by_policy] == ["policy-a", "policy-b"]
    assert [bucket.key for bucket in summary.by_policy_source] == ["default", "canary_match"]
    assert [bucket.key for bucket in summary.by_endpoint] == ["/v1/chat/completions", "/v1/messages"]
    assert [bucket.key for bucket in summary.by_provider] == ["openai", "anthropic"]
    assert [bucket.key for bucket in summary.by_strategy] == ["priority", "weighted_score"]


def test_summarize_route_trace_logs_groups_unknown_values() -> None:
    summary = summarize_route_trace_logs(
        [
            RouteTrace(
                requested_model="gpt-4o",
                selected_model=None,
                selected_provider=None,
                policy_id=None,
                policy_source=None,
                endpoint="",
                strategy=None,
                status="error",
                estimated_cost=None,
                attempts=[],
            ),
        ],
    )

    assert summary.total_count == 1
    assert summary.error_count == 1
    assert summary.estimated_cost == 0.0
    assert summary.average_latency_ms is None
    assert [bucket.key for bucket in summary.by_model] == ["unknown"]
    assert [bucket.key for bucket in summary.by_policy] == ["unknown"]
    assert [bucket.key for bucket in summary.by_policy_source] == ["unknown"]
    assert [bucket.key for bucket in summary.by_endpoint] == ["unknown"]
    assert [bucket.key for bucket in summary.by_provider] == ["unknown"]
    assert [bucket.key for bucket in summary.by_strategy] == ["unknown"]
