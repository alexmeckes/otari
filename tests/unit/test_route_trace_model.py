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
