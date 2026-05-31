from gateway.api.routes._chat_tool_iterations import resolve_max_tool_iterations
from gateway.services.mcp_loop import DEFAULT_MAX_TOOL_ITERATIONS, MAX_TOOL_ITERATIONS_CAP


def test_resolve_max_tool_iterations_uses_default_when_missing() -> None:
    assert resolve_max_tool_iterations(None) == DEFAULT_MAX_TOOL_ITERATIONS


def test_resolve_max_tool_iterations_uses_request_value() -> None:
    assert resolve_max_tool_iterations(3) == 3


def test_resolve_max_tool_iterations_caps_request_value() -> None:
    assert resolve_max_tool_iterations(MAX_TOOL_ITERATIONS_CAP + 1) == MAX_TOOL_ITERATIONS_CAP
