from gateway.services.mcp_loop import DEFAULT_MAX_TOOL_ITERATIONS, MAX_TOOL_ITERATIONS_CAP


def resolve_max_tool_iterations(requested: int | None) -> int:
    return min(requested or DEFAULT_MAX_TOOL_ITERATIONS, MAX_TOOL_ITERATIONS_CAP)
