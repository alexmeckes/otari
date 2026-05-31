import pytest

from gateway.api.routes._chat_streaming_fallback import _first_chunk_timeout_seconds
from gateway.core.config import GatewayConfig


@pytest.mark.parametrize(
    ("platform", "tool_mode", "expected_seconds"),
    [
        ({}, False, 2.0),
        ({}, True, 30.0),
        ({"streaming_first_chunk_timeout_ms": 750}, False, 0.75),
        ({"streaming_first_chunk_timeout_ms_tool_loop": 1250}, True, 1.25),
    ],
)
def test_first_chunk_timeout_seconds_defaults_and_overrides(
    platform: dict[str, int],
    tool_mode: bool,
    expected_seconds: float,
) -> None:
    config = GatewayConfig(mode="platform", platform=platform)

    assert _first_chunk_timeout_seconds(config, tool_mode=tool_mode) == expected_seconds


def test_first_chunk_timeout_seconds_uses_tool_specific_key_only_in_tool_mode() -> None:
    config = GatewayConfig(
        mode="platform",
        platform={
            "streaming_first_chunk_timeout_ms": 1500,
            "streaming_first_chunk_timeout_ms_tool_loop": 7000,
        },
    )

    assert _first_chunk_timeout_seconds(config, tool_mode=False) == 1.5
    assert _first_chunk_timeout_seconds(config, tool_mode=True) == 7.0
