"""Unit tests for MCPClientPool result flattening."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gateway.services.mcp_client import MCPClientPool


class _FakeSession:
    def __init__(self, content: list[Any], *, is_error: bool = False) -> None:
        self._content = content
        self._is_error = is_error
        self.called_with: tuple[str, dict[str, Any]] | None = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.called_with = (name, arguments)
        return SimpleNamespace(content=self._content, isError=self._is_error)


def _pool_with_result(content: list[Any], *, is_error: bool = False) -> tuple[MCPClientPool, _FakeSession]:
    pool = MCPClientPool([])
    session = _FakeSession(content, is_error=is_error)
    pool._tool_owner["lookup"] = "server"
    pool._servers["server"] = SimpleNamespace(session=session)
    return pool, session


@pytest.mark.asyncio
async def test_call_tool_flattens_text_blocks_and_skips_blank_text() -> None:
    pool, session = _pool_with_result(
        [
            SimpleNamespace(text=""),
            SimpleNamespace(text="hello"),
            SimpleNamespace(text="world"),
        ]
    )

    result = await pool.call_tool("lookup", {"query": "docs"})

    assert result == "hello\nworld"
    assert session.called_with == ("lookup", {"query": "docs"})


@pytest.mark.asyncio
async def test_call_tool_summarizes_image_resource_and_unknown_blocks() -> None:
    pool, _session = _pool_with_result(
        [
            SimpleNamespace(type="image", mimeType="image/png", data="abcd"),
            SimpleNamespace(type="embedded_resource", resource=SimpleNamespace(uri="file://result.txt")),
            SimpleNamespace(type="json_blob"),
        ]
    )

    result = await pool.call_tool("lookup", {})

    assert result == "\n".join(
        [
            "[image type=image/png bytes_b64=4]",
            "[resource uri=file://result.txt]",
            "[content type=json_blob]",
        ]
    )


@pytest.mark.asyncio
async def test_call_tool_prefixes_error_results() -> None:
    pool, _session = _pool_with_result([SimpleNamespace(text="failed")], is_error=True)

    result = await pool.call_tool("lookup", {})

    assert result == "[tool error] failed"
