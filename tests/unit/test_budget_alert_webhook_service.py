from typing import Any

import httpx
import pytest

from gateway.services.budget_alert_webhook_service import _post_budget_alert_webhook


def _patch_async_client_transport(
    handlers: dict[str, httpx.Response | Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handle_request(request: httpx.Request) -> httpx.Response:
        handler = handlers.get(str(request.url))
        if handler is None:
            return httpx.Response(404, text=f"no handler for {request.url}")
        if isinstance(handler, Exception):
            raise handler
        return handler

    transport = httpx.MockTransport(handle_request)
    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


@pytest.mark.asyncio
async def test_post_budget_alert_webhook_trims_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    body = "x" * 600
    _patch_async_client_transport(
        {"https://hooks.example.test/alert": httpx.Response(500, text=body)},
        monkeypatch,
    )

    result = await _post_budget_alert_webhook(
        webhook_url="https://hooks.example.test/alert",
        payload={"event": "budget.threshold_crossed"},
    )

    assert result.status_code == 500
    assert result.error == f"HTTP 500: {body}"[:500]


@pytest.mark.asyncio
async def test_post_budget_alert_webhook_trims_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    error_message = "connection failed " + ("x" * 600)
    _patch_async_client_transport(
        {"https://hooks.example.test/alert": httpx.ConnectError(error_message)},
        monkeypatch,
    )

    result = await _post_budget_alert_webhook(
        webhook_url="https://hooks.example.test/alert",
        payload={"event": "budget.threshold_crossed"},
    )

    assert result.status_code is None
    assert result.error == error_message[:500]
