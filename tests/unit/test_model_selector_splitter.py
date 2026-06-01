from typing import Any

import httpx
import pytest

from gateway.api.routes._model_catalog import _canonical_model_id, _stored_model_key
from gateway.api.routes._responses_transform import metadata_from_model_selector, served_metadata
from gateway.core.config import GatewayConfig
from gateway.services import platform_gateway
from gateway.services.routing_policy_shape import model_selector, split_model_selector


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "gpt-4o", "openai:gpt-4o"),
        ("openai", "openai/gpt-4o", "openai:gpt-4o"),
        (None, "openai/gpt-4o", "openai:gpt-4o"),
        (None, "gpt-4o", "gpt-4o"),
        (" openai ", " gpt-4o ", "openai:gpt-4o"),
        (" ", " gpt-4o ", "gpt-4o"),
    ],
)
def test_model_selector_preserves_existing_selector_shapes(
    provider: str | None,
    model: str,
    expected: str,
) -> None:
    assert model_selector(provider, model) == expected


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("openai:gpt-4o", ("openai", "gpt-4o")),
        ("openai/gpt-4o", ("openai", "gpt-4o")),
        ("gpt-4o", (None, "gpt-4o")),
        (":gpt-4o", (None, "gpt-4o")),
    ],
)
def test_split_model_selector_preserves_existing_selector_shapes(
    selector: str,
    expected: tuple[str | None, str],
) -> None:
    assert split_model_selector(selector) == expected


def test_responses_metadata_uses_shared_model_selector_splitter() -> None:
    assert served_metadata("openai", "gpt-4o") == {
        "model": "openai/gpt-4o",
        "vendor": "openai",
    }
    assert metadata_from_model_selector("openai:gpt-4o") == {
        "model": "openai/gpt-4o",
        "vendor": "openai",
    }
    assert metadata_from_model_selector("gpt-4o") is None


def test_model_catalog_helpers_use_shared_model_selector_splitter() -> None:
    assert _canonical_model_id("openai:gpt-4o") == "openai/gpt-4o"
    assert _canonical_model_id("openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert _canonical_model_id("gpt-4o") == "unknown/gpt-4o"
    assert _stored_model_key("openai/gpt-4o") == "openai:gpt-4o"


@pytest.mark.asyncio
async def test_platform_resolution_uses_shared_model_selector_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve_bodies: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        resolve_bodies.append(body)
        return httpx.Response(
            200,
            json={
                "request_id": "request-1",
                "fallback_enabled": False,
                "attempts": [
                    {
                        "attempt_id": "attempt-1",
                        "position": 0,
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-test",
                        "managed": True,
                    }
                ],
            },
        )

    monkeypatch.setattr(platform_gateway, "_post_platform", fake_post_platform)

    route = await platform_gateway.resolve_platform_credentials(
        GatewayConfig(mode="platform", platform={"base_url": "http://platform.test/api/v1"}),
        "user-token",
        "openai/gpt-4o",
    )

    assert resolve_bodies == [{"model": "gpt-4o", "provider": "openai"}]
    assert route.attempts[0].provider == "openai"
    assert route.attempts[0].model == "gpt-4o"
